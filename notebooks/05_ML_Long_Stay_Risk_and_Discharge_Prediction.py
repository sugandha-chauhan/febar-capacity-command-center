# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 05 - ML Long Stay Risk and Discharge Prediction
# MAGIC
# MAGIC Makes the bed board **intelligent**. Two models learn from 2 years of discharged inpatient stays, and a GenAI step writes huddle notes:
# MAGIC
# MAGIC | Model | Predicts | Used for |
# MAGIC | --- | --- | --- |
# MAGIC | `long_stay_risk_model` (classifier) | Probability that a stay runs **2+ days past its expected LOS** | Ranking who case managers work first |
# MAGIC | `los_days_model` (regressor, log-LOS) | Total length of stay in days | Predicted discharge date, and how many beds open today |
# MAGIC | AI huddle note (`ai_query`, Foundation Model API) | A 2-sentence "why at risk / next step" note for high-risk patients, from de-identified facts only | The app's huddle view |
# MAGIC
# MAGIC **Prediction time:** the models use only what is known **at admission** (age, DRG and expected LOS, unit, admit source, payer, complex-discharge screen, living situation, prior admissions). CarePort barriers are **not** model inputs: they are usually raised near the end of the stay, so using them would leak the outcome. The app shows open barriers next to the score instead.
# MAGIC
# MAGIC **Steps:** build features from gold, quick EDA, time-based split (train on admissions before May 2026, test on May-Aug 2026), train and evaluate with MLflow tracking, refit on all data, validate the artifacts in an isolated environment, register both models in Unity Catalog with the `@champion` alias, then score everyone in the hospital now.
# MAGIC
# MAGIC **Outputs in** `<prefix>_gold`: `inpatient_predictions`, `huddle_notes`, `bed_board` (the scored, patient-level operational table synced to Lakebase in 06) and `bed_board_deid` (no names or MRNs, for Genie).
# MAGIC
# MAGIC **Prospective check:** notebook 01 saved the simulation's answer key, so the last step compares the predictions for patients still in the hospital with what actually happens to them after 7:00 AM.
# MAGIC
# MAGIC **How to run (about 8 minutes, CPU serverless is fine):** run notebooks 03 and 04 first, then **Run all**.

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "schauhan_workspace_catalog", "1. Catalog")
dbutils.widgets.text("prefix", "lsh_capacity", "2. Schema prefix")
dbutils.widgets.text("llm_endpoint", "databricks-claude-haiku-4-5", "3. LLM endpoint for huddle notes")

# COMMAND ----------

# DBTITLE 1,Step 1 - Setup and features (same feature query for training and scoring)
import re, numpy as np, pandas as pd, mlflow
from pyspark.sql import functions as F

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("prefix").strip()
LLM = dbutils.widgets.get("llm_endpoint").strip()
for label, val in [("catalog", CATALOG), ("prefix", PREFIX)]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", val or ""):
        raise ValueError(f"Enter a {label} with only letters, numbers and underscores.")
spark.conf.set("spark.sql.session.timeZone", "UTC")
GOLD = f"`{CATALOG}`.`{PREFIX}_gold`"
GOLD_NAME = f"{CATALOG}.{PREFIX}_gold"
RAW = f"`{CATALOG}`.`{PREFIX}_raw`"
CLF_NAME, REG_NAME = f"{GOLD_NAME}.long_stay_risk_model", f"{GOLD_NAME}.los_days_model"
if not spark.catalog.tableExists(f"{GOLD_NAME}.current_inpatients"):
    raise ValueError("Gold tables not found. Run notebooks 03 and 04 first.")

ME = spark.sql("SELECT current_user()").first()[0]
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{ME}/LSH Capacity and LOS Prototype/{PREFIX}_los_models")
SNAP = spark.sql(f"SELECT max(snapshot_ts) AS s FROM {GOLD}.current_inpatients").first().s

CATS = ["facility_code", "unit_type", "drg", "service_line", "sex", "admit_source", "fin_class", "admitted_from"]
NUMS = ["age_at_admit", "expected_los", "complex_discharge", "lives_alone", "is_surgical", "admit_dow", "admit_hour",
        "prior_admits_12m", "prior_long_stay"]
FEATURES = CATS + NUMS

def features_sql(where):
    """Admission-time features for every encounter matching `where`. Used for both training and scoring."""
    return f"""
    WITH e AS (
      SELECT x.encounter_id, x.patient_key, x.admit_ts, x.discharge_ts, x.patient_class, x.admit_source, x.fin_class,
             x.admitted_from, x.complex_discharge, x.lives_alone, x.age_at_admit, x.expected_los, x.los_days, x.is_long_stay,
             x.is_in_house, f.facility_code, u.unit_type, d.drg, d.service_line, d.is_surgical, p.sex
      FROM {GOLD}.fact_encounter x JOIN {GOLD}.dim_facility f USING (facility_key) JOIN {GOLD}.dim_unit u USING (unit_key)
      JOIN {GOLD}.dim_drg d USING (drg_key) JOIN {GOLD}.dim_patient p USING (patient_key)),
    hist AS (   -- the patient's history before this admission (no look-ahead)
      SELECT a.encounter_id, count(b.encounter_id) AS prior_admits_12m, coalesce(max(b.is_long_stay), 0) AS prior_long_stay
      FROM e a LEFT JOIN e b ON a.patient_key = b.patient_key AND b.discharge_ts <= a.admit_ts
                             AND b.admit_ts >= a.admit_ts - INTERVAL 365 DAYS
      GROUP BY a.encounter_id)
    SELECT e.encounter_id, e.admit_ts, e.discharge_ts, e.patient_class, e.los_days, e.is_long_stay,
           {', '.join(f'CAST(e.{c} AS STRING) AS {c}' for c in CATS if c not in ('admit_dow',))},
           CAST(e.age_at_admit AS DOUBLE) AS age_at_admit, CAST(e.expected_los AS DOUBLE) AS expected_los,
           CAST(e.complex_discharge AS DOUBLE) AS complex_discharge, CAST(e.lives_alone AS DOUBLE) AS lives_alone,
           CAST(e.is_surgical AS DOUBLE) AS is_surgical, CAST(dayofweek(e.admit_ts) AS DOUBLE) AS admit_dow,
           CAST(hour(e.admit_ts) AS DOUBLE) AS admit_hour, CAST(h.prior_admits_12m AS DOUBLE) AS prior_admits_12m,
           CAST(h.prior_long_stay AS DOUBLE) AS prior_long_stay
    FROM e JOIN hist h USING (encounter_id) WHERE {where}"""

# Discharged inpatient stays admitted in FY2025-FY2026 (Aug 2024 admissions are skipped: only their long stays
# reach the data window, which would bias the label).
data = spark.sql(features_sql("e.patient_class = 'Inpatient' AND e.discharge_ts IS NOT NULL "
                              "AND e.admit_ts >= '2024-09-01' AND e.admit_ts < '2026-09-01'")).toPandas()
print(f"Training window: {len(data):,} discharged inpatient stays; snapshot {SNAP}")

# COMMAND ----------

# DBTITLE 1,Step 2 - Exploratory data analysis
import matplotlib.pyplot as plt

display(pd.DataFrame({"feature": FEATURES,
                      "type": ["categorical / nominal"] * len(CATS) + ["numeric float", "numeric float", "binary flag", "binary flag",
                               "binary flag", "categorical / ordinal (day of week)", "numeric integer (hour)", "numeric integer (count)", "binary flag"]}))
display(data[NUMS + ["los_days", "is_long_stay"]].describe().T.reset_index())
share = data["is_long_stay"].value_counts(normalize=True).sort_index()
print(f"Long stay (2+ days past expected): {share.get(1, 0):.1%} of stays; not long: {share.get(0, 0):.1%}")

fig, ax = plt.subplots(1, 3, figsize=(17, 4))
share.rename({0: "Not long", 1: "Long stay"}).plot.bar(ax=ax[0], color=["#4E2A84", "#E4002B"], title="Label balance")
band = pd.cut(data["age_at_admit"], [17, 44, 64, 74, 120], labels=["18-44", "45-64", "65-74", "75+"])
data.groupby([band, "complex_discharge"], observed=True)["is_long_stay"].mean().unstack().rename(columns={0.0: "Not complex", 1.0: "Complex discharge"}) \
    .plot.bar(ax=ax[1], title="Long-stay rate by age band and complex discharge screen", rot=0)
data.groupby("fin_class")["is_long_stay"].mean().sort_values().plot.barh(ax=ax[2], title="Long-stay rate by financial class", color="#4E2A84")
plt.tight_layout(); plt.show()

num_corr = data[NUMS + ["is_long_stay", "los_days"]].corr()[["is_long_stay", "los_days"]].drop(["is_long_stay", "los_days"]).round(3)
display(num_corr.reset_index().rename(columns={"index": "feature"}))

# COMMAND ----------

# DBTITLE 1,Step 3 - Time-based split, train, evaluate (tracked in MLflow)
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss, brier_score_loss, f1_score,
                             accuracy_score, confusion_matrix, mean_absolute_error, RocCurveDisplay)
from sklearn.calibration import CalibrationDisplay

# The admit time is real event time and scoring predicts future stays, so hold out the latest admissions.
CUTOFF = "2026-05-01"
train, test = data[data["admit_ts"] < CUTOFF], data[data["admit_ts"] >= CUTOFF]
print(f"Temporal split: train = admissions before {CUTOFF} ({len(train):,}), test = May-Aug 2026 ({len(test):,})")

prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATS),
                          ("num", SimpleImputer(strategy="mean"), NUMS)])
CLF_PARAMS = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
REG_PARAMS = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
make_clf = lambda: Pipeline([("prep", clone(prep)), ("model", HistGradientBoostingClassifier(**CLF_PARAMS))])
make_reg = lambda: TransformedTargetRegressor(regressor=Pipeline([("prep", clone(prep)), ("model", HistGradientBoostingRegressor(**REG_PARAMS))]),
                                              func=np.log1p, inverse_func=np.expm1)   # log-LOS inside the model

clf = make_clf().fit(train[FEATURES], train["is_long_stay"])
reg = make_reg().fit(train[FEATURES], train["los_days"])
p = clf.predict_proba(test[FEATURES])[:, 1]
yhat = (p >= 0.5).astype(int)
los_hat = reg.predict(test[FEATURES])
CLF_METRICS = {"test_roc_auc": roc_auc_score(test["is_long_stay"], p), "test_pr_auc": average_precision_score(test["is_long_stay"], p),
               "test_log_loss": log_loss(test["is_long_stay"], p), "test_brier": brier_score_loss(test["is_long_stay"], p),
               "test_f1": f1_score(test["is_long_stay"], yhat), "test_accuracy": accuracy_score(test["is_long_stay"], yhat),
               "baseline_auc_complex_flag_only": roc_auc_score(test["is_long_stay"], test["complex_discharge"])}
REG_METRICS = {"test_mae_days": mean_absolute_error(test["los_days"], los_hat),
               "test_median_ae_days": float(np.median(np.abs(test["los_days"] - los_hat))),
               "test_within_1_day": float(np.mean(np.abs(test["los_days"] - los_hat) <= 1)),
               "baseline_mae_expected_los": mean_absolute_error(test["los_days"], test["expected_los"]),
               # spread of actual LOS around the prediction (log scale); used to turn total LOS into remaining LOS in Step 5
               "log_residual_sigma": float(np.std(np.log(test["los_days"]) - np.log(los_hat)))}

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
RocCurveDisplay.from_predictions(test["is_long_stay"], p, ax=ax[0], name="Long-stay risk")
CalibrationDisplay.from_predictions(test["is_long_stay"], p, n_bins=10, ax=ax[1], name="Long-stay risk")
plt.tight_layout()
with mlflow.start_run(run_name="long_stay_risk_holdout"):
    mlflow.log_params({**CLF_PARAMS, "split": f"temporal, test = admits >= {CUTOFF}", "features": ",".join(FEATURES)})
    mlflow.log_metrics(CLF_METRICS)
    mlflow.log_figure(fig, "roc_and_calibration.png")
with mlflow.start_run(run_name="los_days_holdout"):
    mlflow.log_params({**REG_PARAMS, "target_transform": "log1p", "split": f"temporal, test = admits >= {CUTOFF}"})
    mlflow.log_metrics(REG_METRICS)
plt.show()
print("Confusion matrix at 0.5 (rows = actual, cols = predicted):\n", confusion_matrix(test["is_long_stay"], yhat))
display(pd.DataFrame([{**{k: round(v, 3) for k, v in CLF_METRICS.items()}}]))
display(pd.DataFrame([{**{k: round(v, 3) for k, v in REG_METRICS.items()}}]))

# COMMAND ----------

# DBTITLE 1,Step 4 - Refit on all data, log, validate in isolation, register in UC as @champion
from mlflow.models import infer_signature
from mlflow import MlflowClient

X_all = data[FEATURES]
clf_final = make_clf().fit(X_all, data["is_long_stay"])
reg_final = make_reg().fit(X_all, data["los_days"])
example = X_all.head(3)

with mlflow.start_run(run_name="long_stay_risk_final") as run:
    mlflow.log_params({**CLF_PARAMS, "n_train": len(train), "n_test": len(test), "n_final_fit": len(data)})
    mlflow.log_metrics(CLF_METRICS)
    clf_info = mlflow.sklearn.log_model(
        clf_final, name="model", pyfunc_predict_fn="predict_proba",         # pyfunc returns [P(not long), P(long)]
        signature=infer_signature(X_all.head(100), clf_final.predict_proba(X_all.head(100))), input_example=example)
with mlflow.start_run(run_name="los_days_final") as run:
    mlflow.log_params({**REG_PARAMS, "n_train": len(train), "n_test": len(test), "n_final_fit": len(data)})
    mlflow.log_metrics(REG_METRICS)
    reg_info = mlflow.sklearn.log_model(
        reg_final, name="model", signature=infer_signature(X_all.head(100), reg_final.predict(X_all.head(100))), input_example=example)

# Serving-readiness check: load each artifact in a fresh environment built only from its declared requirements
for info in [clf_info, reg_info]:
    pyfunc_model = mlflow.pyfunc.load_model(info.model_uri)
    assert pyfunc_model.metadata.signature is not None and pyfunc_model.input_example is not None
    mlflow.models.predict(model_uri=info.model_uri, input_data=pyfunc_model.input_example, env_manager="uv")
print("Both artifacts load and predict in an isolated environment")

client = MlflowClient(registry_uri="databricks-uc")
VERSIONS = {}
for name, info, desc in [(CLF_NAME, clf_info, "Probability that an inpatient stay runs 2+ days past expected LOS (admission-time features). Synthetic LSH data."),
                         (REG_NAME, reg_info, "Predicted total inpatient length of stay in days (admission-time features). Synthetic LSH data.")]:
    mv = mlflow.register_model(model_uri=info.model_uri, name=name, await_registration_for=300)
    client.update_registered_model(name=name, description=desc)
    client.set_registered_model_alias(name=name, alias="champion", version=mv.version)
    VERSIONS[name] = mv.version
    print(f"Registered {name} version {mv.version} as @champion")

# COMMAND ----------

# DBTITLE 1,Step 5 - Score everyone in the hospital now
risk_model = mlflow.pyfunc.load_model(f"models:/{CLF_NAME}@champion")
los_model = mlflow.pyfunc.load_model(f"models:/{REG_NAME}@champion")
clf_v = client.get_model_version_by_alias(CLF_NAME, "champion").version
reg_v = client.get_model_version_by_alias(REG_NAME, "champion").version

cur = spark.sql(features_sql("e.is_in_house")).toPandas()
X_cur = cur[FEATURES]
cur["long_stay_risk"] = np.asarray(risk_model.predict(X_cur))[:, 1]
cur["predicted_los_days"] = np.asarray(los_model.predict(X_cur)).astype(float)
# Models are trained on inpatients only. Observation patients get no risk score, and their LOS is the recent
# median observation stay instead of the inpatient regressor.
is_obs = cur["patient_class"] != "Inpatient"
OBS_MEDIAN = spark.sql(f"""SELECT percentile(los_days, 0.5) AS m FROM {GOLD}.fact_encounter
                           WHERE patient_class = 'Observation' AND discharge_ts IS NOT NULL AND admit_ts >= TIMESTAMP'{SNAP}' - INTERVAL 365 DAYS""").first().m
cur.loc[is_obs, "long_stay_risk"] = np.nan
cur.loc[is_obs, "predicted_los_days"] = OBS_MEDIAN

# The regressor predicts TOTAL LOS at admission. To get REMAINING LOS for someone already t days in, treat LOS as
# log-normal around the prediction (spread = holdout residual sigma) and condition on LOS > t:
#   predicted discharge = conditional median,  P(discharge today) = P(t < LOS <= t + 1 day | LOS > t)
# The window is the 24 hours to 7 AM tomorrow: almost no one is discharged overnight, so this matches "today".
from scipy.stats import norm
SIGMA = REG_METRICS["log_residual_sigma"]
snap_ts = pd.Timestamp(SNAP)
t = ((snap_ts - cur["admit_ts"]).dt.total_seconds() / 86400).clip(lower=0.01)
mu = np.log(cur["predicted_los_days"])
F = lambda x: norm.cdf((np.log(x) - mu) / SIGMA)
Ft = F(t)
cur["predicted_remaining_days"] = np.exp(mu + SIGMA * norm.ppf(Ft + 0.5 * (1 - Ft))) - t
cur["p_discharge_today"] = np.clip((F(t + 1.0) - Ft) / np.maximum(1 - Ft, 1e-9), 0, 1)
cur["past_predicted_los"] = t > cur["predicted_los_days"]
pred = spark.createDataFrame(cur[["encounter_id", "long_stay_risk", "predicted_los_days", "predicted_remaining_days",
                                  "p_discharge_today", "past_predicted_los"]])
pred.createOrReplaceTempView("pred")
spark.sql(f"""
  CREATE OR REPLACE TABLE {GOLD}.inpatient_predictions
  COMMENT "Model scores for current inpatients at the snapshot: long-stay risk and predicted discharge" AS
  SELECT c.encounter_id, round(p.long_stay_risk, 4) AS long_stay_risk,
         CASE WHEN c.patient_class <> 'Inpatient' THEN 'Observation' WHEN p.long_stay_risk >= 0.5 THEN 'High'
              WHEN p.long_stay_risk >= 0.3 THEN 'Medium' ELSE 'Low' END AS risk_tier,
         round(p.predicted_los_days, 2) AS predicted_los_days,
         round(p.predicted_remaining_days, 2) AS predicted_remaining_days, p.past_predicted_los,
         c.snapshot_ts + make_interval(0, 0, 0, 0, 0, 0, CAST(p.predicted_remaining_days * 86400 AS DECIMAL(18, 6))) AS predicted_discharge_ts,
         round(p.p_discharge_today, 4) AS p_discharge_today,
         '{CLF_NAME}' AS risk_model, '{clf_v}' AS risk_model_version, '{REG_NAME}' AS los_model, '{reg_v}' AS los_model_version,
         c.snapshot_ts AS scored_for_snapshot, current_timestamp() AS scored_at
  FROM {GOLD}.current_inpatients c JOIN pred p USING (encounter_id)""")
display(spark.sql(f"""SELECT risk_tier, count(*) AS patients, round(avg(long_stay_risk), 3) AS avg_risk,
                        round(sum(p_discharge_today), 1) AS expected_discharges_today
                      FROM {GOLD}.inpatient_predictions GROUP BY ALL ORDER BY avg_risk DESC NULLS LAST"""))

# COMMAND ----------

# DBTITLE 1,Step 6 - GenAI huddle notes for high-risk patients (de-identified facts only)
# The bed board before notes: current inpatients + scores + plain-language risk drivers
spark.sql(f"""
  CREATE OR REPLACE TEMP VIEW scored AS
  SELECT c.*, p.long_stay_risk, p.risk_tier, p.predicted_los_days, p.predicted_remaining_days, p.past_predicted_los,
         p.predicted_discharge_ts, p.p_discharge_today, p.p_discharge_today >= 0.5 AS likely_discharge_today,
         concat_ws('; ',
           CASE WHEN c.complex_discharge = 1 THEN 'complex discharge screen' END,
           CASE WHEN c.age >= 75 THEN 'age 75+' END,
           CASE WHEN c.lives_alone = 1 THEN 'lives alone' END,
           CASE WHEN c.admitted_from <> 'Home' THEN concat('admitted from ', lower(c.admitted_from)) END,
           CASE WHEN c.fin_class IN ('Medicaid', 'Medicare Advantage') THEN concat(c.fin_class, ' payer') END,
           CASE WHEN c.admit_source = 'Transfer from Hospital' THEN 'transfer admission' END,
           CASE WHEN c.days_past_expected > 0 THEN concat(round(c.days_past_expected, 1), ' days past expected discharge') END,
           CASE WHEN c.open_barrier_category IS NOT NULL THEN concat('open barrier: ', c.open_barrier_category) END) AS risk_drivers,
         p.risk_model_version, p.los_model_version
  FROM {GOLD}.current_inpatients c JOIN {GOLD}.inpatient_predictions p USING (encounter_id)""")

# No names, MRNs or dates of birth go to the model
spark.sql(f"""
  CREATE OR REPLACE TABLE {GOLD}.huddle_notes COMMENT "AI-written 2-sentence huddle notes for the highest-risk inpatients ({LLM}, de-identified input)" AS
  WITH top AS (
    SELECT encounter_id, concat(
      'You are a hospital case management assistant preparing the 7 AM capacity huddle. ',
      'Write exactly two short sentences: first, why this patient is at risk of staying longer than expected; ',
      'second, one specific next step for the case manager today. Use only the facts given and do not invent clinical details. ',
      'Return only the two sentences: no heading, no bullet points, no markdown. ',
      'Facts: age ', age, '; ', unit_name, ' at ', facility_name, '; DRG: ', drg_description,
      '; day ', round(los_so_far_days, 1), ' of an expected ', expected_los, ' days',
      '; long-stay risk ', CAST(round(long_stay_risk * 100) AS INT), '%',
      '; predicted discharge ', date_format(predicted_discharge_ts, 'MMM d'),
      '; risk factors: ', coalesce(nullif(risk_drivers, ''), 'none recorded'), '.') AS prompt
    FROM scored WHERE risk_tier = 'High' ORDER BY long_stay_risk DESC LIMIT 150)
  SELECT encounter_id,
         -- some models still add a bold heading: drop it and any leftover markdown
         regexp_replace(regexp_replace(regexp_replace(r.result, '^[*]{{2}}[^*]+[*]{{2}}', ''), '[*]{{2}}|#', ''), '^[\n\r ]+', '') AS huddle_note,
         r.errorMessage AS note_error, '{LLM}' AS note_model, current_timestamp() AS generated_at
  FROM (SELECT encounter_id, ai_query('{LLM}', prompt, failOnError => false) AS r FROM top)""")
notes = spark.table(f"{GOLD}.huddle_notes")
print(f"Notes: {notes.filter('huddle_note IS NOT NULL').count()} written, {notes.filter('note_error IS NOT NULL').count()} errors")
display(notes.select("encounter_id", "huddle_note").limit(5))

# COMMAND ----------

# DBTITLE 1,Step 7 - Publish bed_board (for Lakebase) and bed_board_deid (for Genie)
spark.sql(f"""
  CREATE OR REPLACE TABLE {GOLD}.bed_board
  COMMENT "Scored bed board: every current inpatient with risk, predicted discharge, barriers and huddle note. Synced to Lakebase for the app. Contains synthetic PHI; no group grants."
  TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
  SELECT s.*, n.huddle_note FROM scored s LEFT JOIN {GOLD}.huddle_notes n USING (encounter_id)""")
spark.sql(f"ALTER TABLE {GOLD}.bed_board ALTER COLUMN encounter_id SET NOT NULL")
spark.sql(f"ALTER TABLE {GOLD}.bed_board ADD CONSTRAINT bed_board_pk PRIMARY KEY (encounter_id)")
spark.sql(f"SET TAG ON TABLE {GOLD}.bed_board lsh_contains_phi = true")
for col, kind in [("mrn", "identifier"), ("patient_name", "name")]:
    spark.sql(f"SET TAG ON COLUMN {GOLD}.bed_board.{col} lsh_phi = {kind}")

spark.sql(f"""
  CREATE OR REPLACE VIEW {GOLD}.bed_board_deid
  COMMENT "De-identified bed board for Genie and analysts: no names, MRNs or encounter IDs" AS
  SELECT facility_code, facility_name, unit_code, unit_name, unit_type, bed, age, sex, patient_class, admit_source, fin_class,
         admitted_from, drg, drg_description, service_line, admit_ts, los_so_far_days, expected_los, expected_discharge_ts,
         days_past_expected, open_barrier_category, open_barrier_days, long_stay_risk, risk_tier, predicted_los_days,
         predicted_remaining_days, predicted_discharge_ts, p_discharge_today, likely_discharge_today, risk_drivers, attending, snapshot_ts
  FROM {GOLD}.bed_board""")
bb = spark.table(f"{GOLD}.bed_board")
assert bb.count() == spark.table(f"{GOLD}.current_inpatients").count() == bb.select("encounter_id").distinct().count()
print(f"bed_board: {bb.count():,} patients (one row each); bed_board_deid created")
display(spark.sql(f"""SELECT facility_code, unit_code, bed, age, drg_description, round(los_so_far_days, 1) AS day_of_stay, expected_los,
                        round(long_stay_risk, 2) AS risk, risk_tier, date_format(predicted_discharge_ts, 'MMM d HH:mm') AS predicted_discharge,
                        open_barrier_category, risk_drivers, huddle_note
                      FROM {GOLD}.bed_board
                      WHERE risk_tier = 'High' ORDER BY long_stay_risk DESC LIMIT 10"""))

# COMMAND ----------

# DBTITLE 1,Step 8 - Prospective check against the simulation answer key
# The answer key from notebook 01 knows when today's inpatients actually leave. The pipeline never used it.
truth = spark.sql(f"""
  SELECT b.facility_code, b.patient_class, b.long_stay_risk, b.predicted_discharge_ts, b.p_discharge_today,
         t.discharge_ts AS actual_discharge_ts, CAST(t.los_days - t.expected_los >= 2 AS INT) AS actual_long_stay,
         to_date(t.discharge_ts) = to_date(b.snapshot_ts) AS actual_discharge_today
  FROM {GOLD}.bed_board b JOIN {RAW}._sim_truth_encounters t ON t.csn = b.encounter_id""").toPandas()
ip = truth[truth["patient_class"] == "Inpatient"]
err = (truth["predicted_discharge_ts"] - truth["actual_discharge_ts"]).abs().dt.total_seconds() / 86400
print(f"Current inpatients: long-stay risk AUC on what actually happened = {roc_auc_score(ip['actual_long_stay'], ip['long_stay_risk']):.3f}")
print(f"Predicted discharge time: median error {err.median():.2f} days, {np.mean(err <= 1):.0%} within 1 day")
print(f"Discharges today among patients in house at 7 AM: expected {truth['p_discharge_today'].sum():.0f}, actual {truth['actual_discharge_today'].sum()}")
display(truth.groupby("facility_code").agg(expected_discharges_today=("p_discharge_today", "sum"),
                                           actual_discharges_today=("actual_discharge_today", "sum")).round(1).reset_index())