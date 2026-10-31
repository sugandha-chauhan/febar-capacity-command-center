# 05 - ML Long Stay Risk and Discharge Prediction

Exported from the Databricks workspace on 2026-10-31 06:17 UTC with each cell's **saved output** from its last run. Synthetic data; organization and hospital names anonymized.


---
### Cell 1: Overview (markdown)

# 05 - ML Long Stay Risk and Discharge Prediction

Makes the bed board **intelligent**. Two models learn from 2 years of discharged inpatient stays, and a GenAI step writes huddle notes:

| Model | Predicts | Used for |
| --- | --- | --- |
| `long_stay_risk_model` (classifier) | Probability that a stay runs **2+ days past its expected LOS** | Ranking who case managers work first |
| `los_days_model` (regressor, log-LOS) | Total length of stay in days | Predicted discharge date, and how many beds open today |
| AI huddle note (`ai_query`, Foundation Model API) | A 2-sentence "why at risk / next step" note for high-risk patients, from de-identified facts only | The app's huddle view |

**Prediction time:** the models use only what is known **at admission** (age, DRG and expected LOS, unit, admit source, payer, complex-discharge screen, living situation, prior admissions). CarePort barriers are **not** model inputs: they are usually raised near the end of the stay, so using them would leak the outcome. The app shows open barriers next to the score instead.

**Steps:** build features from gold, quick EDA, time-based split (train on admissions before May 2026, test on May-Aug 2026), train and evaluate with MLflow tracking, refit on all data, validate the artifacts in an isolated environment, register both models in Unity Catalog with the `@champion` alias, then score everyone in the hospital now.

**Outputs in** `<prefix>_gold`: `inpatient_predictions`, `huddle_notes`, `bed_board` (the scored, patient-level operational table synced to Lakebase in 06) and `bed_board_deid` (no names or MRNs, for Genie).

**Prospective check:** notebook 01 saved the simulation's answer key, so the last step compares the predictions for patients still in the hospital with what actually happens to them after 7:00 AM.

**How to run (about 8 minutes, CPU serverless is fine):** run notebooks 03 and 04 first, then **Run all**.

---
### Cell 2: Parameters

```python
dbutils.widgets.text("catalog", "schauhan_workspace_catalog", "1. Catalog")
dbutils.widgets.text("prefix", "lsh_capacity", "2. Schema prefix")
dbutils.widgets.text("llm_endpoint", "databricks-claude-haiku-4-5", "3. LLM endpoint for huddle notes")
```

_No saved output._


---
### Cell 3: Step 1 - Setup and features (same feature query for training and scoring)

```python
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
```

**Output:**

```text
{"ts": "2026-10-30 18:23:20.181", "level": "WARNING", "logger": "pyspark.sql.connect.logging", "msg": "Effective usage policy for this session is cct.Chpub3RlYm9va3MvMjc5MjM1MTMzMzEzMjAzMRABIAEyJDAxYTE4ZTBkLTFkN2UtN2FmMi1hNjgxLWRiZjRiNDFkYjJjYjokMDkzZDc5MzItYjliYS0zY2FmLWEzMDYtNTIwMmY3YmVkZTRiQiQxYWU0YjE0ZS1kYTNmLTRiMDEtYmQxNy05N2U2Yjg2NzhhYjhKDAiKwZPXBhCA78uoAlABWAFgAWiN8aSA5MSjDYgBAA==.", "context": {}}
{"ts": "2026-10-30 18:23:20.181", "level": "WARNING", "logger": "pyspark.sql.connect.logging", "msg": "Effective usage policy for this session is cct.Chpub3RlYm9va3MvMjc5MjM1MTMzMzEzMjAzMRABIAEyJDAxYTE4ZTBkLTFkN2UtN2FmMi1hNjgxLWRiZjRiNDFkYjJjYjokMDkzZDc5MzItYjliYS0zY2FmLWEzMDYtNTIwMmY3YmVkZTRiQiQxYWU0YjE0ZS1kYTNmLTRiMDEtYmQxNy05N2U2Yjg2NzhhYjhKDAiKwZPXBhCA78uoAlABWAFgAWiN8aSA5MSjDYgBAA==.", "context": {}}
{"ts": "2026-10-30 18:23:20.181", "level": "WARNING", "logger": "pyspark.sql.connect.logging", "msg": "Effective usage policy for this session is cct.Chpub3RlYm9va3MvMjc5MjM1MTMzMzEzMjAzMRABIAEyJDAxYTE4ZTBkLTFkN2UtN2FmMi1hNjgxLWRiZjRiNDFkYjJjYjokMDkzZDc5MzItYjliYS0zY2FmLWEzMDYtNTIwMmY3YmVkZTRiQiQxYWU0YjE0ZS1kYTNmLTRiMDEtYmQxNy05N2U2Yjg2NzhhYjhKDAiKwZPXBhCA78uoAlABWAFgAWiN8aSA5MSjDYgBAA==.", "context": {}}
2026/10/30 18:23:22 INFO mlflow.tracking.fluent: Experiment with name '/Users/sugandha.chauhan@databricks.com/LSH Capacity and LOS Prototype/lsh_capacity_los_models' does not exist. Creating a new experiment.
```

**Output:**

```text
Training window: 155,257 discharged inpatient stays; snapshot 2026-10-30 07:00:00
```


---
### Cell 4: Step 2 - Exploratory data analysis

```python
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
```

**Output (table):**

| feature | type |
| --- | --- |
| facility_code | categorical / nominal |
| unit_type | categorical / nominal |
| drg | categorical / nominal |
| service_line | categorical / nominal |
| sex | categorical / nominal |
| admit_source | categorical / nominal |
| fin_class | categorical / nominal |
| admitted_from | categorical / nominal |
| age_at_admit | numeric float |
| expected_los | numeric float |
| complex_discharge | binary flag |
| lives_alone | binary flag |
| is_surgical | binary flag |
| admit_dow | categorical / ordinal (day of week) |
| admit_hour | numeric integer (hour) |
| prior_admits_12m | numeric integer (count) |
| prior_long_stay | binary flag |

**Output (table):**

| index | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age_at_admit | 155257.0 | 56.37303309995684 | 23.359051050501787 | 15.0 | 35.0 | 56.0 | 77.0 | 97.0 |
| expected_los | 155257.0 | 3.692096652646902 | 2.4403210684672882 | 1.18 | 2.01 | 3.17 | 4.39 | 16.83 |
| complex_discharge | 155257.0 | 0.10632048796511591 | 0.30824836382776283 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| lives_alone | 155257.0 | 0.14961644241483477 | 0.35669620416816433 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| is_surgical | 155257.0 | 0.27145957992232234 | 0.4447140094686304 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| admit_dow | 155257.0 | 3.932486135890813 | 1.9437262738533683 | 1.0 | 2.0 | 4.0 | 6.0 | 7.0 |
| admit_hour | 155257.0 | 14.111846808839537 | 5.635344087042631 | 0.0 | 8.0 | 15.0 | 19.0 | 23.0 |
| prior_admits_12m | 155257.0 | 0.3731941232923475 | 0.6361383662883703 | 0.0 | 0.0 | 0.0 | 1.0 | 6.0 |
| prior_long_stay | 155257.0 | 0.05457402886826359 | 0.22714760964620734 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| los_days | 155257.0 | 4.529719155980085 | 3.7492607005703036 | 0.5 | 2.133 | 3.453 | 5.696 | 97.946 |
| is_long_stay | 155257.0 | 0.20226463219049703 | 0.40168979328600757 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |

**Output:**

```text
Long stay (2+ days past expected): 20.2% of stays; not long: 79.8%
```

**Output:**

```text
<Figure size 1700x400 with 3 Axes>
```

**Output (table):**

| feature | is_long_stay | los_days |
| --- | --- | --- |
| age_at_admit | 0.18 | 0.304 |
| expected_los | 0.241 | 0.771 |
| complex_discharge | 0.185 | 0.189 |
| lives_alone | 0.07 | 0.08 |
| is_surgical | -0.016 | 0.007 |
| admit_dow | 0.023 | 0.015 |
| admit_hour | 0.014 | 0.014 |
| prior_admits_12m | 0.013 | 0.022 |
| prior_long_stay | 0.023 | 0.033 |


---
### Cell 5: Step 3 - Time-based split, train, evaluate (tracked in MLflow)

```python
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
```

**Output:**

```text
Temporal split: train = admissions before 2026-05-01 (129,840), test = May-Aug 2026 (25,417)
```

**Output:**

```text
<Figure size 1100x400 with 2 Axes>
```

**Output:**

```text
Confusion matrix at 0.5 (rows = actual, cols = predicted):
 [[19595   391]
 [ 4884   547]]
```

**Output (table):**

| test_roc_auc | test_pr_auc | test_log_loss | test_brier | test_f1 | test_accuracy | baseline_auc_complex_flag_only |
| --- | --- | --- | --- | --- | --- | --- |
| 0.722 | 0.41 | 0.465 | 0.15 | 0.172 | 0.792 | 0.564 |

**Output (table):**

| test_mae_days | test_median_ae_days | test_within_1_day | baseline_mae_expected_los | log_residual_sigma |
| --- | --- | --- | --- | --- |
| 1.53 | 0.976 | 0.51 | 1.628 | 0.446 |


---
### Cell 6: Step 4 - Refit on all data, log, validate in isolation, register in UC as @champion

```python
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
```

**Output:**

```text
🔗 View Logged Model at: https://fevm-schauhan-workspace.cloud.databricks.com/ml/experiments/2792351333132032/models/m-fa23f2a0f33b4eb688d6e7c8105b95a7?o=7474644328724621
🔗 View Logged Model at: https://fevm-schauhan-workspace.cloud.databricks.com/ml/experiments/2792351333132032/models/m-35d74789ba114c97a8784871e0716c7d?o=7474644328724621
```

**Output:**

```text
Downloading artifacts:   0%|          | 0/11 [00:00<?, ?it/s]
```

**Output:**

```text
Downloading artifacts:   0%|          | 0/11 [00:00<?, ?it/s]
```

**Output:**

```text
2026/10/30 18:25:23 INFO mlflow.models.flavor_backend_registry: Selected backend for flavor 'python_function'
```

**Output:**

```text
Downloading artifacts:   0%|          | 0/11 [00:00<?, ?it/s]
```

**Output:**

```text
2026/10/30 18:25:25 INFO mlflow.utils.virtualenv: Creating a new environment in /tmp/virtualenv_envs/mlflow-4bfa1453a32f1d960f57094995d698fada99ed43 with python version 3.12.3 using uv
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtual environment at: /tmp/virtualenv_envs/mlflow-4bfa1453a32f1d960f57094995d698fada99ed43
Activate with: source /tmp/virtualenv_envs/mlflow-4bfa1453a32f1d960f57094995d698fada99ed43/bin/activate
2026/10/30 18:25:25 INFO mlflow.utils.virtualenv: Installing dependencies
Using Python 3.12.3 environment at: /tmp/virtualenv_envs/mlflow-4bfa1453a32f1d960f57094995d698fada99ed43
Resolved 3 packages in 240ms
Downloading setuptools (1.1MiB)
 Downloaded setuptools
Downloading pip (1.7MiB)
 Downloaded pip
Prepared 3 packages in 309ms
Installed 3 packages in 54ms
 + pip==25.3
 + setuptools==80.9.0
 + wheel==0.45.1
Using Python 3.12.3 environment at: /tmp/virtualenv_envs/mlflow-4bfa1453a32f1d960f57094995d698fada99ed43
Resolved 89 packages in 1.44s
Downloading pydantic-core (2.1MiB)
Downloading cryptography (4.3MiB)
Downloading sqlalchemy (4.5MiB)
Downloading aiohttp (1.8MiB)
Downloading scikit-learn (9.0MiB)
Downloading mlflow-tracing (1.5MiB)
Downloading fonttools (5.2MiB)
Downloading kiwisolver (1.4MiB)
Downloading pandas (11.8MiB)
Downloading scipy (33.7MiB)
Downloading databricks-sdk (1.1MiB)
Downloading pillow (6.8MiB)
Downloading mlflow-skinny (3.1MiB)
Downloading mlflow (10.1MiB)
Downloading pyarrow (40.8MiB)
Downloading matplotlib (9.4MiB)
Downloading numpy (16.0MiB)
 Downloaded pydantic-core
 Downloaded aiohttp
 Downloaded mlflow-tracing
 Downloaded databricks-sdk
 Downloaded sqlalchemy
 Downloaded kiwisolver
 Downloaded cryptography
 Downloaded mlflow-skinny
 Downloaded fonttools
 Downloaded pillow
 Downloaded scikit-learn
 Downloaded matplotlib
 Downloaded mlflow
 Downloaded pandas
 Downloaded numpy
 Downloaded scipy
 Downloaded pyarrow
Prepared 89 packages in 2.52s
Installed 89 packages in 2.88s
 + aiohappyeyeballs==2.7.1
 + aiohttp==3.14.5
 + aiosignal==1.4.0
 + alembic==1.20.1
 + annotated-doc==0.0.5
 + annotated-types==0.8.0
 + anyio==4.15.1
 + attrs==26.1.0
 + blinker==1.9.0
 + cachetools==7.2.7
 + certifi==2026.10.23
 + cffi==2.1.1
 + charset-normalizer==3.5.2
 + click==8.5.0
 + cloudpickle==3.1.1
 + contourpy==1.4.0
 + cryptography==46.0.7
 + cycler==0.12.1
 + databricks-sdk==0.158.0
 + docker==7.2.0
 + fastapi==0.162.0
 + flask==3.1.3
 + flask-cors==6.0.5
 + fonttools==4.68.0
 + frozenlist==1.8.0
 + gitdb==4.0.12
 + gitpython==3.2.1
 + google-auth==2.62.0
 + graphene==3.4.3
 + graphql-core==3.2.13
 + graphql-relay==3.2.0
 + gunicorn==25.3.0
 + h11==0.16.0
 + huey==2.6.0
 + idna==3.20
 + importlib-metadata==9.1.0
 + itsdangerous==2.2.0
 + jinja2==3.1.6
 + joblib==1.6.0
 + kiwisolver==1.5.1
 + mako==1.4.3
 + markupsafe==3.0.4
 + matplotlib==3.11.2
 + mlflow==3.12.0
 + mlflow-skinny==3.12.0
 + mlflow-tracing==3.12.0
 + multidict==7.2.1
 + numpy==2.5.4
 + opentelemetry-api==1.45.2
 + opentelemetry-proto==1.45.2
 + opentelemetry-sdk==1.45.2
 + opentelemetry-semantic-conventions==0.66b2
 + packaging==26.3
 + pandas==2.3.3
 + pillow==13.0.0
 + prettytable==3.18.0
 + propcache==0.5.4
 + protobuf==7.36.2
 + psutil==7.0.0
 + pyarrow==21.0.0
 + pyasn1==0.6.5
 + pyasn1-modules==0.4.2
 + pycparser==3.12
 + pydantic==2.14.2
 + pydantic-core==2.50.2
 + pyparsing==3.3.3
 + python-dateutil==2.9.0.post0
 + python-dotenv==1.2.4
 + pytz==2026.6
 + pyyaml==6.0.3
 + requests==2.34.2
 + scikit-learn==1.7.2
 + scipy==1.18.1
 + six==1.17.0
 + skops==0.16.0
 + smmap==5.0.3
 + sqlalchemy==2.1.6
 + sqlparse==0.6.0
 + starlette==0.52.1
 + threadpoolctl==3.7.0
 + typing-extensions==4.16.0
 + typing-inspection==0.4.4
 + tzdata==2026.6
 + urllib3==2.8.0
 + uvicorn==0.54.0
 + wcwidth==0.9.4
 + werkzeug==3.1.9
 + yarl==1.25.1
 + zipp==4.1.2
2026/10/30 18:25:33 INFO mlflow.utils.environment: === Running command '['bash', '-c', 'source /tmp/virtualenv_envs/mlflow-4bfa1453a32f1d960f57094995d698fada99ed43/bin/activate && python -c ""']'
2026/10/30 18:25:33 INFO mlflow.utils.environment: === Running command '['bash', '-c', 'source /tmp/virtualenv_envs/mlflow-4bfa1453a32f1d960f57094995d698fada99ed43/bin/activate && python /databricks/python/lib/python3.12/site-packages/mlflow/pyfunc/_mlflow_pyfunc_backend_predict.py --model-uri file:///local_disk0/user_tmp_data/spark-ace1498f-74f4-478e-9c52-ea/tmph0xg3hsc --content-type json --input-path /local_disk0/user_tmp_data/spark-ace1498f-74f4-478e-9c52-ea/tmpaam416be/input.json']'
```

**Output:**

```text
{"predictions": [[0.7403695799558105, 0.2596304200441894], [0.7649384253117506, 0.23506157468824945], [0.544450876857704, 0.455549123142296]]}
```

**Output:**

```text
Downloading artifacts:   0%|          | 0/1 [00:00<?, ?it/s]
```

**Output:**

```text
Downloading artifacts:   0%|          | 0/11 [00:00<?, ?it/s]
```

**Output:**

```text
Downloading artifacts:   0%|          | 0/11 [00:00<?, ?it/s]
```

**Output:**

```text
2026/10/30 18:25:47 INFO mlflow.models.flavor_backend_registry: Selected backend for flavor 'python_function'
```

**Output:**

```text
Downloading artifacts:   0%|          | 0/11 [00:00<?, ?it/s]
```

**Output:**

```text
2026/10/30 18:25:49 INFO mlflow.utils.virtualenv: Environment /tmp/virtualenv_envs/mlflow-4bfa1453a32f1d960f57094995d698fada99ed43 already exists
2026/10/30 18:25:49 INFO mlflow.utils.environment: === Running command '['bash', '-c', 'source /tmp/virtualenv_envs/mlflow-4bfa1453a32f1d960f57094995d698fada99ed43/bin/activate && python -c ""']'
2026/10/30 18:25:49 INFO mlflow.utils.environment: === Running command '['bash', '-c', 'source /tmp/virtualenv_envs/mlflow-4bfa1453a32f1d960f57094995d698fada99ed43/bin/activate && python /databricks/python/lib/python3.12/site-packages/mlflow/pyfunc/_mlflow_pyfunc_backend_predict.py --model-uri file:///local_disk0/user_tmp_data/spark-ace1498f-74f4-478e-9c52-ea/tmp1wghxv05 --content-type json --input-path /local_disk0/user_tmp_data/spark-ace1498f-74f4-478e-9c52-ea/tmp2d9zlhla/input.json']'
```

**Output:**

```text
{"predictions": [5.134860920227262, 4.720348521790712, 12.335225986698847]}Both artifacts load and predict in an isolated environment
```

**Output:**

```text
Downloading artifacts:   0%|          | 0/1 [00:00<?, ?it/s]
```

**Output:**

```text
Successfully registered model 'schauhan_workspace_catalog.lsh_capacity_gold.long_stay_risk_model'.
```

**Output:**

```text
Downloading artifacts:   0%|          | 0/11 [00:00<?, ?it/s]
```

**Output:**

```text
Uploading artifacts:   0%|          | 0/12 [00:00<?, ?it/s]
```

**Output:**

```text
🔗 Created version '1' of model 'schauhan_workspace_catalog.lsh_capacity_gold.long_stay_risk_model': https://fevm-schauhan-workspace.cloud.databricks.com/explore/data/models/schauhan_workspace_catalog/lsh_capacity_gold/long_stay_risk_model/version/1?o=7474644328724621
```

**Output:**

```text
Registered schauhan_workspace_catalog.lsh_capacity_gold.long_stay_risk_model version 1 as @champion
```

**Output:**

```text
Successfully registered model 'schauhan_workspace_catalog.lsh_capacity_gold.los_days_model'.
```

**Output:**

```text
Downloading artifacts:   0%|          | 0/11 [00:00<?, ?it/s]
```

**Output:**

```text
Uploading artifacts:   0%|          | 0/12 [00:00<?, ?it/s]
```

**Output:**

```text
🔗 Created version '1' of model 'schauhan_workspace_catalog.lsh_capacity_gold.los_days_model': https://fevm-schauhan-workspace.cloud.databricks.com/explore/data/models/schauhan_workspace_catalog/lsh_capacity_gold/los_days_model/version/1?o=7474644328724621
```

**Output:**

```text
Registered schauhan_workspace_catalog.lsh_capacity_gold.los_days_model version 1 as @champion
```


---
### Cell 7: Step 5 - Score everyone in the hospital now

```python
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
```

**Output:**

```text
Downloading artifacts:   0%|          | 0/12 [00:00<?, ?it/s]
```

**Output:**

```text
Downloading artifacts:   0%|          | 0/12 [00:00<?, ?it/s]
```

**Output (table):**

| risk_tier | patients | avg_risk | expected_discharges_today |
| --- | --- | --- | --- |
| High | 95 | 0.594 | 8.8 |
| Medium | 301 | 0.381 | 35.2 |
| Low | 613 | 0.181 | 171.6 |
| Observation | 55 | None | 46.2 |


---
### Cell 8: Step 6 - GenAI huddle notes for high-risk patients (de-identified facts only)

```python
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
```

**Output:**

```text
Notes: 95 written, 0 errors
```

**Output (table):**

| encounter_id | huddle_note |
| --- | --- |
| 9081406001 | This 77-year-old patient with acute leukemia has a 75% long-stay risk due to complex discharge needs and living alone, which may delay placement planning beyond the predicted November 12 discharge dat |
| 9080906003 | This 97-year-old patient with acute leukemia is at risk of exceeding the expected 14.37-day length of stay due to a complex discharge screen and advanced age. The case manager should initiate discharg |
| 9080320003 | This 76-year-old patient is 3.7 days past expected discharge with a 71% long-stay risk driven by complex discharge needs, advanced age, and living alone. The case manager should complete a comprehensi |
| 9081606001 | This 84-year-old patient admitted from assisted living with complex discharge needs and multiple comorbidities has a 71% long-stay risk and is currently tracking below the expected length of stay benc |
| 9080807002 | This patient is at risk of staying longer than expected due to a complex discharge screen requiring additional coordination and a Medicaid payer status that may delay post-acute placement approval. Th |


---
### Cell 9: Step 7 - Publish bed_board (for Lakebase) and bed_board_deid (for Genie)

```python
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
```

**Output:**

```text
bed_board: 1,064 patients (one row each); bed_board_deid created
```

**Output (table):**

| facility_code | unit_code | bed | age | drg_description | day_of_stay | expected_los | risk | risk_tier | predicted_discharge | open_barrier_category | risk_drivers | huddle_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LMH | LMH-14W | 14W-11 | 77 | Acute leukemia w/o major O.R. procedure w MCC | 6.0 | 12.96 | 0.75 | High | Nov 12 11:19 | None | complex discharge screen; age 75+; lives alone | This 77-year-old patient with acute leukemia has a 75% long-stay risk due to complex discharge needs and living alone, which may delay placement planning beyond the predicted November 12 discharge dat |
| LMH | LMH-14W | 14W-05 | 97 | Acute leukemia w/o major O.R. procedure w MCC | 10.6 | 14.37 | 0.71 | High | Nov 8 16:27 | None | complex discharge screen; age 75+ | This 97-year-old patient with acute leukemia is at risk of exceeding the expected 14.37-day length of stay due to a complex discharge screen and advanced age. The case manager should initiate discharg |
| PCH | PCH-5N | 5N-02 | 76 | Acute leukemia w/o major O.R. procedure w MCC | 16.6 | 12.89 | 0.71 | High | Nov 5 05:09 | None | complex discharge screen; age 75+; lives alone; transfer admission; 3.7 days past expected discharge | This 76-year-old patient is 3.7 days past expected discharge with a 71% long-stay risk driven by complex discharge needs, advanced age, and living alone. The case manager should complete a comprehensi |
| LMH | LMH-14W | 14W-19 | 84 | Acute leukemia w/o major O.R. procedure w MCC | 3.8 | 13.45 | 0.71 | High | Nov 13 20:26 | None | complex discharge screen; age 75+; admitted from assisted living | This 84-year-old patient admitted from assisted living with complex discharge needs and multiple comorbidities has a 71% long-stay risk and is currently tracking below the expected length of stay benc |
| LMH | LMH-15W | 15W-05 | 56 | Acute leukemia w/o major O.R. procedure w MCC | 11.4 | 11.48 | 0.69 | High | Nov 6 06:06 | None | complex discharge screen; Medicaid payer | This patient is at risk of staying longer than expected due to a complex discharge screen requiring additional coordination and a Medicaid payer status that may delay post-acute placement approval. Th |
| FVH | FVH-ICU | ICU-01 | 79 | Septicemia or severe sepsis w MV >96 hours | 13.6 | 14.97 | 0.69 | High | Nov 8 19:59 | None | complex discharge screen; age 75+; admitted from skilled nursing facility | This 79-year-old patient is at risk of exceeding the expected length of stay due to a complex discharge screen combined with advanced age and admission from a skilled nursing facility. The case manage |
| LMH | LMH-10W | 10W-01 | 85 | Intracranial hemorrhage or cerebral infarction w MCC | 10.5 | 5.54 | 0.68 | High | Nov 2 14:35 | None | complex discharge screen; age 75+; transfer admission; 5.0 days past expected discharge | This 85-year-old patient is 5 days past the expected length of stay with a 68% long-stay risk due to complex discharge requirements and transfer admission status. The case manager should complete a co |
| LMH | LMH-NSICU | NSICU-06 | 97 | Craniotomy w major device implant or acute complex CNS PDX w MCC | 8.9 | 10.95 | 0.68 | High | Nov 7 09:35 | None | complex discharge screen; age 75+; admitted from skilled nursing facility; Medicaid payer | This 97-year-old patient has a 68% long-stay risk due to complex discharge needs, advanced age, Medicaid coverage, and admission from skilled nursing facility. The case manager should complete the com |
| LMH | LMH-14W | 14W-18 | 86 | Chemotherapy w/o acute leukemia as secondary diagnosis w MCC | 4.2 | 6.09 | 0.68 | High | Nov 5 01:26 | None | complex discharge screen; age 75+; Medicare Advantage payer; transfer admission | This 86-year-old transfer admission with complex discharge needs and Medicare Advantage coverage presents a 68% long-stay risk that may extend beyond the predicted November 5 discharge. The case manag |
| LMH | LMH-10W | 10W-08 | 79 | Intracranial hemorrhage or cerebral infarction w MCC | 5.7 | 5.37 | 0.67 | High | Nov 3 04:17 | None | complex discharge screen; age 75+; admitted from assisted living; Medicaid payer; 0.4 days past expected discharge | This 79-year-old patient with intracranial hemorrhage is at risk of extended length of stay due to complex discharge planning needs, Medicaid payer status, and admission from assisted living. The case |


---
### Cell 10: Step 8 - Prospective check against the simulation answer key

```python
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
```

**Output:**

```text
Current inpatients: long-stay risk AUC on what actually happened = 0.680
Predicted discharge time: median error 1.19 days, 44% within 1 day
Discharges today among patients in house at 7 AM: expected 262, actual 262
```

**Output (table):**

| facility_code | expected_discharges_today | actual_discharges_today |
| --- | --- | --- |
| PCH | 50.8 | 52 |
| RBH | 16.9 | 18 |
| ORH | 15.6 | 13 |
| CFH | 12.2 | 14 |
| BSH | 28.2 | 29 |
| FVH | 16.5 | 18 |
| LMH | 98.2 | 97 |
| WCH | 23.4 | 21 |
