# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 08 - Databricks App Build Guide: LSH Capacity Command Center
# MAGIC
# MAGIC The last stage of the journey: one **Databricks App** that puts everything from notebooks 01-07 in front of the people at the 7:00 AM capacity huddle.
# MAGIC
# MAGIC **Deployed app:** `lsh-capacity-command-center` (Python / Streamlit, LSH theme). Source: `/Workspace/Users/sugandha.chauhan@databricks.com/lsh-apps/lsh-capacity-command-center` (`app.py`, `app.yaml`, `requirements.txt`). Step 3 below explains how it is wired. Step 1 checks every resource the app needs. Step 2 prints an optional prompt to rebuild the same app with AppKit (React) in Genie Code or Genie App Builder, where `npm` is available.
# MAGIC
# MAGIC **Who uses it and what they decide**
# MAGIC
# MAGIC | Persona | Page | Decision | Data (from) |
# MAGIC | --- | --- | --- | --- |
# MAGIC | System capacity leader | **Command Center** | Where to send today's admissions and transfers; which units need help | `unit_status` (Lakebase) |
# MAGIC | Unit case manager | **Discharge Worklist** | Which patients to work first and what to do for each | `bed_board` (Lakebase), writes `capacity_ops.discharge_plan` and `discharge_actions` |
# MAGIC | CMO / quality | **LOS Performance** | Where the LOS O/E and excess days are, and which barriers cause them | `los_metrics`, `capacity_metrics`, `barrier_metrics` (SQL warehouse) |
# MAGIC | Anyone | **Ask** | Ad hoc questions in plain English | Genie space from notebook 07 |
# MAGIC
# MAGIC **Why this architecture:** Lakebase gives the worklist millisecond reads and transactional write-back; metric views keep every KPI on one governed definition; Genie runs as the user, so Unity Catalog permissions follow the user into the app.
# MAGIC
# MAGIC **How to run (under a minute):** run 04-07 first, then **Run all** to confirm every resource is ready. Notebook 06 Step 6 has already been run for the app's service principal. A rebuilt app gets a new service principal, so run Step 6 again for it.

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "schauhan_workspace_catalog", "1. Catalog")
dbutils.widgets.text("prefix", "lsh_capacity", "2. Schema prefix")
dbutils.widgets.text("lakebase_project", "lsh-capacity", "3. Lakebase project ID")
dbutils.widgets.text("app_name", "lsh-capacity-command-center", "4. App name")

# COMMAND ----------

# DBTITLE 1,Step 1 - Check every resource the app needs
import re
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from mlflow import MlflowClient

CATALOG, PREFIX = dbutils.widgets.get("catalog").strip(), dbutils.widgets.get("prefix").strip()
PROJECT_ID, APP_NAME = dbutils.widgets.get("lakebase_project").strip(), dbutils.widgets.get("app_name").strip()
for label, val in [("catalog", CATALOG), ("prefix", PREFIX)]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", val or ""):
        raise ValueError(f"Enter a {label} with only letters, numbers and underscores.")
G, SERVING = f"{CATALOG}.{PREFIX}_gold", f"{PREFIX}_serving"
w = WorkspaceClient()
ME = w.current_user.me().user_name
checks = []

# Unity Catalog objects
for obj in ["los_metrics", "capacity_metrics", "barrier_metrics", "bed_board", "bed_board_deid", "unit_status"]:
    checks.append(("Unity Catalog", f"{G}.{obj}", spark.catalog.tableExists(f"{G}.{obj}")))
# Models
mc = MlflowClient(registry_uri="databricks-uc")
for m in ["long_stay_risk_model", "los_days_model"]:
    try:
        v = mc.get_model_version_by_alias(f"{G}.{m}", "champion").version
        checks.append(("Model", f"{G}.{m}@champion (v{v})", True))
    except Exception:
        checks.append(("Model", f"{G}.{m}@champion", False))
# Lakebase
try:
    project = w.postgres.get_project(name=f"projects/{PROJECT_ID}")
    BRANCH = next(b.name for b in w.postgres.list_branches(parent=project.name) if b.name.endswith("/production"))
    EP = next(iter(w.postgres.list_endpoints(parent=BRANCH)))
    HOST = EP.status.hosts.host
    checks.append(("Lakebase", f"{BRANCH} ({HOST})", True))
    for t in ["bed_board", "unit_status"]:
        st = w.postgres.get_synced_table(name=f"synced_tables/{CATALOG}.{SERVING}.{t}").status.detailed_state.value
        checks.append(("Lakebase synced table", f"{SERVING}.{t} ({st})", st.startswith("SYNCED_TABLE_ONLINE")))
except NotFound:
    BRANCH = HOST = None
    checks.append(("Lakebase", f"projects/{PROJECT_ID}", False))
# Genie space and warehouse
TITLE = f"LSH Capacity and LOS Analyst ({PREFIX})"
spaces = w.api_client.do("GET", "/api/2.0/genie/spaces", query={"page_size": 100}).get("spaces", [])
space = next((s for s in spaces if s.get("title") == TITLE), None)
SPACE_ID = space["space_id"] if space else None
WAREHOUSE_ID = w.api_client.do("GET", f"/api/2.0/genie/spaces/{SPACE_ID}").get("warehouse_id") if SPACE_ID else None
checks.append(("Genie space", f"{TITLE} ({SPACE_ID})", bool(SPACE_ID)))
checks.append(("SQL warehouse", str(WAREHOUSE_ID), bool(WAREHOUSE_ID)))

display(spark.createDataFrame(checks, "kind string, resource string, ready boolean"))
missing = [c[1] for c in checks if not c[2]]
assert not missing, f"Not ready: {missing}. Run the earlier notebooks first."
print(f"All {len(checks)} app resources are ready.")

# COMMAND ----------

# DBTITLE 1,Step 2 - (Optional) AppKit rebuild prompt
# The deployed app is Python / Streamlit (Step 3). Use this prompt only to rebuild it as an AppKit (React) app
# where npm is available, e.g. Genie App Builder.
PROMPT = f"""Build a Databricks App called {APP_NAME}: the LSH Capacity Command Center for Lakeshore Health's 7:00 AM capacity huddle. All data is synthetic.

Scaffold with AppKit, non-interactively:
  apps init --name {APP_NAME} --features analytics,genie,lakebase --set analytics.sql-warehouse.id={WAREHOUSE_ID} --set genie.genie-space.id={SPACE_ID} --set "genie.genie-space.name={TITLE}" --set lakebase.postgres.branch={BRANCH} --set lakebase.postgres.database=databricks_postgres --skip-install --auto-approve --output-dir /Workspace/Users/{ME}/lsh-apps

Theme: Lakeshore Health executive command-center template (lsh-app-theme skill). Left navigation, LSH indigo palette (primary #4a4b96, text #241852, background #f5f5fa), DM Sans headings, Inter body, rounded cards with soft shadows, light mode only, minimal clicks. Set the AppKit :root tokens in client/src/index.css and use semantic tokens in components.

Pages (KPI cards on top, one primary visual, one exceptions section, one drill-down, one AI insight):

1. Command Center (capacity leader). Lakebase table {SERVING}.unit_status (one row per unit).
   KPI cards: occupancy now = sum(patients_now)/sum(staffed_beds); open beds now; forecast open beds tonight; expected discharges today; high-risk patients; patients past expected discharge.
   Primary visual: one tile per hospital (facility_code, facility_name) with occupancy and forecast open beds tonight; red when occupancy > 95% or forecast open beds tonight < 5% of beds.
   Exceptions: units with forecast_open_beds_tonight <= 2, sorted ascending.
   Drill-down: unit table (unit_name, unit_type, staffed_beds, patients_now, open_beds_now, high_risk_patients, expected_discharges_today, expected_admissions_today, forecast_open_beds_tonight).

2. Discharge Worklist (case manager). Lakebase table {SERVING}.bed_board, filtered by hospital and unit.
   Rank by long_stay_risk descending. Columns: bed, patient_name, age, drg_description, day of stay (los_so_far_days) vs expected_los, risk_tier badge (High red, Medium amber, Low green, Observation grey), predicted_discharge_ts, open_barrier_category, risk_drivers.
   Row detail: huddle_note (AI-written), risk drivers, barrier, and two actions that write to Lakebase:
     - Confirm discharge plan: upsert capacity_ops.discharge_plan (encounter_id, target_discharge_ts, plan_status in Planned/Confirmed/Delayed/Discharged, delay_reason, updated_by = signed-in user, updated_at = now()).
     - Log action: insert capacity_ops.discharge_actions (encounter_id, action_type in Escalate barrier/SNF referral sent/Family meeting/Transport booked/Pharmacy meds-to-beds, note, owner_role, created_by = signed-in user).
   Show each patient's latest plan status and action count by left-joining capacity_ops tables on encounter_id.

3. LOS Performance (CMO / quality). SQL warehouse queries on the metric views (MEASURE() syntax):
   {G}.los_metrics, {G}.capacity_metrics, {G}.barrier_metrics. Default period: Fiscal Year = 'FY2026' (LSH fiscal year starts Sep 1).
   KPI cards: LOS O/E Index, Excess Days, Avoidable Days, Discharges Before Noon %, Long Stay Rate, Occupancy %.
   Primary visual: LOS O/E Index by Facility (bar, reference line at 1.0). Then monthly Excess Days trend, and Avoidable Days by Barrier Category (pareto).
   Use *.obo.sql queries so the viewer's own Unity Catalog permissions apply; add sql to user_api_scopes.

4. Ask. Embed the Genie space {SPACE_ID} with the sample questions as starter chips.

AI insight card on pages 1-3: a short summary of the page's numbers (Genie or the data itself; never invent numbers).

Rules: never grant permissions yourself; list the grants I need. Validate with apps deploy --skip-validation --auto-approve and fix errors from apps logs {APP_NAME}."""
print(PROMPT)

# COMMAND ----------

# DBTITLE 1,Step 3 - The deployed app: wiring, access and operations
# MAGIC %md
# MAGIC ## How the deployed app is wired
# MAGIC
# MAGIC | Page | Data | Identity used |
# MAGIC | --- | --- | --- |
# MAGIC | Command Center | Lakebase `lsh_capacity_serving.unit_status` | App service principal |
# MAGIC | Discharge Worklist | Lakebase `lsh_capacity_serving.bed_board`; writes `capacity_ops.discharge_plan` and `capacity_ops.discharge_actions` | App service principal; `updated_by` / `created_by` = signed-in user's email |
# MAGIC | LOS Performance | Metric views `los_metrics`, `capacity_metrics`, `barrier_metrics` via SQL warehouse | **Signed-in user** (user authorization scope `sql`) |
# MAGIC | Ask | Genie space from notebook 07 | **Signed-in user** (scope `dashboards.genie`) |
# MAGIC
# MAGIC * **Configuration** lives in `app.yaml` env vars: warehouse ID, Genie space ID, gold schema, Lakebase endpoint, host and database, plus the LSH light theme.
# MAGIC * **Lakebase access:** the app's service principal (`bf14a74d-7000-4d1b-a198-a414f7903a96`) has a Postgres role with SELECT on the synced tables and SELECT / INSERT / UPDATE on `capacity_ops` (notebook 06, Step 6).
# MAGIC * **Fallback:** if Lakebase is unreachable, the app reads the same tables from gold through the warehouse, shows a banner and turns off saving.
# MAGIC
# MAGIC ## Access to set up (owner or admin)
# MAGIC
# MAGIC 1. **App users:** give **CAN USE** on the app only to the care team. The worklist shows (synthetic) patient names.
# MAGIC 2. **For each app user:** `CAN USE` on the SQL warehouse, `CAN RUN` on the Genie space, and `USE CATALOG` / `USE SCHEMA` / `SELECT` on the gold schema. Notebook 04 Step 6 grants the gold schema to the `lsh_capacity_analysts` and `lsh_capacity_care_team` groups once they exist.
# MAGIC 3. **First visit:** each user is asked to consent to the app using their identity for SQL and Genie.
# MAGIC
# MAGIC ## Operations
# MAGIC
# MAGIC * **Daily refresh:** the job **LSH Capacity Command Center - Daily Refresh** runs 02 → 03 → 04 → 05 → 06 at 6:30 AM Central. It was created **paused**: enable the schedule on the job page. Notebook 05 registers a new model version on every run, so for a daily schedule consider retraining weekly instead.
# MAGIC * **Code changes:** edit the files in the source folder, then redeploy from the app page (or `databricks apps deploy lsh-capacity-command-center --source-code-path <source folder>`). In the Streamlit `st.markdown` style block, keep CSS free of blank lines, or Markdown shows it as text.
# MAGIC
# MAGIC ## 10-minute demo path
# MAGIC
# MAGIC 1. **Command Center:** LMH is ~94% full, a few units have 2 or fewer beds forecast tonight, and ~262 discharges are expected today system-wide.
# MAGIC 2. **Discharge Worklist:** LMH 14 West Oncology. The top patient has a 75% long-stay risk (complex discharge, 75+, lives alone). Read the AI huddle note, then confirm a 2 PM discharge and log a family meeting. The write lands in Lakebase in milliseconds.
# MAGIC 3. **LOS Performance:** Willow Creek and LMH have the highest LOS O/E. SNF placement is the top barrier. Prairie Central's before-noon discharges jumped after its March 2026 initiative.
# MAGIC 4. **Ask:** "How many high-risk patients are past their expected discharge at LMH right now?" Genie answers from the de-identified bed board.
# MAGIC 5. **Governance:** in Catalog Explorer, show the PHI tags, the MRN mask, the facility row filter and the lineage from the landing files to `bed_board`.