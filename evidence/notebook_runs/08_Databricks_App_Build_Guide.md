# 08 - Databricks App Build Guide

Exported from the Databricks workspace on 2026-10-31 06:17 UTC with each cell's **saved output** from its last run. Synthetic data; organization and hospital names anonymized.


---
### Cell 1: Overview (markdown)

# 08 - Databricks App Build Guide: LSH Capacity Command Center

The last stage of the journey: one **Databricks App** that puts everything from notebooks 01-07 in front of the people at the 7:00 AM capacity huddle.

**Deployed app:** `lsh-capacity-command-center` (Python / Streamlit, LSH theme). Source: `/Workspace/Users/sugandha.chauhan@databricks.com/lsh-apps/lsh-capacity-command-center` (`app.py`, `app.yaml`, `requirements.txt`). Step 3 below explains how it is wired. Step 1 checks every resource the app needs. Step 2 prints an optional prompt to rebuild the same app with AppKit (React) in Genie Code or Genie App Builder, where `npm` is available.

**Who uses it and what they decide**

| Persona | Page | Decision | Data (from) |
| --- | --- | --- | --- |
| System capacity leader | **Command Center** | Where to send today's admissions and transfers; which units need help | `unit_status` (Lakebase) |
| Unit case manager | **Discharge Worklist** | Which patients to work first and what to do for each | `bed_board` (Lakebase), writes `capacity_ops.discharge_plan` and `discharge_actions` |
| CMO / quality | **LOS Performance** | Where the LOS O/E and excess days are, and which barriers cause them | `los_metrics`, `capacity_metrics`, `barrier_metrics` (SQL warehouse) |
| Anyone | **Ask** | Ad hoc questions in plain English | Genie space from notebook 07 |

**Why this architecture:** Lakebase gives the worklist millisecond reads and transactional write-back; metric views keep every KPI on one governed definition; Genie runs as the user, so Unity Catalog permissions follow the user into the app.

**How to run (under a minute):** run 04-07 first, then **Run all** to confirm every resource is ready. Notebook 06 Step 6 has already been run for the app's service principal. A rebuilt app gets a new service principal, so run Step 6 again for it.

---
### Cell 2: Parameters

```python
dbutils.widgets.text("catalog", "schauhan_workspace_catalog", "1. Catalog")
dbutils.widgets.text("prefix", "lsh_capacity", "2. Schema prefix")
dbutils.widgets.text("lakebase_project", "lsh-capacity", "3. Lakebase project ID")
dbutils.widgets.text("app_name", "lsh-capacity-command-center", "4. App name")
```

_No saved output._


---
### Cell 3: Step 1 - Check every resource the app needs

```python
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
```

**Output:**

```text
{"ts": "2026-10-30 19:53:31.796", "level": "WARNING", "logger": "pyspark.sql.connect.logging", "msg": "Effective usage policy for this session is cct.Chpub3RlYm9va3MvMjc5MjM1MTMzMzEzMjAzNxABIAEyJDAxYTE4ZTVmLWE5MWUtNzVkZS1iYzNhLTg4NTkxNTEzNGY2MTokMDkzZDc5MzItYjliYS0zY2FmLWEzMDYtNTIwMmY3YmVkZTRiQiQxYWU0YjE0ZS1kYTNmLTRiMDEtYmQxNy05N2U2Yjg2NzhhYjhKDAis65PXBhCAl9GXAVABWAFgAWiN8aSA5MSjDYgBAA==.", "context": {}}
{"ts": "2026-10-30 19:53:31.796", "level": "WARNING", "logger": "pyspark.sql.connect.logging", "msg": "Effective usage policy for this session is cct.Chpub3RlYm9va3MvMjc5MjM1MTMzMzEzMjAzNxABIAEyJDAxYTE4ZTVmLWE5MWUtNzVkZS1iYzNhLTg4NTkxNTEzNGY2MTokMDkzZDc5MzItYjliYS0zY2FmLWEzMDYtNTIwMmY3YmVkZTRiQiQxYWU0YjE0ZS1kYTNmLTRiMDEtYmQxNy05N2U2Yjg2NzhhYjhKDAis65PXBhCAl9GXAVABWAFgAWiN8aSA5MSjDYgBAA==.", "context": {}}
{"ts": "2026-10-30 19:53:31.796", "level": "WARNING", "logger": "pyspark.sql.connect.logging", "msg": "Effective usage policy for this session is cct.Chpub3RlYm9va3MvMjc5MjM1MTMzMzEzMjAzNxABIAEyJDAxYTE4ZTVmLWE5MWUtNzVkZS1iYzNhLTg4NTkxNTEzNGY2MTokMDkzZDc5MzItYjliYS0zY2FmLWEzMDYtNTIwMmY3YmVkZTRiQiQxYWU0YjE0ZS1kYTNmLTRiMDEtYmQxNy05N2U2Yjg2NzhhYjhKDAis65PXBhCAl9GXAVABWAFgAWiN8aSA5MSjDYgBAA==.", "context": {}}
```

**Output (table):**

| kind | resource | ready |
| --- | --- | --- |
| Unity Catalog | schauhan_workspace_catalog.lsh_capacity_gold.los_metrics | True |
| Unity Catalog | schauhan_workspace_catalog.lsh_capacity_gold.capacity_metrics | True |
| Unity Catalog | schauhan_workspace_catalog.lsh_capacity_gold.barrier_metrics | True |
| Unity Catalog | schauhan_workspace_catalog.lsh_capacity_gold.bed_board | True |
| Unity Catalog | schauhan_workspace_catalog.lsh_capacity_gold.bed_board_deid | True |
| Unity Catalog | schauhan_workspace_catalog.lsh_capacity_gold.unit_status | True |
| Model | schauhan_workspace_catalog.lsh_capacity_gold.long_stay_risk_model@champion (v1) | True |
| Model | schauhan_workspace_catalog.lsh_capacity_gold.los_days_model@champion (v1) | True |
| Lakebase | projects/lsh-capacity/branches/production (ep-round-grass-d1za0378.database.us-west-2.cloud.databricks.com) | True |
| Lakebase synced table | lsh_capacity_serving.bed_board (SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE) | True |
| Lakebase synced table | lsh_capacity_serving.unit_status (SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE) | True |
| Genie space | LSH Capacity and LOS Analyst (lsh_capacity) (01f1d49b09321c7db9f9e412c68bde9e) | True |
| SQL warehouse | a87df88e51fce476 | True |

**Output:**

```text
All 13 app resources are ready.
```


---
### Cell 4: Step 2 - (Optional) AppKit rebuild prompt

```python
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
```

**Output:**

```text
Build a Databricks App called lsh-capacity-command-center: the LSH Capacity Command Center for Lakeshore Health's 7:00 AM capacity huddle. All data is synthetic.

Scaffold with AppKit, non-interactively:
  apps init --name lsh-capacity-command-center --features analytics,genie,lakebase --set analytics.sql-warehouse.id=a87df88e51fce476 --set genie.genie-space.id=01f1d49b09321c7db9f9e412c68bde9e --set "genie.genie-space.name=LSH Capacity and LOS Analyst (lsh_capacity)" --set lakebase.postgres.branch=projects/lsh-capacity/branches/production --set lakebase.postgres.database=databricks_postgres --skip-install --auto-approve --output-dir /Workspace/Users/sugandha.chauhan@databricks.com/lsh-apps

Theme: Lakeshore Health executive command-center template (lsh-app-theme skill). Left navigation, LSH indigo palette (primary #4a4b96, text #241852, background #f5f5fa), DM Sans headings, Inter body, rounded cards with soft shadows, light mode only, minimal clicks. Set the AppKit :root tokens in client/src/index.css and use semantic tokens in components.

Pages (KPI cards on top, one primary visual, one exceptions section, one drill-down, one AI insight):

1. Command Center (capacity leader). Lakebase table lsh_capacity_serving.unit_status (one row per unit).
   KPI cards: occupancy now = sum(patients_now)/sum(staffed_beds); open beds now; forecast open beds tonight; expected discharges today; high-risk patients; patients past expected discharge.
   Primary visual: one tile per hospital (facility_code, facility_name) with occupancy and forecast open beds tonight; red when occupancy > 95% or forecast open beds tonight < 5% of beds.
   Exceptions: units with forecast_open_beds_tonight <= 2, sorted ascending.
   Drill-down: unit table (unit_name, unit_type, staffed_beds, patients_now, open_beds_now, high_risk_patients, expected_discharges_today, expected_admissions_today, forecast_open_beds_tonight).

2. Discharge Worklist (case manager). Lakebase table lsh_capacity_serving.bed_board, filtered by hospital and unit.
   Rank by long_stay_risk descending. Columns: bed, patient_name, age, drg_description, day of stay (los_so_far_days) vs expected_los, risk_tier badge (High red, Medium amber, Low green, Observation grey), predicted_discharge_ts, open_barrier_category, risk_drivers.
   Row detail: huddle_note (AI-written), risk drivers, barrier, and two actions that write to Lakebase:
     - Confirm discharge plan: upsert capacity_ops.discharge_plan (encounter_id, target_discharge_ts, plan_status in Planned/Confirmed/Delayed/Discharged, delay_reason, updated_by = signed-in user, updated_at = now()).
     - Log action: insert capacity_ops.discharge_actions (encounter_id, action_type in Escalate barrier/SNF referral sent/Family meeting/Transport booked/Pharmacy meds-to-beds, note, owner_role, created_by = signed-in user).
   Show each patient's latest plan status and action count by left-joining capacity_ops tables on encounter_id.

3. LOS Performance (CMO / quality). SQL warehouse queries on the metric views (MEASURE() syntax):
   schauhan_workspace_catalog.lsh_capacity_gold.los_metrics, schauhan_workspace_catalog.lsh_capacity_gold.capacity_metrics, schauhan_workspace_catalog.lsh_capacity_gold.barrier_metrics. Default period: Fiscal Year = 'FY2026' (LSH fiscal year starts Sep 1).
   KPI cards: LOS O/E Index, Excess Days, Avoidable Days, Discharges Before Noon %, Long Stay Rate, Occupancy %.
   Primary visual: LOS O/E Index by Facility (bar, reference line at 1.0). Then monthly Excess Days trend, and Avoidable Days by Barrier Category (pareto).
   Use *.obo.sql queries so the viewer's own Unity Catalog permissions apply; add sql to user_api_scopes.

4. Ask. Embed the Genie space 01f1d49b09321c7db9f9e412c68bde9e with the sample questions as starter chips.

AI insight card on pages 1-3: a short summary of the page's numbers (Genie or the data itself; never invent numbers).

Rules: never grant permissions yourself; list the grants I need. Validate with apps deploy --skip-validation --auto-approve and fix errors from apps logs lsh-capacity-command-center.
```


---
### Cell 5: Step 3 - The deployed app: wiring, access and operations (markdown)

## How the deployed app is wired

| Page | Data | Identity used |
| --- | --- | --- |
| Command Center | Lakebase `lsh_capacity_serving.unit_status` | App service principal |
| Discharge Worklist | Lakebase `lsh_capacity_serving.bed_board`; writes `capacity_ops.discharge_plan` and `capacity_ops.discharge_actions` | App service principal; `updated_by` / `created_by` = signed-in user's email |
| LOS Performance | Metric views `los_metrics`, `capacity_metrics`, `barrier_metrics` via SQL warehouse | **Signed-in user** (user authorization scope `sql`) |
| Ask | Genie space from notebook 07 | **Signed-in user** (scope `dashboards.genie`) |

* **Configuration** lives in `app.yaml` env vars: warehouse ID, Genie space ID, gold schema, Lakebase endpoint, host and database, plus the LSH light theme.
* **Lakebase access:** the app's service principal (`bf14a74d-7000-4d1b-a198-a414f7903a96`) has a Postgres role with SELECT on the synced tables and SELECT / INSERT / UPDATE on `capacity_ops` (notebook 06, Step 6).
* **Fallback:** if Lakebase is unreachable, the app reads the same tables from gold through the warehouse, shows a banner and turns off saving.

## Access to set up (owner or admin)

1. **App users:** give **CAN USE** on the app only to the care team. The worklist shows (synthetic) patient names.
2. **For each app user:** `CAN USE` on the SQL warehouse, `CAN RUN` on the Genie space, and `USE CATALOG` / `USE SCHEMA` / `SELECT` on the gold schema. Notebook 04 Step 6 grants the gold schema to the `lsh_capacity_analysts` and `lsh_capacity_care_team` groups once they exist.
3. **First visit:** each user is asked to consent to the app using their identity for SQL and Genie.

## Operations

* **Daily refresh:** the job **LSH Capacity Command Center - Daily Refresh** runs 02 → 03 → 04 → 05 → 06 at 6:30 AM Central. It was created **paused**: enable the schedule on the job page. Notebook 05 registers a new model version on every run, so for a daily schedule consider retraining weekly instead.
* **Code changes:** edit the files in the source folder, then redeploy from the app page (or `databricks apps deploy lsh-capacity-command-center --source-code-path <source folder>`). In the Streamlit `st.markdown` style block, keep CSS free of blank lines, or Markdown shows it as text.

## 10-minute demo path

1. **Command Center:** LMH is ~94% full, a few units have 2 or fewer beds forecast tonight, and ~262 discharges are expected today system-wide.
2. **Discharge Worklist:** LMH 14 West Oncology. The top patient has a 75% long-stay risk (complex discharge, 75+, lives alone). Read the AI huddle note, then confirm a 2 PM discharge and log a family meeting. The write lands in Lakebase in milliseconds.
3. **LOS Performance:** Willow Creek and LMH have the highest LOS O/E. SNF placement is the top barrier. Prairie Central's before-noon discharges jumped after its March 2026 initiative.
4. **Ask:** "How many high-risk patients are past their expected discharge at LMH right now?" Genie answers from the de-identified bed board.
5. **Governance:** in Catalog Explorer, show the PHI tags, the MRN mask, the facility row filter and the lineage from the landing files to `bed_board`.