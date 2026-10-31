# 07 - Genie Agent for Capacity and LOS

Exported from the Databricks workspace on 2026-10-31 06:17 UTC with each cell's **saved output** from its last run. Synthetic data; organization and hospital names anonymized.


---
### Cell 1: Overview (markdown)

# 07 - Genie Agent for Capacity and LOS

Creates the **LSH Capacity and LOS Analyst** Genie space: the "Ask" feature of the app and a self-service analyst for capacity leaders. It answers both trend questions ("Which hospital has the worst LOS O/E this year?") and right-now questions ("How many high-risk patients are on LMH units this morning?").

| Data source | Type | Answers |
| --- | --- | --- |
| `los_metrics` | metric view | LOS, O/E index, excess and avoidable days, long stays, before-noon discharges, readmissions |
| `capacity_metrics` | metric view | Occupancy, average daily census, admissions, discharges, unit-days above 95% |
| `barrier_metrics` | metric view | CarePort barriers and avoidable days by category |
| `bed_board_deid` | view | Current inpatients with risk scores and predicted discharge, **no names or MRNs** |
| `unit_status` | table | Unit capacity right now and forecast open beds tonight |

**Governance:** the space has no access to names, MRNs, bronze or silver. Genie runs every query as the person asking, so the Unity Catalog grants, masks and row filters from notebook 04 still apply.

**What gets configured:** instructions (fiscal calendar, metric definitions, hospital names), sample questions, 3 example queries, and 14 benchmark questions with answer SQL. The last step asks Genie 4 benchmark questions through the Conversation API and checks its answers against SQL.

**How to run (about 3 minutes):** run 04 and 05 first, then **Run all**. Re-running updates the same space instead of creating a duplicate.

---
### Cell 2: Parameters

```python
dbutils.widgets.text("catalog", "schauhan_workspace_catalog", "1. Catalog")
dbutils.widgets.text("prefix", "lsh_capacity", "2. Schema prefix")
dbutils.widgets.text("warehouse_id", "", "3. SQL warehouse ID (blank = pick one)")
```

_No saved output._


---
### Cell 3: Step 1 - Setup and pick a SQL warehouse

```python
import re, json, time
from databricks.sdk import WorkspaceClient

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("prefix").strip()
for label, val in [("catalog", CATALOG), ("prefix", PREFIX)]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", val or ""):
        raise ValueError(f"Enter a {label} with only letters, numbers and underscores.")
spark.conf.set("spark.sql.session.timeZone", "UTC")
G = f"{CATALOG}.{PREFIX}_gold"
for obj in ["los_metrics", "capacity_metrics", "barrier_metrics", "bed_board_deid", "unit_status"]:
    if not spark.catalog.tableExists(f"{G}.{obj}"):
        raise ValueError(f"{G}.{obj} not found. Run notebooks 04, 05 and 06 first.")
SNAP = spark.sql(f"SELECT max(snapshot_ts) AS s FROM {G}.bed_board_deid").first().s

w = WorkspaceClient()
ME = w.current_user.me().user_name
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id").strip()
if not WAREHOUSE_ID:   # prefer a running serverless warehouse
    whs = sorted(w.warehouses.list(), key=lambda x: (not x.enable_serverless_compute, str(x.state) != "State.RUNNING"))
    if not whs:
        raise RuntimeError("No SQL warehouse found. Ask for CAN USE on a SQL warehouse and enter its ID.")
    WAREHOUSE_ID = whs[0].id
    print(f"Using SQL warehouse {whs[0].name} ({WAREHOUSE_ID})")
TITLE = f"LSH Capacity and LOS Analyst ({PREFIX})"
print("Data snapshot:", SNAP)
```

**Output:**

```text
Using SQL warehouse Serverless Starter Warehouse (a87df88e51fce476)
Data snapshot: 2026-10-30 07:00:00
```


---
### Cell 4: Step 2 - Instructions, sample questions, example queries and benchmarks

```python
LOS, CAP, BAR, BB, US = (f"{G}.{x}" for x in ["los_metrics", "capacity_metrics", "barrier_metrics", "bed_board_deid", "unit_status"])

INSTRUCTIONS = f"""You are the capacity and length-of-stay analyst for Lakeshore Health (LSH) capacity leaders, case managers and nursing directors. All data is synthetic. The current snapshot is the 7:00 AM capacity huddle on {SNAP:%A, %B %d, %Y}.

Data sources:
- los_metrics (metric view): discharged encounters by discharge date. LOS, LOS O/E index, excess days, avoidable days, long-stay rate, discharges before noon, 30-day readmissions.
- capacity_metrics (metric view): one row per unit per day with the midnight census and staffed beds. Occupancy, average daily census, admissions, discharges, unit-days above 95%.
- barrier_metrics (metric view): CarePort avoidable-day barriers by the date they were opened.
- bed_board_deid: everyone in the hospital right now, with long_stay_risk (0-1), risk_tier (High >= 0.5, Medium >= 0.3, Low, or Observation), predicted_discharge_ts, p_discharge_today, days_past_expected, open_barrier_category and risk_drivers. It has no names or MRNs; never try to identify a patient.
- unit_status: one row per unit right now: staffed_beds, patients_now, open_beds_now, high_risk_patients, expected_discharges_today, expected_admissions_today, forecast_open_beds_tonight.
Use bed_board_deid and unit_status for "now", "today", "this morning" or "currently" questions. Use the metric views for trends and periods.

Fiscal calendar: the LSH fiscal year starts September 1. FY2026 = Sep 2025 - Aug 2026. Q1 = Sep-Nov, Q2 = Dec-Feb, Q3 = Mar-May, Q4 = Jun-Aug. "This fiscal year" = FY2027, which has only September and October 2026 so far, so say that. "Last fiscal year" = FY2026. If the period is unclear, use FY2026 and state the assumption.

Metrics: always use the predefined measures with MEASURE() and never recompute ratios. LOS O/E Index above 1.0 means patients stay longer than expected (Vizient-style). Excess Days are inpatient days beyond the expected LOS. Occupancy % is midnight census / staffed beds; LSH considers a unit above 95% "full". LOS, O/E, excess days, long stays and before-noon discharges are inpatient-only.

Names: Memorial, LMH or Lakeshore Memorial = LMH; Prairie Central = PCH; Riverbend = RBH; Bluffside = BSH; Oak Ridge = ORH; Fox Valley = FVH; Willow Creek = WCH; Cornfield = CFH. Units are named like '14 West Oncology' with codes like LMH-14W.

Answers: start with a one-sentence takeaway for a capacity leader, then the table. Show O/E with 2 decimals, percentages with 1 decimal, days with 1 decimal. Sort time series oldest to newest."""

SAMPLES = [
    "Which hospital had the highest LOS O/E index in FY2026?",
    "How many high-risk patients are in each hospital right now?",
    "Which units are forecast to have the fewest open beds tonight?",
    "What barrier category causes the most avoidable days?",
    "Show the discharges before noon trend at Prairie Central in FY2026",
    "Which LMH units ran above 95% occupancy most often in FY2026?",
]
EXAMPLES = [
    ("What was the LOS O/E index and excess days by hospital in FY2026?",
     f"SELECT `Facility`, MEASURE(`LOS O/E Index`) AS los_oe_index, MEASURE(`Excess Days`) AS excess_days, MEASURE(`Inpatient Discharges`) AS inpatient_discharges\nFROM {LOS}\nWHERE `Fiscal Year` = 'FY2026'\nGROUP BY ALL\nORDER BY los_oe_index DESC"),
    ("Show monthly occupancy at LMH for the last 12 months",
     f"SELECT `Month`, MEASURE(`Occupancy %`) AS occupancy\nFROM {CAP}\nWHERE `Facility Code` = 'LMH' AND `Month` >= add_months(date_trunc('MONTH', DATE'{SNAP:%Y-%m-%d}'), -12)\n  AND `Month` < date_trunc('MONTH', DATE'{SNAP:%Y-%m-%d}')\nGROUP BY ALL\nORDER BY `Month` ASC"),
    ("Which high-risk patients at LMH are already past their expected discharge?",
     f"SELECT unit_name, bed, age, drg_description, round(los_so_far_days, 1) AS day_of_stay, expected_los,\n       round(long_stay_risk, 2) AS long_stay_risk, open_barrier_category, risk_drivers\nFROM {BB}\nWHERE facility_code = 'LMH' AND risk_tier = 'High' AND days_past_expected > 0\nORDER BY long_stay_risk DESC"),
]
BENCHMARKS = [
    ("Which hospital had the highest LOS O/E index in FY2026?",
     f"SELECT `Facility`, MEASURE(`LOS O/E Index`) AS los_oe_index\nFROM {LOS}\nWHERE `Fiscal Year` = 'FY2026'\nGROUP BY ALL\nORDER BY los_oe_index DESC\nLIMIT 1"),
    ("How many excess days did Lakeshore Memorial have in FY2026?",
     f"SELECT MEASURE(`Excess Days`) AS excess_days\nFROM {LOS}\nWHERE `Facility Code` = 'LMH' AND `Fiscal Year` = 'FY2026'\nGROUP BY ALL"),
    ("What was the discharges before noon percentage at Prairie Central by month in FY2026?",
     f"SELECT `Discharge Month`, MEASURE(`Discharges Before Noon %`) AS before_noon\nFROM {LOS}\nWHERE `Facility Code` = 'PCH' AND `Fiscal Year` = 'FY2026'\nGROUP BY ALL\nORDER BY `Discharge Month` ASC"),
    ("What was average occupancy at LMH by fiscal quarter in FY2026?",
     f"SELECT `Fiscal Quarter`, MEASURE(`Occupancy %`) AS occupancy\nFROM {CAP}\nWHERE `Facility Code` = 'LMH' AND `Fiscal Year` = 'FY2026'\nGROUP BY ALL\nORDER BY `Fiscal Quarter` ASC"),
    ("Which 5 LMH units had the highest occupancy in FY2026?",
     f"SELECT `Unit`, MEASURE(`Occupancy %`) AS occupancy\nFROM {CAP}\nWHERE `Facility Code` = 'LMH' AND `Fiscal Year` = 'FY2026'\nGROUP BY ALL\nORDER BY occupancy DESC\nLIMIT 5"),
    ("Which barrier category caused the most avoidable days in FY2026?",
     f"SELECT `Barrier Category`, MEASURE(`Avoidable Days`) AS avoidable_days\nFROM {BAR}\nWHERE `Fiscal Year` = 'FY2026'\nGROUP BY ALL\nORDER BY avoidable_days DESC\nLIMIT 1"),
    ("Compare avoidable days and barriers in FY2025 vs FY2026",
     f"SELECT `Fiscal Year`, MEASURE(`Barriers`) AS barriers, MEASURE(`Avoidable Days`) AS avoidable_days\nFROM {BAR}\nWHERE `Fiscal Year` IN ('FY2025', 'FY2026')\nGROUP BY ALL\nORDER BY `Fiscal Year` ASC"),
    ("What was the long-stay rate by age band in FY2026?",
     f"SELECT `Age Band`, MEASURE(`Long Stay Rate`) AS long_stay_rate\nFROM {LOS}\nWHERE `Fiscal Year` = 'FY2026'\nGROUP BY ALL\nORDER BY `Age Band` ASC"),
    ("Which 5 DRGs at LMH had the most excess days in FY2026?",
     f"SELECT `DRG Description`, MEASURE(`Excess Days`) AS excess_days\nFROM {LOS}\nWHERE `Facility Code` = 'LMH' AND `Fiscal Year` = 'FY2026'\nGROUP BY ALL\nORDER BY excess_days DESC\nLIMIT 5"),
    ("What was the 30-day readmission rate by discharge disposition in FY2026?",
     f"SELECT `Discharge Disposition`, MEASURE(`30-Day Readmission Rate`) AS readmit_rate\nFROM {LOS}\nWHERE `Fiscal Year` = 'FY2026'\nGROUP BY ALL\nORDER BY readmit_rate DESC"),
    ("How many high-risk patients are in each hospital right now?",
     f"SELECT facility_code, count(*) AS high_risk_patients\nFROM {BB}\nWHERE risk_tier = 'High'\nGROUP BY facility_code\nORDER BY high_risk_patients DESC"),
    ("How many patients at LMH are past their expected discharge right now?",
     f"SELECT count(*) AS past_expected\nFROM {BB}\nWHERE facility_code = 'LMH' AND days_past_expected > 0"),
    ("Which 5 units are forecast to have the fewest open beds tonight?",
     f"SELECT facility_code, unit_name, staffed_beds, patients_now, forecast_open_beds_tonight\nFROM {US}\nORDER BY forecast_open_beds_tonight ASC\nLIMIT 5"),
    ("How many discharges are expected today at each hospital?",
     f"SELECT facility_code, round(sum(expected_discharges_today), 1) AS expected_discharges_today\nFROM {US}\nGROUP BY facility_code\nORDER BY expected_discharges_today DESC"),
]

# Every example and benchmark must run and return rows before it goes into the space
for q, s in EXAMPLES + BENCHMARKS:
    n = spark.sql(s).count()
    assert n > 0, f"No rows for: {q}"
print(f"Validated {len(EXAMPLES)} example queries and {len(BENCHMARKS)} benchmark queries: all run and return rows")
```

**Output:**

```text
Validated 3 example queries and 14 benchmark queries: all run and return rows
```


---
### Cell 5: Step 3 - Create or update the Genie space

```python
def gid(prefix, i):
    return f"{prefix}{i:02x}".rjust(32, "0")
def sql_lines(s):
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

TABLE_DESC = {
    BB: "Current inpatients at the 7:00 AM snapshot with ML long-stay risk, predicted discharge, open barriers and risk drivers. De-identified: no names or MRNs.",
    US: "One row per nursing unit right now: staffed beds, patients, open beds, high-risk patients, expected discharges and admissions today, forecast open beds tonight.",
}
MV_DESC = {
    BAR: "CarePort avoidable-day barriers by opened date: barriers, open barriers, avoidable days by category, facility and unit.",
    CAP: "Daily unit capacity: occupancy, average daily census, admissions, discharges, unit-days above 95%. By date, month, fiscal period, facility, unit and unit type.",
    LOS: "Length of stay KPIs by discharge date: ALOS, LOS O/E index, excess days, avoidable days, long-stay rate, discharges before noon, 30-day readmissions.",
}
ENTITY_COLS = {LOS: ["Facility", "Facility Code", "Unit", "Unit Type", "Service Line", "DRG Description", "Discharge Disposition", "Barrier Category"],
               CAP: ["Facility", "Facility Code", "Unit", "Unit Type"],
               BAR: ["Facility", "Facility Code", "Unit", "Barrier Category"]}
space = {
    "version": 2,
    "config": {"sample_questions": [{"id": gid("a1", i), "question": [q]} for i, q in enumerate(SAMPLES)]},
    "data_sources": {
        "tables": [{"identifier": t, "description": [TABLE_DESC[t]]} for t in sorted(TABLE_DESC)],
        "metric_views": [{"identifier": mv, "description": [MV_DESC[mv]],
                          "column_configs": [{"column_name": c, "enable_entity_matching": True, "enable_format_assistance": True}
                                             for c in sorted(ENTITY_COLS[mv])]} for mv in sorted(MV_DESC)],
    },
    "instructions": {
        "text_instructions": [{"id": gid("b1", 0), "content": [INSTRUCTIONS]}],
        "example_question_sqls": [{"id": gid("c1", i), "question": [q], "sql": sql_lines(s)} for i, (q, s) in enumerate(EXAMPLES)],
    },
    "benchmarks": {"questions": [{"id": gid("d1", i), "question": [q], "answer": [{"format": "SQL", "content": sql_lines(s)}]}
                                 for i, (q, s) in enumerate(BENCHMARKS)]},
}
body = {"title": TITLE, "warehouse_id": WAREHOUSE_ID, "serialized_space": json.dumps(space),
        "description": f"Synthetic LSH capacity and length-of-stay analyst for the Capacity Command Center app. Metric views and a de-identified bed board in {G}. No PHI."}

SPACE_ID = None
page = {}
while True:   # re-running updates this prefix's existing space instead of creating a duplicate
    r = w.api_client.do("GET", "/api/2.0/genie/spaces", query={"page_size": 100, **page})
    SPACE_ID = next((s["space_id"] for s in r.get("spaces", []) if s.get("title") == TITLE), None)
    if SPACE_ID or not r.get("next_page_token"):
        break
    page = {"page_token": r["next_page_token"]}
if SPACE_ID:
    w.api_client.do("PATCH", f"/api/2.0/genie/spaces/{SPACE_ID}", body=body)
    print("Updated Genie space", SPACE_ID)
else:
    SPACE_ID = w.api_client.do("POST", "/api/2.0/genie/spaces", body={**body, "parent_path": f"/Users/{ME}/LSH Capacity and LOS Prototype"})["space_id"]
    print("Created Genie space", SPACE_ID)

ss = json.loads(w.api_client.do("GET", f"/api/2.0/genie/spaces/{SPACE_ID}", query={"include_serialized_space": "true"})["serialized_space"])
sources = [d["identifier"] for k in ("tables", "metric_views") for d in ss["data_sources"].get(k, [])]   # the API may list metric views under tables
assert set(sources) == set(TABLE_DESC) | set(MV_DESC), sources
print("Configured:", len(sources), "data sources (3 metric views, 2 tables),",
      len(ss["config"]["sample_questions"]), "sample questions,", len(ss["instructions"]["example_question_sqls"]), "example queries,",
      len(ss.get("benchmarks", {}).get("questions", [])), "benchmarks")
```

**Output:**

```text
Updated Genie space 01f1d49b09321c7db9f9e412c68bde9e
Configured: 5 data sources (3 metric views, 2 tables), 6 sample questions, 3 example queries, 14 benchmarks
```


---
### Cell 6: Step 4 - Ask Genie 4 benchmark questions and check the answers

```python
def ask_genie(question, timeout_s=180):
    """Conversation API: returns (generated SQL, result rows) for one question."""
    r = w.api_client.do("POST", f"/api/2.0/genie/spaces/{SPACE_ID}/start-conversation", body={"content": question})
    cid, mid = r["conversation_id"], r["message_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = w.api_client.do("GET", f"/api/2.0/genie/spaces/{SPACE_ID}/conversations/{cid}/messages/{mid}")
        if m["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(3)
    att = next((a for a in m.get("attachments", []) if a.get("query")), None)
    if m["status"] != "COMPLETED" or not att:
        return None, None
    res = w.api_client.do("GET", f"/api/2.0/genie/spaces/{SPACE_ID}/conversations/{cid}/messages/{mid}/attachments/{att['attachment_id']}/query-result")
    return att["query"]["query"], res["statement_response"]["result"].get("data_array", [])

def first_number(rows):
    for v in (rows or [[None]])[0]:
        try:
            return float(v)
        except (TypeError, ValueError):
            continue

CHECKS = [BENCHMARKS[0], BENCHMARKS[1], BENCHMARKS[5], BENCHMARKS[11]]   # O/E leader, LMH excess days, top barrier, LMH past expected
results = []
for q, expected_sql in CHECKS:
    exp_row = [str(v) for v in spark.sql(expected_sql).first()]
    gen_sql, rows = ask_genie(q)
    got = rows[0] if rows else []
    exp_num, got_num = first_number([exp_row]), first_number(rows)
    text_ok = all(v in " ".join(map(str, got)) for v in exp_row if not v.replace(".", "").replace("-", "").isdigit())
    num_ok = exp_num is None or (got_num is not None and abs(got_num - exp_num) <= max(0.005 * abs(exp_num), 0.01))
    results.append((q, ", ".join(exp_row), ", ".join(map(str, got)) if got else "(no answer)", bool(rows) and text_ok and num_ok))
display(spark.createDataFrame(results, "question string, expected string, genie_answer string, matches boolean"))
print(f"Genie matched {sum(r[3] for r in results)} of {len(results)} benchmark answers")
```

**Output (table):**

| question | expected | genie_answer | matches |
| --- | --- | --- | --- |
| Which hospital had the highest LOS O/E index in FY2026? | Willow Creek Hospital, 1.339034935201206 | Willow Creek Hospital, 1.339034935201206 | True |
| How many excess days did Lakeshore Memorial have in FY2026? | 42368.77299999988 | 42368.77299999988 | True |
| Which barrier category caused the most avoidable days in FY2026? | SNF placement, 14791 | SNF placement, 14791 | True |
| How many patients at LMH are past their expected discharge right now? | 121 | 121 | True |

**Output:**

```text
Genie matched 4 of 4 benchmark answers
```


---
### Cell 7: Step 5 - Values for the app

```python
host = spark.conf.get("spark.databricks.workspaceUrl", "")
display(spark.createDataFrame([
    ("Genie space ID", SPACE_ID), ("Genie space name", TITLE), ("Genie space link", f"https://{host}/genie/rooms/{SPACE_ID}"),
    ("SQL warehouse ID", WAREHOUSE_ID), ("Metric views", f"{LOS}, {CAP}, {BAR}"), ("De-identified tables", f"{BB}, {US}"),
], "item string, value string"))
print("To share: give the app users CAN RUN on this space, and SELECT on the gold schema (notebook 04, Step 6).")
```

**Output (table):**

| item | value |
| --- | --- |
| Genie space ID | 01f1d49b09321c7db9f9e412c68bde9e |
| Genie space name | LSH Capacity and LOS Analyst (lsh_capacity) |
| Genie space link | https://fevm-schauhan-workspace.cloud.databricks.com/genie/rooms/01f1d49b09321c7db9f9e412c68bde9e |
| SQL warehouse ID | a87df88e51fce476 |
| Metric views | schauhan_workspace_catalog.lsh_capacity_gold.los_metrics, schauhan_workspace_catalog.lsh_capacity_gold.capacity_metrics, schauhan_workspace_catalog.lsh_capacity_gold.barrier_metrics |
| De-identified tables | schauhan_workspace_catalog.lsh_capacity_gold.bed_board_deid, schauhan_workspace_catalog.lsh_capacity_gold.unit_status |

**Output:**

```text
To share: give the app users CAN RUN on this space, and SELECT on the gold schema (notebook 04, Step 6).
```
