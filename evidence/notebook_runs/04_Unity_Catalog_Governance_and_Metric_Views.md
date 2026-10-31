# 04 - Unity Catalog Governance and Metric Views

Exported from the Databricks workspace on 2026-10-31 06:17 UTC with each cell's **saved output** from its last run. Synthetic data; organization and hospital names anonymized.


---
### Cell 1: Overview (markdown)

# 04 - Unity Catalog Governance and Metric Views

Governs the capacity and LOS data in **Unity Catalog** and publishes the governed business definitions that Genie (07) and the app (08) use.

| Control | What this notebook does |
| --- | --- |
| **Documentation** | Table and column comments on the gold star schema, so Catalog Explorer, Genie and the app builder understand every field |
| **Classification tags** | `lsh_data_domain`, `lsh_quality_tier` and `lsh_synthetic` on the schemas; `lsh_contains_phi` on tables and an `lsh_phi` tag on every identity column (MRN, name, birth date) in silver and gold |
| **Column masks** | MRN shows only the last 4 digits, names show only the first initial, birth date shows only the year, unless you are the data steward or in the care team group |
| **Row filter** | `current_inpatients` shows a user only the hospitals they're allowed to see (a group per facility, or the all-facilities group) |
| **Least-privilege grants** | Analysts get the gold schema and metric views, but no bronze or silver (which hold unmasked identity columns) |
| **De-identified serving view** | `bed_board_deid` (created in notebook 05) has no names or MRNs; it's the only patient-level object Genie can see |
| **Lineage** | Unity Catalog records lineage automatically from the raw files to the app tables; the last step shows it |
| **Metric views** | `los_metrics`, `capacity_metrics` and `barrier_metrics`: one governed definition of LOS O/E, excess days, occupancy and more |

**Groups (optional):** the notebook uses `<prefix>_analysts`, `<prefix>_care_team`, `<prefix>_all_facilities` and `<prefix>_facility_<code>` (for example `lsh_capacity_facility_bsh`). Grants to groups that don't exist yet are skipped with a message. The person who runs the notebook is the **data steward** and always sees unmasked data, so the pipeline (05, 06) keeps working.

**Upgrade path:** these are table-level masks and filters. With governed tags (account admin), the same functions can be applied as **ABAC policies** that cover every table tagged `lsh_phi`, including new ones.

**How to run (about 1 minute):** run after notebook 03, every time 03 runs, because 03 recreates the gold tables.

---
### Cell 2: Parameters

```python
dbutils.widgets.text("catalog", "schauhan_workspace_catalog", "1. Catalog")
dbutils.widgets.text("prefix", "lsh_capacity", "2. Schema prefix")
```

_No saved output._


---
### Cell 3: Step 1 - Setup

```python
import re
from pyspark.sql import functions as F
from databricks.sdk import WorkspaceClient

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("prefix").strip()
for label, val in [("catalog", CATALOG), ("prefix", PREFIX)]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", val or ""):
        raise ValueError(f"Enter a {label} with only letters, numbers and underscores.")
q = lambda s: f"`{CATALOG}`.`{PREFIX}_{s}`"
BRONZE, SILVER, GOLD = q("bronze"), q("silver"), q("gold")
GOLD_NAME = f"{CATALOG}.{PREFIX}_gold"
if not spark.catalog.tableExists(f"{GOLD_NAME}.fact_encounter"):
    raise ValueError("Gold tables not found. Run notebook 03 first.")

STEWARD = spark.sql("SELECT current_user()").first()[0]
GROUPS = {"analysts": f"{PREFIX}_analysts", "care_team": f"{PREFIX}_care_team", "all_facilities": f"{PREFIX}_all_facilities"}
w = WorkspaceClient()
EXISTING_GROUPS = {g.display_name for g in w.groups.list(attributes="displayName") if g.display_name}
print(f"Data steward: {STEWARD}")
for role, g in GROUPS.items():
    print(f"  {role:15s} group {g}: {'found' if g in EXISTING_GROUPS else 'not found (grants to it are skipped)'}")
```

**Output:**

```text
Data steward: sugandha.chauhan@databricks.com
  analysts        group lsh_capacity_analysts: not found (grants to it are skipped)
  care_team       group lsh_capacity_care_team: not found (grants to it are skipped)
  all_facilities  group lsh_capacity_all_facilities: not found (grants to it are skipped)
```


---
### Cell 4: Step 2 - Documentation: table and column comments

```python
COLUMN_COMMENTS = {
    "fact_encounter": {
        "encounter_id": "Epic encounter CSN. One row per inpatient or observation stay",
        "los_days": "Observed length of stay in days (discharge - admit). Null while the patient is in the hospital",
        "expected_los": "Vizient-style risk-adjusted expected LOS in days for the DRG, age and complexity",
        "excess_days": "Inpatient only: max(0, observed LOS - expected LOS). The days a bed was used beyond expectation",
        "is_long_stay": "Inpatient only: 1 when observed LOS is 2+ days above expected (the label the model in notebook 05 predicts)",
        "avoidable_days": "CarePort avoidable days: days the patient stayed only because of a non-clinical barrier",
        "discharged_before_noon": "1 when the discharge time is before 12:00",
        "readmit_30d": "Inpatient, non-expired discharges: 1 when the same patient is readmitted within 30 days",
        "complex_discharge": "Case management screen at admission flagged a complex discharge (1/0)",
        "is_in_house": "True when the patient is still in the hospital at the data snapshot"},
    "fact_unit_census_day": {
        "midnight_census": "Patients on the unit at midnight at the end of the day (the standard census count)",
        "staffed_beds": "Staffed beds on the unit that day (effective-dated, includes surge beds)",
        "occupancy_rate": "midnight_census / staffed_beds for the unit-day. Can exceed 1.0 when hallway beds are used",
        "discharges_before_noon": "Discharges from the unit that day with a discharge time before 12:00"},
    "current_inpatients": {
        "los_so_far_days": "Days since admission at the snapshot time",
        "expected_discharge_ts": "Admit time + expected LOS",
        "days_past_expected": "Days the patient is already past the expected discharge time (0 if not yet)",
        "open_barrier_category": "Open CarePort barrier blocking discharge, if any",
        "mrn": "Medical record number (synthetic). PHI: masked unless steward or care team",
        "patient_name": "Patient name (synthetic). PHI: masked unless steward or care team"},
    "dim_patient": {
        "mrn": "Medical record number (synthetic). PHI: masked", "first_name": "PHI: masked", "last_name": "PHI: masked",
        "patient_name": "PHI: masked", "birth_date": "PHI: masked to January 1 of the birth year"},
}
for table, cols in COLUMN_COMMENTS.items():
    for col, comment in cols.items():
        spark.sql(f"ALTER TABLE {GOLD}.{table} ALTER COLUMN {col} COMMENT \"{comment}\"")
print(f"Commented {sum(len(c) for c in COLUMN_COMMENTS.values())} columns on {len(COLUMN_COMMENTS)} gold tables")
```

**Output:**

```text
Commented 25 columns on 4 gold tables
```


---
### Cell 5: Step 3 - Classification tags (domain, quality tier, PHI)

```python
# Tag keys are namespaced with lsh_ so they don't collide with governed tags that already exist in the account
# (for example, a governed data_domain tag that only allows other values).
for layer, tier in [("bronze", "bronze"), ("silver", "silver"), ("gold", "gold")]:
    for tag, value in [("lsh_data_domain", "capacity_los"), ("lsh_quality_tier", tier), ("lsh_synthetic", "true")]:
        spark.sql(f"SET TAG ON SCHEMA {q(layer)} {tag} = {value}")

# PHI columns, wherever they live
PHI = {
    (SILVER, "encounters"): {"mrn": "identifier", "first_name": "name", "last_name": "name", "birth_date": "dob"},
    (SILVER, "adt_events"): {"mrn": "identifier"},
    (SILVER, "adt_stays"): {"mrn": "identifier"},
    (GOLD, "dim_patient"): {"mrn": "identifier", "first_name": "name", "last_name": "name", "patient_name": "name", "birth_date": "dob"},
    (GOLD, "current_inpatients"): {"mrn": "identifier", "patient_name": "name"},
}
for (schema, table), cols in PHI.items():
    spark.sql(f"SET TAG ON TABLE {schema}.{table} lsh_contains_phi = true")
    for col, kind in cols.items():
        spark.sql(f"SET TAG ON COLUMN {schema}.{table}.{col} lsh_phi = {kind}")

display(spark.sql(f"""SELECT schema_name, table_name, column_name, tag_value AS phi
                      FROM `{CATALOG}`.information_schema.column_tags
                      WHERE schema_name LIKE '{PREFIX}_%' AND tag_name = 'lsh_phi' ORDER BY 1, 2, 3"""))
```

**Output (table):**

| schema_name | table_name | column_name | phi |
| --- | --- | --- | --- |
| lsh_capacity_gold | bed_board | mrn | identifier |
| lsh_capacity_gold | bed_board | patient_name | name |
| lsh_capacity_gold | current_inpatients | mrn | identifier |
| lsh_capacity_gold | current_inpatients | patient_name | name |
| lsh_capacity_gold | dim_patient | birth_date | dob |
| lsh_capacity_gold | dim_patient | first_name | name |
| lsh_capacity_gold | dim_patient | last_name | name |
| lsh_capacity_gold | dim_patient | mrn | identifier |
| lsh_capacity_gold | dim_patient | patient_name | name |
| lsh_capacity_silver | adt_events | mrn | identifier |
| lsh_capacity_silver | adt_stays | mrn | identifier |
| lsh_capacity_silver | encounters | birth_date | dob |
| lsh_capacity_silver | encounters | first_name | name |
| lsh_capacity_silver | encounters | last_name | name |
| lsh_capacity_silver | encounters | mrn | identifier |


---
### Cell 6: Step 4 - Column masks and row filter

```python
CARE, ALLF = GROUPS["care_team"], GROUPS["all_facilities"]

# Who may see identity columns: the data steward (runs the pipeline) and the care team group
spark.sql(f"""CREATE OR REPLACE FUNCTION {GOLD}.phi_visible() RETURNS BOOLEAN
  COMMENT 'True for the data steward and members of {CARE}'
  RETURN current_user() = '{STEWARD}' OR is_account_group_member('{CARE}')""")
spark.sql(f"""CREATE OR REPLACE FUNCTION {GOLD}.mask_mrn(v STRING) RETURNS STRING
  COMMENT 'MRN mask: last 4 digits only' RETURN CASE WHEN {GOLD}.phi_visible() THEN v ELSE concat('NM****', right(v, 4)) END""")
spark.sql(f"""CREATE OR REPLACE FUNCTION {GOLD}.mask_name(v STRING) RETURNS STRING
  COMMENT 'Name mask: first initial only' RETURN CASE WHEN {GOLD}.phi_visible() THEN v ELSE concat(left(v, 1), '*****') END""")
spark.sql(f"""CREATE OR REPLACE FUNCTION {GOLD}.mask_dob(v DATE) RETURNS DATE
  COMMENT 'Birth date mask: Jan 1 of the birth year' RETURN CASE WHEN {GOLD}.phi_visible() THEN v ELSE make_date(year(v), 1, 1) END""")
# Which hospitals a user may see on the bed board
spark.sql(f"""CREATE OR REPLACE FUNCTION {GOLD}.facility_access(facility_code STRING) RETURNS BOOLEAN
  COMMENT 'Row filter: steward, {ALLF}, or {PREFIX}_facility_<code> for that hospital'
  RETURN current_user() = '{STEWARD}' OR is_account_group_member('{ALLF}')
         OR is_account_group_member(concat('{PREFIX}_facility_', lower(facility_code)))""")

MASKS = {"dim_patient": {"mrn": "mask_mrn", "first_name": "mask_name", "last_name": "mask_name", "patient_name": "mask_name", "birth_date": "mask_dob"},
         "current_inpatients": {"mrn": "mask_mrn", "patient_name": "mask_name"}}
for table, cols in MASKS.items():
    for col, fn in cols.items():
        spark.sql(f"ALTER TABLE {GOLD}.{table} ALTER COLUMN {col} SET MASK {GOLD}.{fn}")
spark.sql(f"ALTER TABLE {GOLD}.current_inpatients SET ROW FILTER {GOLD}.facility_access ON (facility_code)")

display(spark.sql(f"""SELECT table_name, column_name, mask_name FROM `{CATALOG}`.information_schema.column_masks
                      WHERE table_schema = '{PREFIX}_gold' ORDER BY 1, 2"""))
display(spark.sql(f"""SELECT table_name, filter_name, target_columns FROM `{CATALOG}`.information_schema.row_filters
                      WHERE table_schema = '{PREFIX}_gold'"""))
```

**Output (table):**

| table_name | column_name | mask_name |
| --- | --- | --- |
| current_inpatients | mrn | schauhan_workspace_catalog.lsh_capacity_gold.mask_mrn |
| current_inpatients | patient_name | schauhan_workspace_catalog.lsh_capacity_gold.mask_name |
| dim_patient | birth_date | schauhan_workspace_catalog.lsh_capacity_gold.mask_dob |
| dim_patient | first_name | schauhan_workspace_catalog.lsh_capacity_gold.mask_name |
| dim_patient | last_name | schauhan_workspace_catalog.lsh_capacity_gold.mask_name |
| dim_patient | mrn | schauhan_workspace_catalog.lsh_capacity_gold.mask_mrn |
| dim_patient | patient_name | schauhan_workspace_catalog.lsh_capacity_gold.mask_name |

**Output (table):**

| table_name | filter_name | target_columns |
| --- | --- | --- |
| current_inpatients | schauhan_workspace_catalog.lsh_capacity_gold.facility_access | facility_code |


---
### Cell 7: Step 5 - Check the masks: what you see vs what a non-care-team analyst sees

```python
# You are the steward, so the table returns real values. The second set of columns applies the same
# redaction expressions the mask functions use for everyone else, so you can see both views side by side.
check = spark.sql(f"""
  SELECT mrn, patient_name, birth_date,
         concat('NM****', right(mrn, 4)) AS mrn_as_analyst, concat(left(patient_name, 1), '*****') AS name_as_analyst,
         make_date(year(birth_date), 1, 1) AS birth_date_as_analyst
  FROM {GOLD}.dim_patient LIMIT 5""")
display(check)
visible = spark.sql(f"SELECT {GOLD}.phi_visible() AS v").first().v
rows_seen = spark.table(f"{GOLD}.current_inpatients").count()
rows_all = spark.table(f"{GOLD}.fact_encounter").filter("is_in_house").count()
assert visible and rows_seen == rows_all, "The steward should see every row unmasked"
print(f"Steward sees unmasked PHI and all {rows_seen:,} current inpatients. Non-members see the masked values above, "
      f"and only the hospitals their {PREFIX}_facility_<code> groups allow.")
```

**Output (table):**

| mrn | patient_name | birth_date | mrn_as_analyst | name_as_analyst | birth_date_as_analyst |
| --- | --- | --- | --- | --- | --- |
| NM01000000 | Sofia Quinn | 1995-04-21 | NM****0000 | S***** | 1995-01-01 |
| NM01000001 | Hannah Brooks | 1972-12-22 | NM****0001 | H***** | 1972-01-01 |
| NM01000002 | Linda Underwood | 1984-05-06 | NM****0002 | L***** | 1984-01-01 |
| NM01000004 | Thomas Zimmerman | 1936-09-20 | NM****0004 | T***** | 1936-01-01 |
| NM01000008 | Jennifer Young | 1961-04-15 | NM****0008 | J***** | 1961-01-01 |

**Output:**

```text
Steward sees unmasked PHI and all 1,064 current inpatients. Non-members see the masked values above, and only the hospitals their lsh_capacity_facility_<code> groups allow.
```


---
### Cell 8: Step 6 - Least-privilege grants (skipped for groups that don't exist)

```python
GRANTS = {
    GROUPS["analysts"]: [f"GRANT USE CATALOG ON CATALOG `{CATALOG}` TO `{{g}}`",
                         f"GRANT USE SCHEMA, SELECT, EXECUTE ON SCHEMA {GOLD} TO `{{g}}`"],          # gold only; masks still apply
    GROUPS["care_team"]: [f"GRANT USE CATALOG ON CATALOG `{CATALOG}` TO `{{g}}`",
                          f"GRANT USE SCHEMA, SELECT, EXECUTE ON SCHEMA {GOLD} TO `{{g}}`"],         # sees PHI through phi_visible()
}
result = []
for group, stmts in GRANTS.items():
    if group not in EXISTING_GROUPS:
        result.append((group, "skipped: group not found"))
        continue
    for s in stmts:
        spark.sql(s.format(g=group))
    result.append((group, "granted gold read access"))
display(spark.createDataFrame(result, "group string, result string"))
print("Bronze and silver are not granted to any group: they hold unmasked identity columns and are for the data engineering team only.")
```

**Output (table):**

| group | result |
| --- | --- |
| lsh_capacity_analysts | skipped: group not found |
| lsh_capacity_care_team | skipped: group not found |

**Output:**

```text
Bronze and silver are not granted to any group: they hold unmasked identity columns and are for the data engineering team only.
```


---
### Cell 9: Step 7 - Metric views: los_metrics, capacity_metrics, barrier_metrics

```python
PCT = "{type: percentage, decimal_places: {type: exact, places: 1}}"
NUM1 = "{type: number, decimal_places: {type: exact, places: 1}}"
NUM2 = "{type: number, decimal_places: {type: exact, places: 2}}"
FISCAL = """
  - name: Fiscal Year
    expr: dt.fiscal_year
    comment: LSH fiscal year, e.g. FY2026 = Sep 2025 - Aug 2026
    synonyms: [FY]
  - name: Fiscal Quarter
    expr: dt.fiscal_quarter
    comment: Q1 = Sep-Nov, Q2 = Dec-Feb, Q3 = Mar-May, Q4 = Jun-Aug
  - name: Fiscal Month
    expr: dt.fiscal_month
    comment: 1 = Sep ... 12 = Aug"""
FAC_UNIT = """
  - name: Facility
    expr: fac.facility_name
    synonyms: [hospital, site]
  - name: Facility Code
    expr: fac.facility_code
    comment: LMH, PCH, BSH, RBH, ORH, FVH, WCH, CFH
  - name: Facility Tier
    expr: fac.tier
  - name: Unit
    expr: un.unit_name
    synonyms: [nursing unit, floor]
  - name: Unit Code
    expr: un.unit_code
  - name: Unit Type
    expr: un.unit_type
    comment: ICU, Surgical ICU, CVICU, Neuro ICU, Telemetry, Med/Surg, Oncology, Ortho/Neuro, Transplant, Postpartum, Behavioral, Observation"""
JOINS = """
  - name: fac
    source: __GOLD__.dim_facility
    on: source.facility_key = fac.facility_key
  - name: un
    source: __GOLD__.dim_unit
    on: source.unit_key = un.unit_key"""

LOS_YAML = f"""
version: 1.1
comment: LSH length of stay KPIs by discharge date (synthetic). One row per discharged encounter. LOS O/E, excess and avoidable days, long stays and before-noon discharges are inpatient-only, like the Vizient LOS index.
source: __GOLD__.fact_encounter
filter: discharge_ts IS NOT NULL
joins:
  - name: dt
    source: __GOLD__.dim_date
    on: source.discharge_date_key = dt.date_key{JOINS}
  - name: dg
    source: __GOLD__.dim_drg
    on: source.drg_key = dg.drg_key
dimensions:
  - name: Discharge Month
    expr: dt.month_start_date
    synonyms: [month]
    format: {{type: date, date_format: locale_short_month}}
  - name: Discharge Date
    expr: dt.calendar_date{FISCAL}
  - name: Discharge Day of Week
    expr: dt.day_of_week{FAC_UNIT}
  - name: Service Line
    expr: dg.service_line
    synonyms: [specialty]
  - name: DRG
    expr: dg.drg
    comment: MS-DRG code
  - name: DRG Description
    expr: dg.drg_description
  - name: Patient Class
    expr: source.patient_class
    comment: Inpatient or Observation
  - name: Admit Source
    expr: source.admit_source
  - name: Financial Class
    expr: source.fin_class
    synonyms: [payer, payer class]
  - name: Discharge Disposition
    expr: source.disposition
    synonyms: [disposition, discharged to]
  - name: Age Band
    expr: CASE WHEN source.age_at_admit < 45 THEN '18-44' WHEN source.age_at_admit < 65 THEN '45-64' WHEN source.age_at_admit < 75 THEN '65-74' ELSE '75+' END
  - name: Complex Discharge
    expr: CASE WHEN source.complex_discharge = 1 THEN 'Yes' ELSE 'No' END
  - name: Barrier Category
    expr: coalesce(source.barrier_category, 'No barrier')
    comment: Latest CarePort avoidable-day barrier on the stay
measures:
  - name: Discharges
    expr: COUNT(1)
    synonyms: [volume, cases]
  - name: Inpatient Discharges
    expr: COUNT(1) FILTER (WHERE source.patient_class = 'Inpatient')
  - name: ALOS
    expr: AVG(source.los_days) FILTER (WHERE source.patient_class = 'Inpatient')
    comment: Average inpatient length of stay in days
    synonyms: [average length of stay, LOS]
    format: {NUM2}
  - name: Expected ALOS
    expr: AVG(source.expected_los) FILTER (WHERE source.patient_class = 'Inpatient')
    synonyms: [GMLOS, expected LOS]
    format: {NUM2}
  - name: Patient Days
    expr: SUM(source.los_days) FILTER (WHERE source.patient_class = 'Inpatient')
    format: {NUM1}
  - name: Expected Days
    expr: SUM(source.expected_los) FILTER (WHERE source.patient_class = 'Inpatient')
    format: {NUM1}
  - name: LOS O/E Index
    expr: MEASURE(`Patient Days`) / MEASURE(`Expected Days`)
    comment: Observed / expected inpatient LOS. Above 1.0 = patients stay longer than expected
    synonyms: [LOS index, O/E, observed to expected]
    format: {NUM2}
  - name: Excess Days
    expr: SUM(source.excess_days)
    comment: Inpatient days beyond the expected LOS
    synonyms: [excess bed days, opportunity days]
    format: {NUM1}
  - name: Avoidable Days
    expr: SUM(source.avoidable_days)
    comment: CarePort avoidable days (non-clinical barriers)
  - name: Avoidable Days per 100 Discharges
    expr: 100.0 * SUM(source.avoidable_days) / COUNT(1)
    format: {NUM1}
  - name: Long Stay Rate
    expr: AVG(source.is_long_stay)
    comment: Share of inpatient stays 2+ days above expected LOS
    synonyms: [long stay percent, outlier rate]
    format: {PCT}
  - name: Discharges Before Noon %
    expr: AVG(source.discharged_before_noon) FILTER (WHERE source.patient_class = 'Inpatient')
    synonyms: [DBN, discharge before noon]
    format: {PCT}
  - name: 30-Day Readmission Rate
    expr: AVG(source.readmit_30d)
    synonyms: [readmissions, readmit rate]
    format: {PCT}
  - name: Observation Avg Hours
    expr: AVG(source.los_days * 24) FILTER (WHERE source.patient_class = 'Observation')
    format: {NUM1}
"""

CAP_YAML = f"""
version: 1.1
comment: LSH bed capacity KPIs (synthetic). One row per unit per day with the midnight census and staffed beds.
source: __GOLD__.fact_unit_census_day
joins:
  - name: dt
    source: __GOLD__.dim_date
    on: source.date_key = dt.date_key{JOINS}
dimensions:
  - name: Date
    expr: dt.calendar_date
    synonyms: [day, census date]
  - name: Month
    expr: dt.month_start_date
    format: {{type: date, date_format: locale_short_month}}{FISCAL}
  - name: Day of Week
    expr: dt.day_of_week
  - name: Is Weekend
    expr: dt.is_weekend{FAC_UNIT}
measures:
  - name: Occupancy %
    expr: SUM(source.midnight_census) / SUM(source.staffed_beds)
    comment: Midnight census / staffed beds over the selected days and units (average occupancy)
    synonyms: [occupancy, bed utilization, utilization]
    format: {PCT}
  - name: Average Daily Census
    expr: SUM(source.midnight_census) / COUNT(DISTINCT source.date_key)
    synonyms: [ADC, census]
    format: {NUM1}
  - name: Patient Days
    expr: SUM(source.midnight_census)
  - name: Staffed Beds
    expr: SUM(source.staffed_beds)
    comment: Staffed beds on the latest day in the selection (snapshot, not summed across days)
    synonyms: [beds, capacity]
    window:
      - order: Date
        range: current
        semiadditive: last
  - name: Latest Midnight Census
    expr: SUM(source.midnight_census)
    comment: Midnight census on the latest day in the selection
    window:
      - order: Date
        range: current
        semiadditive: last
  - name: Admissions
    expr: SUM(source.admissions)
  - name: Discharges
    expr: SUM(source.discharges)
  - name: Discharges Before Noon %
    expr: SUM(source.discharges_before_noon) / SUM(source.discharges)
    synonyms: [DBN]
    format: {PCT}
  - name: Unit Days Above 95%
    expr: COUNT_IF(source.occupancy_rate > 0.95)
    comment: Unit-days where a unit was more than 95% full at midnight
  - name: Share of Unit Days Above 95%
    expr: AVG(CASE WHEN source.occupancy_rate > 0.95 THEN 1.0 ELSE 0.0 END)
    format: {PCT}
"""

BAR_YAML = f"""
version: 1.1
comment: CarePort avoidable-day barriers by the date the barrier was opened (synthetic).
source: __GOLD__.fact_barrier
joins:
  - name: dt
    source: __GOLD__.dim_date
    on: source.opened_date_key = dt.date_key{JOINS}
dimensions:
  - name: Opened Month
    expr: dt.month_start_date
    format: {{type: date, date_format: locale_short_month}}{FISCAL}{FAC_UNIT}
  - name: Barrier Category
    expr: source.barrier_category
    synonyms: [barrier, delay reason, avoidable day reason]
  - name: Status
    expr: source.status
    comment: Open or Resolved
measures:
  - name: Barriers
    expr: COUNT(1)
  - name: Open Barriers
    expr: COUNT(1) FILTER (WHERE source.status = 'Open')
  - name: Avoidable Days
    expr: SUM(source.avoidable_days)
  - name: Avg Avoidable Days per Barrier
    expr: AVG(source.avoidable_days)
    format: {NUM1}
"""
METRIC_VIEWS = {"los_metrics": LOS_YAML, "capacity_metrics": CAP_YAML, "barrier_metrics": BAR_YAML}
for name, yaml in METRIC_VIEWS.items():
    spark.sql(f"CREATE OR REPLACE VIEW {GOLD}.{name} WITH METRICS LANGUAGE YAML AS $$\n{yaml.replace('__GOLD__', GOLD_NAME)}\n$$")
print("Created metric views:", ", ".join(f"{GOLD_NAME}.{n}" for n in METRIC_VIEWS))
```

**Output:**

```text
Created metric views: schauhan_workspace_catalog.lsh_capacity_gold.los_metrics, schauhan_workspace_catalog.lsh_capacity_gold.capacity_metrics, schauhan_workspace_catalog.lsh_capacity_gold.barrier_metrics
```


---
### Cell 10: Step 8 - Check the metric views against plain SQL

```python
mv = spark.sql(f"""
  WITH a AS (SELECT `Facility Code` AS fac, MEASURE(`LOS O/E Index`) AS oe, MEASURE(`Excess Days`) AS excess,
                    MEASURE(`Discharges Before Noon %`) AS dbn
             FROM {GOLD}.los_metrics WHERE `Fiscal Year` = 'FY2026' GROUP BY ALL),
       b AS (SELECT f.facility_code AS fac, SUM(los_days) FILTER (WHERE patient_class = 'Inpatient') / SUM(expected_los) FILTER (WHERE patient_class = 'Inpatient') AS oe,
                    SUM(excess_days) AS excess, AVG(discharged_before_noon) FILTER (WHERE patient_class = 'Inpatient') AS dbn
             FROM {GOLD}.fact_encounter x JOIN {GOLD}.dim_facility f USING (facility_key) JOIN {GOLD}.dim_date d ON d.date_key = x.discharge_date_key
             WHERE d.fiscal_year = 'FY2026' GROUP BY 1),
       c AS (SELECT `Facility Code` AS fac, MEASURE(`Occupancy %`) AS occ FROM {GOLD}.capacity_metrics WHERE `Fiscal Year` = 'FY2026' GROUP BY ALL),
       e AS (SELECT f.facility_code AS fac, SUM(midnight_census) / SUM(staffed_beds) AS occ
             FROM {GOLD}.fact_unit_census_day x JOIN {GOLD}.dim_facility f USING (facility_key) JOIN {GOLD}.dim_date d USING (date_key)
             WHERE d.fiscal_year = 'FY2026' GROUP BY 1)
  SELECT a.fac, round(a.oe, 3) AS los_oe, round(a.excess) AS excess_days, round(a.dbn, 3) AS before_noon, round(c.occ, 3) AS occupancy,
         abs(a.oe - b.oe) < 1e-9 AND abs(a.excess - b.excess) < 1e-6 AND abs(a.dbn - b.dbn) < 1e-9 AND abs(c.occ - e.occ) < 1e-9 AS matches
  FROM a JOIN b USING (fac) JOIN c USING (fac) JOIN e USING (fac) ORDER BY los_oe DESC""")
display(mv)
rows = mv.collect()
assert len(rows) == 8 and all(r.matches for r in rows), "A metric view doesn't match plain SQL"
print("Metric views match plain SQL for all 8 hospitals (FY2026).")
```

**Output (table):**

| fac | los_oe | excess_days | before_noon | occupancy | matches |
| --- | --- | --- | --- | --- | --- |
| WCH | 1.339 | 8396.0 | 0.168 | 0.853 | True |
| LMH | 1.264 | 42369.0 | 0.17 | 0.884 | True |
| PCH | 1.237 | 18732.0 | 0.269 | 0.881 | True |
| CFH | 1.225 | 3253.0 | 0.209 | 0.722 | True |
| FVH | 1.219 | 6677.0 | 0.182 | 0.836 | True |
| ORH | 1.219 | 4895.0 | 0.188 | 0.819 | True |
| RBH | 1.218 | 6036.0 | 0.178 | 0.818 | True |
| BSH | 1.1 | 5683.0 | 0.288 | 0.791 | True |

**Output:**

```text
Metric views match plain SQL for all 8 hospitals (FY2026).
```


---
### Cell 11: Step 9 - Lineage recorded by Unity Catalog

```python
# Unity Catalog captures lineage automatically (it can take a few minutes to appear after a run).
# Catalog Explorer > current_inpatients > Lineage shows the same graph visually.
try:
    lineage = spark.sql(f"""
      SELECT DISTINCT source_table_full_name AS upstream, target_table_full_name AS downstream
      FROM system.access.table_lineage
      WHERE event_date >= current_date() - INTERVAL 2 DAYS
        AND target_table_catalog = '{CATALOG}' AND target_table_schema IN ('{PREFIX}_bronze', '{PREFIX}_silver', '{PREFIX}_gold')
        AND source_table_full_name IS NOT NULL
      ORDER BY downstream, upstream""")
    display(lineage)
    if lineage.isEmpty():
        print("No lineage rows yet: the system table can lag by up to a few hours after a run. Catalog Explorer > Lineage shows it sooner.")
except Exception as e:
    print(f"Lineage system table not available to you ({type(e).__name__}). Use Catalog Explorer > Lineage instead.")
```

**Output (table):**

| upstream | downstream |
| --- | --- |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_date | schauhan_workspace_catalog.lsh_capacity_gold.barrier_metrics |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_facility | schauhan_workspace_catalog.lsh_capacity_gold.barrier_metrics |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_unit | schauhan_workspace_catalog.lsh_capacity_gold.barrier_metrics |
| schauhan_workspace_catalog.lsh_capacity_gold.fact_barrier | schauhan_workspace_catalog.lsh_capacity_gold.barrier_metrics |
| schauhan_workspace_catalog.lsh_capacity_gold.current_inpatients | schauhan_workspace_catalog.lsh_capacity_gold.bed_board |
| schauhan_workspace_catalog.lsh_capacity_gold.huddle_notes | schauhan_workspace_catalog.lsh_capacity_gold.bed_board |
| schauhan_workspace_catalog.lsh_capacity_gold.inpatient_predictions | schauhan_workspace_catalog.lsh_capacity_gold.bed_board |
| schauhan_workspace_catalog.lsh_capacity_gold.bed_board | schauhan_workspace_catalog.lsh_capacity_gold.bed_board_deid |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_date | schauhan_workspace_catalog.lsh_capacity_gold.capacity_metrics |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_facility | schauhan_workspace_catalog.lsh_capacity_gold.capacity_metrics |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_unit | schauhan_workspace_catalog.lsh_capacity_gold.capacity_metrics |
| schauhan_workspace_catalog.lsh_capacity_gold.fact_unit_census_day | schauhan_workspace_catalog.lsh_capacity_gold.capacity_metrics |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_drg | schauhan_workspace_catalog.lsh_capacity_gold.current_inpatients |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_facility | schauhan_workspace_catalog.lsh_capacity_gold.current_inpatients |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_patient | schauhan_workspace_catalog.lsh_capacity_gold.current_inpatients |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_unit | schauhan_workspace_catalog.lsh_capacity_gold.current_inpatients |
| schauhan_workspace_catalog.lsh_capacity_gold.fact_encounter | schauhan_workspace_catalog.lsh_capacity_gold.current_inpatients |
| schauhan_workspace_catalog.lsh_capacity_silver.adt_stays | schauhan_workspace_catalog.lsh_capacity_gold.current_inpatients |
| schauhan_workspace_catalog.lsh_capacity_silver.barriers | schauhan_workspace_catalog.lsh_capacity_gold.current_inpatients |
| schauhan_workspace_catalog.lsh_capacity_silver.ref_drg | schauhan_workspace_catalog.lsh_capacity_gold.dim_drg |
| schauhan_workspace_catalog.lsh_capacity_silver.ref_facility | schauhan_workspace_catalog.lsh_capacity_gold.dim_facility |
| schauhan_workspace_catalog.lsh_capacity_silver.encounters | schauhan_workspace_catalog.lsh_capacity_gold.dim_patient |
| schauhan_workspace_catalog.lsh_capacity_silver.bed_capacity | schauhan_workspace_catalog.lsh_capacity_gold.dim_unit |
| schauhan_workspace_catalog.lsh_capacity_gold.fact_encounter | schauhan_workspace_catalog.lsh_capacity_gold.fact_barrier |
| schauhan_workspace_catalog.lsh_capacity_silver.barriers | schauhan_workspace_catalog.lsh_capacity_gold.fact_barrier |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_drg | schauhan_workspace_catalog.lsh_capacity_gold.fact_encounter |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_facility | schauhan_workspace_catalog.lsh_capacity_gold.fact_encounter |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_patient | schauhan_workspace_catalog.lsh_capacity_gold.fact_encounter |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_unit | schauhan_workspace_catalog.lsh_capacity_gold.fact_encounter |
| schauhan_workspace_catalog.lsh_capacity_silver.barriers | schauhan_workspace_catalog.lsh_capacity_gold.fact_encounter |
| schauhan_workspace_catalog.lsh_capacity_silver.encounters | schauhan_workspace_catalog.lsh_capacity_gold.fact_encounter |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_facility | schauhan_workspace_catalog.lsh_capacity_gold.fact_unit_census_day |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_unit | schauhan_workspace_catalog.lsh_capacity_gold.fact_unit_census_day |
| schauhan_workspace_catalog.lsh_capacity_silver.adt_stays | schauhan_workspace_catalog.lsh_capacity_gold.fact_unit_census_day |
| schauhan_workspace_catalog.lsh_capacity_silver.bed_capacity | schauhan_workspace_catalog.lsh_capacity_gold.fact_unit_census_day |
| schauhan_workspace_catalog.lsh_capacity_gold.current_inpatients | schauhan_workspace_catalog.lsh_capacity_gold.huddle_notes |
| schauhan_workspace_catalog.lsh_capacity_gold.inpatient_predictions | schauhan_workspace_catalog.lsh_capacity_gold.huddle_notes |
| schauhan_workspace_catalog.lsh_capacity_gold.current_inpatients | schauhan_workspace_catalog.lsh_capacity_gold.inpatient_predictions |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_date | schauhan_workspace_catalog.lsh_capacity_gold.los_metrics |
| schauhan_workspace_catalog.lsh_capacity_gold.dim_drg | schauhan_workspace_catalog.lsh_capacity_gold.los_metrics |

_20 more rows not shown (60 rows in the saved output)._
