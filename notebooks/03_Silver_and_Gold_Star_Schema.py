# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 03 - Silver and Gold Star Schema
# MAGIC
# MAGIC Turns the bronze extracts from notebook 02 into **trusted, conformed silver tables** and a **gold star schema** for capacity and length of stay.
# MAGIC
# MAGIC **Silver (`<prefix>_silver`)**
# MAGIC * `encounters`: parses dates and flags, conforms facility / DRG / financial class / admit source / patient class / disposition spellings through the master-data crosswalk, and keeps the latest extract of each encounter (in-house rows are replaced once the patient is discharged).
# MAGIC * `adt_events`: parses the two timestamp formats, normalizes unit codes, and drops re-sent duplicate messages. `adt_stays`: one row per encounter built from the ADT events, the census source of truth.
# MAGIC * `barriers`, `bed_capacity` and the reference tables.
# MAGIC * A **data quality** check: every key must map, and the ADT feed and the encounter extract must agree.
# MAGIC
# MAGIC **Gold (`<prefix>_gold`)**
# MAGIC
# MAGIC | Table | Grain | Used by |
# MAGIC | --- | --- | --- |
# MAGIC | `dim_date` | day, LSH fiscal year (starts Sep 1) | all |
# MAGIC | `dim_facility`, `dim_unit`, `dim_drg`, `dim_patient` | one row per facility / unit / DRG / patient (MRN) | all; `dim_patient` holds the PHI-like columns masked in notebook 04 |
# MAGIC | `fact_encounter` | one row per encounter: LOS, expected LOS, excess and avoidable days, long-stay flag, before-noon discharge, 30-day readmission | LOS metric view, ML training (05), Genie |
# MAGIC | `fact_unit_census_day` | unit x day: midnight census, staffed beds, occupancy, admissions, discharges | capacity metric view, app, Genie |
# MAGIC | `fact_barrier` | one row per CarePort barrier | LOS metric view, app |
# MAGIC | `current_inpatients` | everyone in the hospital at the 7:00 AM snapshot | ML scoring (05), bed board in Lakebase (06), app |
# MAGIC
# MAGIC **How to run (about 3 minutes):** run notebook 02 first, then **Parameters**, then **Run all**. Silver and gold are rebuilt from bronze on every run.

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "schauhan_workspace_catalog", "1. Catalog")
dbutils.widgets.text("prefix", "lsh_capacity", "2. Schema prefix")

# COMMAND ----------

# DBTITLE 1,Step 1 - Setup and SQL helpers
import re
from pyspark.sql import functions as F

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("prefix").strip()
for label, val in [("catalog", CATALOG), ("prefix", PREFIX)]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", val or ""):
        raise ValueError(f"Enter a {label} with only letters, numbers and underscores.")
spark.conf.set("spark.sql.session.timeZone", "UTC")      # timestamps are hospital local wall-clock time

q = lambda s: f"`{CATALOG}`.`{PREFIX}_{s}`"
BRONZE, SILVER, GOLD = q("bronze"), q("silver"), q("gold")
if not spark.catalog.tableExists(f"{CATALOG}.{PREFIX}_bronze.epic_adt_events"):
    raise ValueError("Bronze tables not found. Run notebook 02 with the same catalog and prefix first.")
for schema, comment in [(SILVER, "Silver: typed, conformed, de-duplicated LSH capacity and LOS data"),
                        (GOLD, "Gold: LSH capacity and LOS star schema, metric views, model scores")]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema} COMMENT '{comment}. Synthetic, no PHI.'")

def key(c):
    """Match key for messy source spellings: upper case, letters / digits / '+' only."""
    return f"upper(regexp_replace({c}, '[^A-Za-z0-9+]', ''))"

def match(domain, alias):
    return f"(SELECT * FROM {SILVER}.ref_code_match WHERE domain = '{domain}') {alias}"

def ctas(schema, name, comment, sql):
    spark.sql(f"CREATE OR REPLACE TABLE {schema}.{name} COMMENT \"{comment}\" AS {sql}")
print(f"Silver: {CATALOG}.{PREFIX}_silver   Gold: {CATALOG}.{PREFIX}_gold")

# COMMAND ----------

# DBTITLE 1,Step 2 - Silver: reference data and bed capacity
ctas(SILVER, "ref_code_match", "Source spelling match key -> canonical value, by domain", f"""
  SELECT DISTINCT DOMAIN AS domain, {key('SOURCE_VALUE')} AS match_key, trim(CANONICAL_VALUE) AS canonical_value,
         CAST(SORT_ORDER AS INT) AS sort_order
  FROM {BRONZE}.mdm_code_xwalk""")
ambiguous = spark.sql(f"SELECT domain, match_key FROM {SILVER}.ref_code_match GROUP BY ALL HAVING count(*) > 1").collect()
assert not ambiguous, f"Ambiguous crosswalk keys: {ambiguous}"

ctas(SILVER, "ref_facility", "Active LSH facilities", f"""
  SELECT upper(trim(FACILITY_CODE)) AS facility_code, trim(FACILITY_NAME) AS facility_name, trim(CITY) AS city, trim(TIER) AS tier
  FROM {BRONZE}.mdm_facilities WHERE upper(ACTIVE) = 'Y'""")
ctas(SILVER, "ref_facility_match", "Facility code or name match key -> facility_code", f"""
  SELECT DISTINCT match_key, facility_code FROM (
    SELECT {key('facility_code')} AS match_key, facility_code FROM {SILVER}.ref_facility
    UNION ALL SELECT {key('facility_name')}, facility_code FROM {SILVER}.ref_facility)""")
ctas(SILVER, "ref_drg", "MS-DRG reference with geometric mean LOS and service line", f"""
  SELECT lpad(trim(MS_DRG), 3, '0') AS drg, trim(DRG_DESCRIPTION) AS drg_description, CAST(GMLOS AS DOUBLE) AS gmlos,
         trim(SERVICE_LINE) AS service_line, CAST(SURGICAL_FLAG AS INT) AS is_surgical
  FROM {BRONZE}.mdm_drg""")
ctas(SILVER, "bed_capacity", "Staffed beds by unit, effective-dated (effective_to is exclusive)", f"""
  SELECT upper(trim(FACILITY)) AS facility_code, concat(upper(trim(FACILITY)), '-', upper(trim(UNIT_CODE))) AS unit_code,
         upper(trim(UNIT_CODE)) AS unit, trim(UNIT_NAME) AS unit_name, trim(UNIT_TYPE) AS unit_type,
         CAST(STAFFED_BEDS AS INT) AS staffed_beds, CAST(LICENSED_BEDS AS INT) AS licensed_beds,
         to_date(EFFECTIVE_FROM) AS effective_from, coalesce(to_date(nullif(EFFECTIVE_TO, '')), DATE'9999-12-31') AS effective_to
  FROM {BRONZE}.bed_capacity""")
overlaps = spark.sql(f"""SELECT a.unit_code FROM {SILVER}.bed_capacity a JOIN {SILVER}.bed_capacity b
    ON a.unit_code = b.unit_code AND a.effective_from < b.effective_from AND b.effective_from < a.effective_to""").count()
assert overlaps == 0, "Bed capacity periods overlap"
display(spark.sql(f"SELECT facility_code, count(DISTINCT unit_code) AS units, sum(staffed_beds) FILTER (WHERE effective_to = DATE'9999-12-31') AS staffed_beds_now FROM {SILVER}.bed_capacity GROUP BY ALL ORDER BY 1"))

# COMMAND ----------

# DBTITLE 1,Step 3 - Silver: encounters, ADT events and stays, barriers
# ---- Encounters: conform spellings, keep the latest extract of each encounter ----
ctas(SILVER, "encounters", "Encounters from the Epic extract: typed, conformed, latest extract per encounter", f"""
  WITH p AS (
    SELECT CAST(e.PAT_ENC_CSN_ID AS BIGINT) AS csn, trim(e.PAT_MRN_ID) AS mrn,
           initcap(trim(split(e.PAT_NAME, ',')[0])) AS last_name, trim(split(e.PAT_NAME, ',')[1]) AS first_name,
           to_date(e.BIRTH_DATE, 'MM/dd/yyyy') AS birth_date, upper(trim(e.SEX)) AS sex,
           f.facility_code, concat(f.facility_code, '-', upper(trim(e.UNIT))) AS unit_code, nullif(trim(e.ROOM_BED), '') AS room_bed,
           to_timestamp(e.ADMIT_DTTM, 'MM/dd/yyyy HH:mm') AS admit_ts, to_timestamp(nullif(e.DISCH_DTTM, ''), 'MM/dd/yyyy HH:mm') AS discharge_ts,
           src.canonical_value AS admit_source, cls.canonical_value AS patient_class, trim(e.HOSP_SERVICE) AS service_line,
           lpad(regexp_extract(e.MSDRG, '([0-9]+)', 1), 3, '0') AS drg, fin.canonical_value AS fin_class,
           trim(e.ATTENDING_PROV) AS attending,
           CASE WHEN upper(trim(e.COMPLEX_DC_FLAG)) IN ('Y', '1') THEN 1 ELSE 0 END AS complex_discharge,
           CASE WHEN upper(trim(e.LIVES_ALONE)) IN ('Y', '1') THEN 1 ELSE 0 END AS lives_alone,
           trim(e.ADMITTED_FROM) AS admitted_from, disp.canonical_value AS disposition,
           CAST(regexp_replace(e.EXP_LOS_DAYS, ',', '') AS DOUBLE) AS expected_los,
           to_timestamp(e.EXTRACT_TS) AS extract_ts,
           e.FACILITY AS src_facility, e.MSDRG AS src_drg, e.FIN_CLASS AS src_fin_class, e.DISCH_DISP AS src_disposition, e._source_file
    FROM {BRONZE}.epic_encounters e
    LEFT JOIN {SILVER}.ref_facility_match f ON {key('e.FACILITY')} = f.match_key
    LEFT JOIN {match('admit_source', 'src')} ON {key('e.ADMIT_SOURCE')} = src.match_key
    LEFT JOIN {match('patient_class', 'cls')} ON {key('e.PATIENT_CLASS')} = cls.match_key
    LEFT JOIN {match('fin_class', 'fin')} ON {key('e.FIN_CLASS')} = fin.match_key
    LEFT JOIN {match('disposition', 'disp')} ON {key('e.DISCH_DISP')} = disp.match_key)
  SELECT * FROM p QUALIFY row_number() OVER (PARTITION BY csn ORDER BY extract_ts DESC, discharge_ts DESC NULLS LAST) = 1""")

# ---- ADT: two timestamp formats, three unit code styles, re-sent duplicates ----
ctas(SILVER, "adt_events", "Epic ADT events: parsed timestamps, normalized unit codes, duplicates removed", f"""
  WITH p AS (
    SELECT control_id, substring_index(message_type, '^', -1) AS event_type,
           CASE WHEN event_ts LIKE '____-__-__T%' THEN to_timestamp(substr(event_ts, 1, 19), "yyyy-MM-dd'T'HH:mm:ss")
                ELSE to_timestamp(event_ts, 'yyyyMMddHHmmss') END AS event_ts,
           CAST(patient:csn::string AS BIGINT) AS csn, patient:mrn::string AS mrn,
           upper(trim(location:facility::string)) AS facility_code, upper(trim(location:unit::string)) AS raw_unit,
           location:bed::string AS bed, patient_class AS raw_class, event_ts AS src_event_ts, _source_file, _ingested_at
    FROM {BRONZE}.epic_adt_events)
  SELECT control_id, event_type, event_ts, csn, mrn, facility_code,
         concat(facility_code, '-', regexp_replace(raw_unit, concat('^', facility_code, '[- ]'), '')) AS unit_code, bed,
         CASE raw_class WHEN 'I' THEN 'Inpatient' WHEN 'O' THEN 'Observation' END AS patient_class,
         src_event_ts, _source_file, _ingested_at
  FROM p QUALIFY row_number() OVER (PARTITION BY control_id ORDER BY _ingested_at, _source_file) = 1""")

ctas(SILVER, "adt_stays", "One row per encounter from ADT: admit, discharge, unit, current bed (census source of truth)", f"""
  SELECT csn, max(mrn) AS mrn, max(facility_code) AS facility_code,
         max_by(unit_code, event_ts) FILTER (WHERE event_type IN ('A01', 'A02')) AS unit_code,
         max_by(bed, event_ts) FILTER (WHERE event_type IN ('A01', 'A02')) AS bed,
         min(event_ts) FILTER (WHERE event_type = 'A01') AS admit_ts,
         max(event_ts) FILTER (WHERE event_type = 'A03') AS discharge_ts,
         CASE WHEN count_if(event_type = 'A06') > 0 THEN 'Inpatient'
              ELSE max_by(patient_class, event_ts) FILTER (WHERE event_type = 'A01') END AS patient_class,
         count(*) AS events
  FROM {SILVER}.adt_events GROUP BY csn""")

# ---- CarePort barriers ----
ctas(SILVER, "barriers", "CarePort avoidable-day barriers, conformed categories, latest update per barrier", f"""
  SELECT b.barrier_id, CAST(b.csn AS BIGINT) AS csn, c.canonical_value AS barrier_category,
         to_date(b.opened_dt) AS opened_date, to_date(b.resolved_dt) AS resolved_date,
         CAST(b.avoidable_days AS INT) AS avoidable_days, b.status, to_timestamp(b.updated_at) AS updated_at,
         b.category AS src_category
  FROM {BRONZE}.careport_barriers b
  LEFT JOIN {match('barrier_category', 'c')} ON {key('b.category')} = c.match_key
  QUALIFY row_number() OVER (PARTITION BY b.barrier_id ORDER BY to_timestamp(b.updated_at) DESC) = 1""")

snap = spark.sql(f"SELECT date_trunc('HOUR', max(event_ts)) + INTERVAL 1 HOUR AS ts FROM {SILVER}.adt_events").first().ts
print(f"Data snapshot time (from the ADT feed): {snap}")

# COMMAND ----------

# DBTITLE 1,Step 4 - Silver data quality: keys map, and the ADT feed agrees with the encounter extract
units = f"(SELECT DISTINCT unit_code FROM {SILVER}.bed_capacity)"
dq = spark.sql(f"""
  SELECT 'encounters: unmapped facility / unit / DRG' AS check,
         count_if(facility_code IS NULL OR unit_code NOT IN {units} OR drg NOT IN (SELECT drg FROM {SILVER}.ref_drg)) AS failures
  FROM {SILVER}.encounters
  UNION ALL SELECT 'encounters: unmapped admit source / class / financial class',
         count_if(admit_source IS NULL OR patient_class IS NULL OR fin_class IS NULL) FROM {SILVER}.encounters
  UNION ALL SELECT 'encounters: discharged without a mapped disposition',
         count_if(discharge_ts IS NOT NULL AND disposition IS NULL) FROM {SILVER}.encounters
  UNION ALL SELECT 'ADT: unknown unit or unparsed timestamp',
         count_if(unit_code NOT IN {units} OR event_ts IS NULL) FROM {SILVER}.adt_events
  UNION ALL SELECT 'barriers: unmapped category or unknown encounter',
         count_if(barrier_category IS NULL OR csn NOT IN (SELECT csn FROM {SILVER}.encounters)) FROM {SILVER}.barriers
  UNION ALL SELECT 'ADT vs extract: encounter in one source only',
         (SELECT count(*) FROM {SILVER}.adt_stays a FULL JOIN {SILVER}.encounters e USING (csn) WHERE a.csn IS NULL OR e.csn IS NULL)
  UNION ALL SELECT 'ADT vs extract: admit / discharge time (to the minute) or unit differs',
         (SELECT count(*) FROM {SILVER}.adt_stays a JOIN {SILVER}.encounters e USING (csn)
          WHERE date_trunc('MINUTE', a.admit_ts) <> e.admit_ts OR a.unit_code <> e.unit_code
             OR NOT (date_trunc('MINUTE', a.discharge_ts) <=> e.discharge_ts))   -- the extract keeps minutes only
""")
display(dq)
failed = [r.check for r in dq.collect() if r.failures]
assert not failed, f"Data quality failed: {failed}"

raw = {t: spark.table(f"{BRONZE}.{t}").count() for t in ["epic_encounters", "epic_adt_events", "careport_barriers"]}
slv = {t: spark.table(f"{SILVER}.{t}").count() for t in ["encounters", "adt_events", "barriers"]}
print(f"Encounters: {raw['epic_encounters']:,} extract rows -> {slv['encounters']:,} encounters (older snapshots replaced)")
print(f"ADT: {raw['epic_adt_events']:,} messages -> {slv['adt_events']:,} events (re-sent duplicates removed)")
print(f"Barriers: {raw['careport_barriers']:,} -> {slv['barriers']:,}")
print("Silver data quality passed: every key maps, and the ADT feed and the encounter extract agree on every stay.")

# COMMAND ----------

# DBTITLE 1,Step 5 - Gold: dimensions
SNAP = str(snap)                    # data snapshot time, e.g. 2026-10-30 07:00:00

# LSH fiscal year starts Sep 1: FY2026 = Sep 2025 - Aug 2026, fiscal month 1 = Sep, Q1 = Sep-Nov
ctas(GOLD, "dim_date", "Calendar and LSH fiscal calendar (FY starts Sep 1), one row per day", """
  SELECT CAST(date_format(d, 'yyyyMMdd') AS INT) AS date_key, d AS calendar_date, trunc(d, 'MM') AS month_start_date,
         year(d) AS calendar_year, month(d) AS calendar_month, date_format(d, 'MMM yyyy') AS month_label,
         date_format(d, 'EEEE') AS day_of_week, dayofweek(d) IN (1, 7) AS is_weekend,
         IF(month(d) >= 9, year(d) + 1, year(d)) AS fiscal_year_num, concat('FY', IF(month(d) >= 9, year(d) + 1, year(d))) AS fiscal_year,
         (month(d) + 3) % 12 + 1 AS fiscal_month, concat('Q', CAST(ceil(((month(d) + 3) % 12 + 1) / 3) AS INT)) AS fiscal_quarter
  FROM (SELECT explode(sequence(DATE'2024-08-01', DATE'2027-12-31', INTERVAL 1 DAY)) AS d)""")

ctas(GOLD, "dim_facility", "LSH hospitals", f"""
  SELECT CAST(row_number() OVER (ORDER BY tier <> 'Academic Medical Center', facility_code) AS INT) AS facility_key,
         facility_code, facility_name, city, tier FROM {SILVER}.ref_facility""")

ctas(GOLD, "dim_unit", "Inpatient units with type and current staffed beds", f"""
  SELECT CAST(row_number() OVER (ORDER BY facility_code, unit_code) AS INT) AS unit_key, unit_code, facility_code, unit,
         unit_name, unit_type, unit_type LIKE '%ICU%' AS is_icu, staffed_beds AS current_staffed_beds
  FROM {SILVER}.bed_capacity WHERE effective_to = DATE'9999-12-31'""")

ctas(GOLD, "dim_drg", "MS-DRGs with geometric mean LOS and service line", f"""
  SELECT CAST(drg AS INT) AS drg_key, drg, drg_description, gmlos, service_line, is_surgical FROM {SILVER}.ref_drg""")

# Patient identity lives only here (and in current_inpatients). Notebook 04 masks these columns.
ctas(GOLD, "dim_patient", "Patients (synthetic). Holds the PHI-like identity columns that notebook 04 masks.", f"""
  SELECT CAST(substr(mrn, 3) AS BIGINT) AS patient_key, mrn, first_name, last_name,
         concat(first_name, ' ', last_name) AS patient_name, birth_date, sex
  FROM {SILVER}.encounters QUALIFY row_number() OVER (PARTITION BY mrn ORDER BY admit_ts DESC) = 1""")

for d, k in [("dim_date", "date_key"), ("dim_facility", "facility_key"), ("dim_unit", "unit_key"), ("dim_drg", "drg_key"), ("dim_patient", "patient_key")]:
    r = spark.sql(f"SELECT count(*) AS n, count(DISTINCT {k}) AS k FROM {GOLD}.{d}").first()
    assert r.n == r.k, f"{d}.{k} is not unique"
    print(f"{d}: {r.n:,} rows, unique {k}")

# COMMAND ----------

# DBTITLE 1,Step 6 - Gold: fact_encounter, fact_barrier, fact_unit_census_day
LOS = "round((unix_timestamp(e.discharge_ts) - unix_timestamp(e.admit_ts)) / 86400, 3)"
ctas(GOLD, "fact_encounter", "One row per encounter: LOS vs expected, excess and avoidable days, long stay, readmission", f"""
  WITH b AS (SELECT csn, sum(avoidable_days) AS avoidable_days, count(*) AS barrier_count,
                    max_by(barrier_category, opened_date) AS barrier_category
             FROM {SILVER}.barriers GROUP BY csn),
       r AS (SELECT csn, lead(admit_ts) OVER (PARTITION BY mrn ORDER BY admit_ts) AS next_admit_ts FROM {SILVER}.encounters)
  SELECT e.csn AS encounter_id, p.patient_key, f.facility_key, u.unit_key, d.drg_key,
         CAST(date_format(e.admit_ts, 'yyyyMMdd') AS INT) AS admit_date_key,
         CAST(date_format(e.discharge_ts, 'yyyyMMdd') AS INT) AS discharge_date_key,
         e.admit_ts, e.discharge_ts, e.patient_class, e.admit_source, e.fin_class, e.admitted_from, e.disposition, e.attending,
         e.complex_discharge, e.lives_alone, CAST(floor(months_between(e.admit_ts, e.birth_date) / 12) AS INT) AS age_at_admit,
         e.expected_los, {LOS} AS los_days,
         CASE WHEN e.patient_class = 'Inpatient' AND e.discharge_ts IS NOT NULL THEN round(greatest(0, {LOS} - e.expected_los), 3) END AS excess_days,
         CASE WHEN e.patient_class = 'Inpatient' AND e.discharge_ts IS NOT NULL THEN CAST({LOS} - e.expected_los >= 2 AS INT) END AS is_long_stay,
         coalesce(b.avoidable_days, 0) AS avoidable_days, coalesce(b.barrier_count, 0) AS barrier_count, b.barrier_category,
         CASE WHEN e.discharge_ts IS NOT NULL THEN CAST(hour(e.discharge_ts) < 12 AS INT) END AS discharged_before_noon,
         CASE WHEN e.patient_class = 'Inpatient' AND e.discharge_ts IS NOT NULL AND e.disposition <> 'Expired'
              THEN coalesce(CAST(r.next_admit_ts <= e.discharge_ts + INTERVAL 30 DAYS AS INT), 0) END AS readmit_30d,
         e.discharge_ts IS NULL AS is_in_house
  FROM {SILVER}.encounters e
  JOIN {GOLD}.dim_patient p USING (mrn) JOIN {GOLD}.dim_facility f USING (facility_code)
  JOIN {GOLD}.dim_unit u USING (unit_code) JOIN {GOLD}.dim_drg d USING (drg)
  LEFT JOIN b USING (csn) LEFT JOIN r USING (csn)""")

ctas(GOLD, "fact_barrier", "One row per CarePort avoidable-day barrier", f"""
  SELECT b.barrier_id, b.csn AS encounter_id, x.facility_key, x.unit_key, x.drg_key,
         CAST(date_format(b.opened_date, 'yyyyMMdd') AS INT) AS opened_date_key,
         CAST(date_format(b.resolved_date, 'yyyyMMdd') AS INT) AS resolved_date_key,
         b.barrier_category, b.avoidable_days, b.status
  FROM {SILVER}.barriers b JOIN {GOLD}.fact_encounter x ON b.csn = x.encounter_id""")

# Midnight census from the ADT stays: a patient counts on day d if admitted on or before d and still in at midnight.
ctas(GOLD, "fact_unit_census_day", "Unit x day: midnight census, staffed beds, occupancy, admissions, discharges", f"""
  WITH days AS (SELECT explode(sequence(DATE'2024-09-01', date_sub(to_date(TIMESTAMP'{SNAP}'), 1))) AS d),
       s AS (SELECT unit_code, admit_ts, discharge_ts, to_date(admit_ts) AS a, to_date(coalesce(discharge_ts, TIMESTAMP'{SNAP}')) AS z
             FROM {SILVER}.adt_stays),
       mid AS (SELECT unit_code, d, count(*) AS midnight_census FROM s
               LATERAL VIEW explode(sequence(a, date_sub(z, 1))) t AS d WHERE z > a GROUP BY ALL),
       adm AS (SELECT unit_code, to_date(admit_ts) AS d, count(*) AS admissions FROM s GROUP BY ALL),
       dis AS (SELECT unit_code, to_date(discharge_ts) AS d, count(*) AS discharges,
                      count_if(hour(discharge_ts) < 12) AS discharges_before_noon FROM s WHERE discharge_ts IS NOT NULL GROUP BY ALL),
       grid AS (SELECT c.unit_code, days.d, c.staffed_beds FROM days JOIN {SILVER}.bed_capacity c
                ON days.d >= c.effective_from AND days.d < c.effective_to)
  SELECT CAST(date_format(g.d, 'yyyyMMdd') AS INT) AS date_key, u.unit_key, u.facility_key, g.staffed_beds,
         coalesce(m.midnight_census, 0) AS midnight_census,
         round(coalesce(m.midnight_census, 0) / g.staffed_beds, 4) AS occupancy_rate,
         coalesce(adm.admissions, 0) AS admissions, coalesce(dis.discharges, 0) AS discharges,
         coalesce(dis.discharges_before_noon, 0) AS discharges_before_noon
  FROM grid g
  JOIN (SELECT du.unit_key, du.unit_code, df.facility_key FROM {GOLD}.dim_unit du JOIN {GOLD}.dim_facility df USING (facility_code)) u USING (unit_code)
  LEFT JOIN mid m ON m.unit_code = g.unit_code AND m.d = g.d
  LEFT JOIN adm ON adm.unit_code = g.unit_code AND adm.d = g.d
  LEFT JOIN dis ON dis.unit_code = g.unit_code AND dis.d = g.d""")

for t in ["fact_encounter", "fact_barrier", "fact_unit_census_day"]:
    print(f"{t}: {spark.table(f'{GOLD}.{t}').count():,} rows")

# COMMAND ----------

# DBTITLE 1,Step 7 - Gold: current inpatients at the 7:00 AM snapshot
ctas(GOLD, "current_inpatients", f"Everyone in the hospital at the {SNAP} snapshot, with expected discharge and open barriers", f"""
  WITH ob AS (SELECT csn, max_by(barrier_category, opened_date) AS open_barrier_category,
                     max(avoidable_days) AS open_barrier_days
              FROM {SILVER}.barriers WHERE status = 'Open' GROUP BY csn)
  SELECT x.encounter_id, x.patient_key, p.mrn, p.patient_name, x.age_at_admit AS age, p.sex,
         f.facility_code, f.facility_name, u.unit_code, u.unit_name, u.unit_type, s.bed,
         x.admit_ts, round((unix_timestamp(TIMESTAMP'{SNAP}') - unix_timestamp(x.admit_ts)) / 86400, 2) AS los_so_far_days,
         x.patient_class, x.admit_source, x.fin_class, x.admitted_from, x.complex_discharge, x.lives_alone,
         d.drg, d.drg_description, d.service_line, x.expected_los,
         x.admit_ts + make_interval(0, 0, 0, 0, 0, 0, x.expected_los * 86400) AS expected_discharge_ts,
         round(greatest(0, (unix_timestamp(TIMESTAMP'{SNAP}') - unix_timestamp(x.admit_ts)) / 86400 - x.expected_los), 2) AS days_past_expected,
         ob.open_barrier_category, ob.open_barrier_days, x.attending, TIMESTAMP'{SNAP}' AS snapshot_ts
  FROM {GOLD}.fact_encounter x
  JOIN {GOLD}.dim_patient p USING (patient_key) JOIN {GOLD}.dim_facility f USING (facility_key)
  JOIN {GOLD}.dim_unit u USING (unit_key) JOIN {GOLD}.dim_drg d USING (drg_key)
  JOIN {SILVER}.adt_stays s ON s.csn = x.encounter_id
  LEFT JOIN ob ON ob.csn = x.encounter_id
  WHERE x.is_in_house""")
display(spark.sql(f"""SELECT facility_code, count(*) AS in_house, count_if(days_past_expected > 0) AS past_expected_discharge,
                       count_if(open_barrier_category IS NOT NULL) AS with_open_barrier
                     FROM {GOLD}.current_inpatients GROUP BY ALL ORDER BY in_house DESC"""))

# COMMAND ----------

# DBTITLE 1,Step 8 - Gold checks and the synthetic baseline KPIs
# 1. Independent census check: midnights summed from the census fact must equal midnights counted per stay
chk = spark.sql(f"""
  SELECT (SELECT sum(midnight_census) FROM {GOLD}.fact_unit_census_day) AS census_fact_bed_days,
         (SELECT sum(greatest(0, datediff(least(to_date(coalesce(discharge_ts, TIMESTAMP'{SNAP}')), to_date(TIMESTAMP'{SNAP}')),
                                          greatest(to_date(admit_ts), DATE'2024-09-01'))))
          FROM {GOLD}.fact_encounter) AS per_stay_bed_days""").first()
print(f"Bed days: census fact {chk.census_fact_bed_days:,} vs per-stay count {chk.per_stay_bed_days:,}")
assert chk.census_fact_bed_days == chk.per_stay_bed_days, "Census fact doesn't reconcile to the encounters"

# 2. Every encounter made it into the fact (no dimension join dropped rows), and current inpatients = in-house encounters
n_silver = spark.table(f"{SILVER}.encounters").count()
n_fact = spark.table(f"{GOLD}.fact_encounter").count()
n_cur = spark.table(f"{GOLD}.current_inpatients").count()
n_house = spark.table(f"{GOLD}.fact_encounter").filter("is_in_house").count()
assert n_silver == n_fact and n_cur == n_house, (n_silver, n_fact, n_cur, n_house)
print(f"fact_encounter = silver encounters ({n_fact:,}); current_inpatients = in-house encounters ({n_cur:,})")

# 3. Synthetic baseline for the problem statement KPIs (FY2026 = Sep 2025 - Aug 2026)
display(spark.sql(f"""
  WITH e AS (SELECT x.*, f.facility_code, dd.fiscal_year FROM {GOLD}.fact_encounter x JOIN {GOLD}.dim_facility f USING (facility_key)
             JOIN {GOLD}.dim_date dd ON dd.date_key = x.discharge_date_key WHERE x.patient_class = 'Inpatient'),
       c AS (SELECT f.facility_code, dd.fiscal_year, sum(midnight_census) / sum(staffed_beds) AS occupancy,
                    avg(CAST(occ > 0.95 AS INT)) AS share_days_over_95
             FROM (SELECT facility_key, date_key, sum(midnight_census) AS midnight_census, sum(staffed_beds) AS staffed_beds,
                          sum(midnight_census) / sum(staffed_beds) AS occ FROM {GOLD}.fact_unit_census_day GROUP BY ALL) t
             JOIN {GOLD}.dim_facility f USING (facility_key) JOIN {GOLD}.dim_date dd USING (date_key) GROUP BY ALL)
  SELECT e.facility_code, e.fiscal_year, count(*) AS ip_discharges, round(avg(los_days), 2) AS alos,
         round(sum(los_days) / sum(expected_los), 3) AS los_oe_index, round(sum(excess_days)) AS excess_days,
         sum(avoidable_days) AS avoidable_days, round(avg(discharged_before_noon), 3) AS before_noon_share,
         round(avg(is_long_stay), 3) AS long_stay_rate, round(any_value(c.occupancy), 3) AS occupancy,
         round(any_value(c.share_days_over_95), 3) AS share_days_over_95
  FROM e JOIN c USING (facility_code, fiscal_year) WHERE e.fiscal_year = 'FY2026' GROUP BY ALL ORDER BY ip_discharges DESC"""))