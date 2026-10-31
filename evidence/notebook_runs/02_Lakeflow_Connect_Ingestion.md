# 02 - Lakeflow Connect Ingestion

Exported from the Databricks workspace on 2026-10-31 06:17 UTC with each cell's **saved output** from its last run. Synthetic data; organization and hospital names anonymized.


---
### Cell 1

```python

```

_No saved output._


---
### Cell 2: Parameters

```python
dbutils.widgets.text("catalog", "schauhan_workspace_catalog", "1. Catalog")
dbutils.widgets.text("prefix", "lsh_capacity", "2. Schema prefix")
dbutils.widgets.dropdown("full_refresh", "no", ["no", "yes"], "3. Full refresh (re-ingest all files)")
```

_No saved output._


---
### Cell 3: Step 1 - Check inputs, create the bronze schema

```python
import re
from pyspark.sql import functions as F

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("prefix").strip()
FULL_REFRESH = dbutils.widgets.get("full_refresh") == "yes"
for label, val in [("catalog", CATALOG), ("prefix", PREFIX)]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", val or ""):
        raise ValueError(f"Enter a {label} with only letters, numbers and underscores.")

BRONZE = f"`{CATALOG}`.`{PREFIX}_bronze`"
LANDING = f"/Volumes/{CATALOG}/{PREFIX}_raw/landing"
STATE = f"/Volumes/{CATALOG}/{PREFIX}_bronze/ingest_state"
try:
    assert dbutils.fs.ls(LANDING)
except Exception:
    raise ValueError(f"No raw files in {LANDING}. Run notebook 01 with the same catalog and prefix first.")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {BRONZE} COMMENT 'Bronze: raw LSH capacity / LOS extracts ingested by Lakeflow Connect, as received. Synthetic, no PHI.'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {BRONZE}.ingest_state COMMENT 'Auto Loader schema and checkpoint state'")

SOURCES = [   # (bronze table, landing folder, format, what it is)
    ("epic_adt_events", "epic_adt", "json", "Epic ADT messages: admit, transfer, discharge, obs-to-inpatient"),
    ("epic_encounters", "epic_encounters", "csv", "Epic Clarity encounter / DRG extract, with nightly snapshots"),
    ("careport_barriers", "careport_barriers", "json", "CarePort avoidable-day barriers"),
    ("bed_capacity", "bed_capacity", "csv", "Staffed beds by unit, effective-dated"),
    ("mdm_facilities", "mdm_facilities", "csv", "Facility master data"),
    ("mdm_drg", "mdm_drg", "csv", "MS-DRG reference with geometric mean LOS"),
    ("mdm_code_xwalk", "mdm_code_xwalk", "csv", "Source spelling -> canonical value crosswalk"),
]
print(f"Landing: {LANDING}\nBronze: {CATALOG}.{PREFIX}_bronze")
```

**Output:**

```text
Landing: /Volumes/schauhan_workspace_catalog/lsh_capacity_raw/landing
Bronze: schauhan_workspace_catalog.lsh_capacity_bronze
```


---
### Cell 4: Step 1b - Full refresh only: reset bronze (does nothing when Full refresh = no)

```python
if FULL_REFRESH:
    for table, *_ in SOURCES:
        spark.sql(f"DROP TABLE IF EXISTS {BRONZE}.{table}")
    try:
        dbutils.fs.rm(STATE, True)
    except Exception:
        pass
    print("Full refresh: bronze tables and ingestion state reset")
else:
    print("Incremental run: only new landing files will be ingested")
```

_No saved output._


---
### Cell 5: Step 2 - Lakeflow Connect: Auto Loader into bronze

```python
def start_ingest(table, folder, fmt, comment):
    reader = (spark.readStream.format("cloudFiles")
              .option("cloudFiles.format", fmt)
              .option("cloudFiles.schemaLocation", f"{STATE}/{table}/schema")
              .option("cloudFiles.inferColumnTypes", "false")        # land as text; silver owns typing
              .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
              .option("pathGlobFilter", f"*.{fmt}"))
    if fmt == "csv":
        reader = reader.option("header", "true")
    df = (reader.load(f"{LANDING}/{folder}")
          .select("*", F.col("_metadata.file_path").alias("_source_file"),
                  F.col("_metadata.file_modification_time").alias("_source_file_ts"),
                  F.current_timestamp().alias("_ingested_at")))
    return (df.writeStream
            .option("checkpointLocation", f"{STATE}/{table}/checkpoint")
            .option("mergeSchema", "true")
            .trigger(availableNow=True)                          # process all new files, then stop
            .toTable(f"{BRONZE}.{table}"))

def row_count(t):
    return spark.table(f"{BRONZE}.{t}").count() if spark.catalog.tableExists(f"{CATALOG}.{PREFIX}_bronze.{t}") else 0

before = {t: row_count(t) for t, *_ in SOURCES}
queries = {t: start_ingest(t, folder, fmt, c) for t, folder, fmt, c in SOURCES}   # all sources in parallel
for query in queries.values():
    query.awaitTermination()
results = [(t, row_count(t) - before[t], row_count(t)) for t in queries]
for t, _, _, c in SOURCES:
    spark.sql(f"COMMENT ON TABLE {BRONZE}.{t} IS 'Bronze (as received): {c}. Loaded by Lakeflow Connect / Auto Loader.'")
display(spark.createDataFrame(results, "bronze_table string, rows_ingested_this_run long, total_rows long"))
```

**Output (table):**

| bronze_table | rows_ingested_this_run | total_rows |
| --- | --- | --- |
| epic_adt_events | 0 | 478644 |
| epic_encounters | 0 | 219731 |
| careport_barriers | 0 | 21451 |
| bed_capacity | 0 | 55 |
| mdm_facilities | 0 | 8 |
| mdm_drg | 0 | 29 |
| mdm_code_xwalk | 0 | 78 |


---
### Cell 6: Step 3 - Bronze audit: rows, files and rescued data vs the landing volume

```python
# Independent check: read the landing files directly and compare with what Auto Loader wrote.
audit = []
for table, folder, fmt, _ in SOURCES:
    reader = spark.read.option("header", True) if fmt == "csv" else spark.read
    file_rows = getattr(reader, fmt)(f"{LANDING}/{folder}/*.{fmt}").count()
    b = spark.table(f"{BRONZE}.{table}").agg(F.count("*").alias("n"), F.countDistinct("_source_file").alias("files"),
                                            F.count("_rescued_data").alias("rescued")).first()
    audit.append((table, file_rows, b.n, b.files, b.rescued, file_rows == b.n))
display(spark.createDataFrame(audit, "bronze_table string, landing_rows long, bronze_rows long, files long, rescued_rows long, matches boolean"))
bad = [a[0] for a in audit if not a[5]]
assert not bad, f"Bronze doesn't match the landing files for {bad}. Re-run with Full refresh = yes."
print(f"All {len(audit)} bronze tables match the landing files.")
display(spark.table(f"{BRONZE}.epic_adt_events").select("control_id", "message_type", "event_ts", "location", "_source_file").limit(5))
```

**Output (table):**

| bronze_table | landing_rows | bronze_rows | files | rescued_rows | matches |
| --- | --- | --- | --- | --- | --- |
| epic_adt_events | 478644 | 478644 | 20 | 0 | True |
| epic_encounters | 219731 | 219731 | 28 | 0 | True |
| careport_barriers | 21451 | 21451 | 2 | 0 | True |
| bed_capacity | 55 | 55 | 1 | 0 | True |
| mdm_facilities | 8 | 8 | 1 | 0 | True |
| mdm_drg | 29 | 29 | 1 | 0 | True |
| mdm_code_xwalk | 78 | 78 | 1 | 0 | True |

**Output:**

```text
All 7 bronze tables match the landing files.
```

**Output (table):**

| control_id | message_type | event_ts | location | _source_file |
| --- | --- | --- | --- | --- |
| 9032904017-A01 | ADT^A01 | 2025-06-26T16:25:00-05:00 | {"facility":"LMH","unit":"LMH-8E","bed":"8E-31"} | /Volumes/schauhan_workspace_catalog/lsh_capacity_raw/landing/epic_adt/part-00003-tid-7370249922406145399-89ba4351-36e4-4812-9a7c-60131e836421-575-1-c000.json |
| 9032803001-A01 | ADT^A01 | 20250625075400 | {"facility":"LMH","unit":"nmh nsicu","bed":"NSICU-04"} | /Volumes/schauhan_workspace_catalog/lsh_capacity_raw/landing/epic_adt/part-00003-tid-7370249922406145399-89ba4351-36e4-4812-9a7c-60131e836421-575-1-c000.json |
| 9018302003-A01 | ADT^A01 | 2025-01-31T07:05:00-06:00 | {"facility":"LMH","unit":"CVICU","bed":"CVICU-22"} | /Volumes/schauhan_workspace_catalog/lsh_capacity_raw/landing/epic_adt/part-00003-tid-7370249922406145399-89ba4351-36e4-4812-9a7c-60131e836421-575-1-c000.json |
| 9018201006-A01 | ADT^A01 | 2025-01-30T11:45:00-06:00 | {"facility":"LMH","unit":"SICU","bed":"SICU-08"} | /Volumes/schauhan_workspace_catalog/lsh_capacity_raw/landing/epic_adt/part-00003-tid-7370249922406145399-89ba4351-36e4-4812-9a7c-60131e836421-575-1-c000.json |
| 9017803006-A01 | ADT^A01 | 20250126112800 | {"facility":"LMH","unit":"NSICU","bed":"NSICU-20"} | /Volumes/schauhan_workspace_catalog/lsh_capacity_raw/landing/epic_adt/part-00003-tid-7370249922406145399-89ba4351-36e4-4812-9a7c-60131e836421-575-1-c000.json |
