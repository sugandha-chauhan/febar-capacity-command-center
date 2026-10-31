# 06 - Lakebase Operational Serving

Exported from the Databricks workspace on 2026-10-31 06:17 UTC with each cell's **saved output** from its last run. Synthetic data; organization and hospital names anonymized.


---
### Cell 1: Overview (markdown)

# 06 - Lakebase Operational Serving

The app needs **millisecond reads** of the bed board and a place to **write back** what case managers do. Analytics tables in the lakehouse aren't built for that, so this notebook serves the operational data from **Lakebase** (serverless Postgres):

| Object | Where | Direction | Purpose |
| --- | --- | --- | --- |
| `unit_status` | gold Delta table, created here | Delta | One row per unit: beds, patients now, high-risk patients, expected discharges and admissions today, forecast open beds tonight |
| `bed_board`, `unit_status` | Lakebase `<prefix>_serving` schema | **Synced** from gold (reverse ETL, triggered) | What the app reads |
| `capacity_ops.discharge_plan` | Lakebase | **Written by the app** | Case manager confirms a target discharge time for a patient |
| `capacity_ops.discharge_actions` | Lakebase | **Written by the app** | Actions logged at the huddle (escalate barrier, SNF referral sent, family meeting, ...) |

**Steps:** build `unit_status`; get or create the Lakebase project; sync the two gold tables; create the write-back tables; run a test read and write the way the app will; check row counts match the lakehouse.

**App access:** when the app exists (notebook 08), enter its service principal's application ID in the widget and re-run Step 6 to give it read access to the synced tables and write access to `capacity_ops`.

**How to run (about 10 minutes the first time, while the sync pipelines start):** run 05 first, then **Run all**. Re-running refreshes the synced tables from the latest gold data.

---
### Cell 2: Install the Postgres driver

```python
%pip install -q "psycopg[binary]>=3.1"
dbutils.library.restartPython()
```

**Output:**

```text
Note: you may need to restart the kernel using %restart_python or dbutils.library.restartPython() to use updated packages.
```


---
### Cell 3: Parameters

```python
dbutils.widgets.text("catalog", "schauhan_workspace_catalog", "1. Catalog")
dbutils.widgets.text("prefix", "lsh_capacity", "2. Schema prefix")
dbutils.widgets.text("lakebase_project", "lsh-capacity", "3. Lakebase project ID")
dbutils.widgets.text("app_sp_client_id", "", "4. App service principal application ID (optional)")
```

_No saved output._


---
### Cell 4: Step 1 - Setup and the unit_status table

```python
import re, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, AlreadyExists, ResourceAlreadyExists

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("prefix").strip()
PROJECT_ID = dbutils.widgets.get("lakebase_project").strip()
APP_SP = dbutils.widgets.get("app_sp_client_id").strip()
for label, val in [("catalog", CATALOG), ("prefix", PREFIX)]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", val or ""):
        raise ValueError(f"Enter a {label} with only letters, numbers and underscores.")
if not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", PROJECT_ID):
    raise ValueError("Lakebase project ID: lowercase letters, numbers and hyphens, starting with a letter.")
spark.conf.set("spark.sql.session.timeZone", "UTC")
GOLD, GOLD_NAME = f"`{CATALOG}`.`{PREFIX}_gold`", f"{CATALOG}.{PREFIX}_gold"
SERVING = f"{PREFIX}_serving"                   # UC schema for synced tables = Postgres schema name
PG_DB = "databricks_postgres"
if not spark.catalog.tableExists(f"{GOLD_NAME}.bed_board"):
    raise ValueError("bed_board not found. Run notebook 05 first.")
w = WorkspaceClient()
ME = w.current_user.me().user_name

# One row per unit at the snapshot. Expected admissions = average for the same weekday over the last 8 weeks.
spark.sql(f"""
  CREATE OR REPLACE TABLE {GOLD}.unit_status
  COMMENT "Unit capacity at the snapshot: beds, patients, high-risk patients, expected discharges and admissions, forecast open beds tonight"
  TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
  WITH snap AS (SELECT max(snapshot_ts) AS s FROM {GOLD}.bed_board),
       pts AS (SELECT unit_code, count(*) AS patients_now, count_if(risk_tier = 'High') AS high_risk_patients,
                      count_if(days_past_expected > 0) AS past_expected_discharge, count_if(open_barrier_category IS NOT NULL) AS open_barriers,
                      sum(p_discharge_today) AS expected_discharges_today, count_if(likely_discharge_today) AS likely_discharges_today
               FROM {GOLD}.bed_board GROUP BY unit_code),
       adm AS (SELECT u.unit_code, avg(c.admissions) AS expected_admissions_today
               FROM {GOLD}.fact_unit_census_day c JOIN {GOLD}.dim_unit u USING (unit_key) JOIN {GOLD}.dim_date d USING (date_key) CROSS JOIN snap
               WHERE d.calendar_date >= date_sub(to_date(snap.s), 56) AND dayofweek(d.calendar_date) = dayofweek(snap.s)
               GROUP BY u.unit_code)
  SELECT u.unit_code, u.facility_code, f.facility_name, u.unit_name, u.unit_type, u.current_staffed_beds AS staffed_beds,
         coalesce(p.patients_now, 0) AS patients_now, u.current_staffed_beds - coalesce(p.patients_now, 0) AS open_beds_now,
         round(coalesce(p.patients_now, 0) / u.current_staffed_beds, 3) AS occupancy_now,
         coalesce(p.high_risk_patients, 0) AS high_risk_patients, coalesce(p.past_expected_discharge, 0) AS past_expected_discharge,
         coalesce(p.open_barriers, 0) AS open_barriers, round(coalesce(p.expected_discharges_today, 0), 1) AS expected_discharges_today,
         coalesce(p.likely_discharges_today, 0) AS likely_discharges_today, round(coalesce(a.expected_admissions_today, 0), 1) AS expected_admissions_today,
         round(u.current_staffed_beds - (coalesce(p.patients_now, 0) + coalesce(a.expected_admissions_today, 0)
               - coalesce(p.expected_discharges_today, 0)), 1) AS forecast_open_beds_tonight,
         snap.s AS snapshot_ts
  FROM {GOLD}.dim_unit u JOIN {GOLD}.dim_facility f USING (facility_code)
  LEFT JOIN pts p USING (unit_code) LEFT JOIN adm a USING (unit_code) CROSS JOIN snap""")
spark.sql(f"ALTER TABLE {GOLD}.unit_status ALTER COLUMN unit_code SET NOT NULL")
spark.sql(f"ALTER TABLE {GOLD}.unit_status ADD CONSTRAINT unit_status_pk PRIMARY KEY (unit_code)")
display(spark.sql(f"""SELECT facility_code, sum(staffed_beds) AS beds, sum(patients_now) AS patients, sum(open_beds_now) AS open_now,
                        sum(high_risk_patients) AS high_risk, round(sum(expected_discharges_today)) AS exp_discharges,
                        round(sum(expected_admissions_today)) AS exp_admissions, round(sum(forecast_open_beds_tonight)) AS open_tonight
                      FROM {GOLD}.unit_status GROUP BY ALL ORDER BY beds DESC"""))
```

**Output (table):**

| facility_code | beds | patients | open_now | high_risk | exp_discharges | exp_admissions | open_tonight |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LMH | 486 | 459 | 27 | 60 | 98.0 | 97.0 | 29.0 |
| PCH | 232 | 211 | 21 | 17 | 51.0 | 56.0 | 15.0 |
| BSH | 100 | 85 | 15 | 0 | 28.0 | 27.0 | 17.0 |
| WCH | 94 | 85 | 9 | 6 | 23.0 | 23.0 | 10.0 |
| FVH | 92 | 69 | 23 | 4 | 17.0 | 19.0 | 21.0 |
| RBH | 84 | 66 | 18 | 3 | 17.0 | 19.0 | 16.0 |
| ORH | 70 | 52 | 18 | 2 | 16.0 | 14.0 | 20.0 |
| CFH | 52 | 37 | 15 | 3 | 12.0 | 12.0 | 16.0 |


---
### Cell 5: Step 2 - Get or create the Lakebase project

```python
from databricks.sdk.service.postgres import Project, ProjectSpec, EndpointType

try:
    project = w.postgres.get_project(name=f"projects/{PROJECT_ID}")
    print("Using existing Lakebase project:", project.name)
except NotFound:
    project = w.postgres.create_project(
        project=Project(spec=ProjectSpec(display_name="LSH Capacity Command Center", pg_version=17)),
        project_id=PROJECT_ID).wait()
    print("Created Lakebase project:", project.name)

branches = list(w.postgres.list_branches(parent=project.name))
branch = next((b for b in branches if b.name.endswith("/production")), branches[0])
endpoints = list(w.postgres.list_endpoints(parent=branch.name))
endpoint = next((e for e in endpoints if e.status and e.status.endpoint_type == EndpointType.ENDPOINT_TYPE_READ_WRITE), endpoints[0])
BRANCH, EP_NAME, HOST = branch.name, endpoint.name, endpoint.status.hosts.host
print(f"Branch: {BRANCH}\nEndpoint: {EP_NAME}\nHost: {HOST}\nDatabase: {PG_DB}")
```

**Output:**

```text
Using existing Lakebase project: projects/lsh-capacity
Branch: projects/lsh-capacity/branches/production
Endpoint: projects/lsh-capacity/branches/production/endpoints/primary
Host: ep-round-grass-d1za0378.database.us-west-2.cloud.databricks.com
Database: databricks_postgres
```


---
### Cell 6: Step 3 - Sync bed_board and unit_status to Lakebase (reverse ETL)

```python
from databricks.sdk.service.postgres import (SyncedTable, SyncedTableSyncedTableSpec,
                                             SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy as Policy)
from databricks.sdk.errors import ResourceConflict

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SERVING}` COMMENT 'Lakebase synced tables for the LSH Capacity Command Center app'")
SYNCS = {"bed_board": ["encounter_id"], "unit_status": ["unit_code"]}

refreshes = {}   # table -> (pipeline_id, update_id) for syncs that already existed
for table, pk in SYNCS.items():
    target = f"{CATALOG}.{SERVING}.{table}"
    try:
        st = w.postgres.get_synced_table(name=f"synced_tables/{target}")
        # full refresh: the gold tables are recreated upstream (CREATE OR REPLACE), which an incremental sync can't follow
        try:
            upd = w.pipelines.start_update(pipeline_id=st.status.pipeline_id, full_refresh=True)
        except ResourceConflict:   # an earlier incremental update is still retrying: stop it, then full refresh
            w.pipelines.stop_and_wait(pipeline_id=st.status.pipeline_id)
            upd = w.pipelines.start_update(pipeline_id=st.status.pipeline_id, full_refresh=True)
        refreshes[table] = (st.status.pipeline_id, upd.update_id)
        print(f"{target}: exists, refresh started")
    except NotFound:
        w.postgres.create_synced_table(
            synced_table=SyncedTable(spec=SyncedTableSyncedTableSpec(
                source_table_full_name=f"{GOLD_NAME}.{table}", branch=BRANCH, postgres_database=PG_DB,
                primary_key_columns=pk, scheduling_policy=Policy.TRIGGERED, create_database_objects_if_missing=True)),
            synced_table_id=target)
        print(f"{target}: created (initial snapshot running)")

# Wait: new syncs until they are online; existing syncs until the refresh just started has completed
deadline = time.time() + 20 * 60
pending = set(SYNCS)
while pending and time.time() < deadline:
    for table in list(pending):
        s = w.postgres.get_synced_table(name=f"synced_tables/{CATALOG}.{SERVING}.{table}").status
        state = s.detailed_state.value if s and s.detailed_state else "UNKNOWN"
        if "FAILED" in state:
            raise RuntimeError(f"{table} sync failed: {state} {s.message}")
        if table in refreshes:
            pid, uid = refreshes[table]
            u = w.pipelines.get_update(pipeline_id=pid, update_id=uid).update.state.value
            if u in ("FAILED", "CANCELED"):
                raise RuntimeError(f"{table} refresh {u}")
            done = u == "COMPLETED"
        else:
            done = state in ("SYNCED_TABLE_ONLINE", "SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE")
        if done:
            print(f"{table}: {state}")
            pending.discard(table)
    if pending:
        time.sleep(20)
if pending:
    raise TimeoutError(f"Still syncing after 20 minutes: {pending}. Re-run this cell to keep waiting.")
```

**Output:**

```text
schauhan_workspace_catalog.lsh_capacity_serving.bed_board: exists, refresh started
schauhan_workspace_catalog.lsh_capacity_serving.unit_status: exists, refresh started
unit_status: SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE
bed_board: SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE
```


---
### Cell 7: Step 4 - Connect and create the write-back tables

```python
import psycopg

def pg_conn():
    """Short-lived OAuth credential for the Lakebase endpoint (valid 1 hour). Never printed."""
    token = w.postgres.generate_database_credential(endpoint=EP_NAME).token
    return psycopg.connect(host=HOST, dbname=PG_DB, user=ME, password=token, sslmode="require", connect_timeout=30)

DDL = [
    "CREATE SCHEMA IF NOT EXISTS capacity_ops",
    """CREATE TABLE IF NOT EXISTS capacity_ops.discharge_plan (
         encounter_id        BIGINT PRIMARY KEY,
         target_discharge_ts TIMESTAMPTZ,
         plan_status         TEXT NOT NULL DEFAULT 'Planned' CHECK (plan_status IN ('Planned', 'Confirmed', 'Delayed', 'Discharged')),
         delay_reason        TEXT,
         updated_by          TEXT NOT NULL,
         updated_at          TIMESTAMPTZ NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS capacity_ops.discharge_actions (
         action_id    BIGSERIAL PRIMARY KEY,
         encounter_id BIGINT NOT NULL,
         action_type  TEXT NOT NULL,
         note         TEXT,
         owner_role   TEXT,
         status       TEXT NOT NULL DEFAULT 'Open' CHECK (status IN ('Open', 'Done')),
         created_by   TEXT NOT NULL,
         created_at   TIMESTAMPTZ NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS discharge_actions_encounter ON capacity_ops.discharge_actions (encounter_id)",
]
with pg_conn() as conn, conn.cursor() as cur:
    for stmt in DDL:
        cur.execute(stmt)
    cur.execute("""SELECT table_schema, table_name FROM information_schema.tables
                   WHERE table_schema IN (%s, 'capacity_ops') ORDER BY 1, 2""", (SERVING,))
    for row in cur.fetchall():
        print("  %s.%s" % row)
```

**Output:**

```text
capacity_ops.discharge_actions
  capacity_ops.discharge_plan
  lsh_capacity_serving.bed_board
  lsh_capacity_serving.unit_status
```


---
### Cell 8: Step 5 - Test read and write the way the app will

```python
import statistics

with pg_conn() as conn, conn.cursor() as cur:
    # 1. The case manager's worklist: highest-risk patients at LMH
    cur.execute(f"""SELECT encounter_id, unit_code, bed, round(long_stay_risk::numeric, 2), risk_tier,
                           to_char(predicted_discharge_ts, 'Mon DD HH24:MI'), coalesce(open_barrier_category, '-')
                    FROM {SERVING}.bed_board WHERE facility_code = 'LMH' AND risk_tier = 'High'
                    ORDER BY long_stay_risk DESC LIMIT 5""")
    worklist = cur.fetchall()
    for r in worklist:
        print(r)
    top = worklist[0][0]

    # 2. Point-lookup latency (what the app does when a patient row is opened)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        cur.execute(f"SELECT * FROM {SERVING}.bed_board WHERE encounter_id = %s", (top,))
        cur.fetchone()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"Point lookup: median {statistics.median(times):.1f} ms over 20 reads")

    # 3. Write back: confirm a discharge plan and log a huddle action, in one transaction
    cur.execute("""INSERT INTO capacity_ops.discharge_plan (encounter_id, target_discharge_ts, plan_status, updated_by)
                   VALUES (%s, date_trunc('day', now()) + interval '14 hours', 'Confirmed', %s)
                   ON CONFLICT (encounter_id) DO UPDATE SET target_discharge_ts = EXCLUDED.target_discharge_ts,
                     plan_status = EXCLUDED.plan_status, updated_by = EXCLUDED.updated_by, updated_at = now()""",
                (top, f"{ME} (notebook 06 test)"))
    cur.execute("""INSERT INTO capacity_ops.discharge_actions (encounter_id, action_type, note, owner_role, created_by)
                   VALUES (%s, 'Family meeting', 'Test entry from notebook 06', 'Case manager', %s) RETURNING action_id""",
                (top, f"{ME} (notebook 06 test)"))
    action_id = cur.fetchone()[0]
    conn.commit()

    # 4. Read it back the way the app joins plan + actions onto the synced board
    cur.execute(f"""SELECT b.encounter_id, b.unit_code, p.plan_status, p.target_discharge_ts, count(a.action_id) AS actions
                    FROM {SERVING}.bed_board b LEFT JOIN capacity_ops.discharge_plan p USING (encounter_id)
                    LEFT JOIN capacity_ops.discharge_actions a USING (encounter_id)
                    WHERE b.encounter_id = %s GROUP BY 1, 2, 3, 4""", (top,))
    print("Read back:", cur.fetchone(), "| new action_id", action_id)
```

**Output:**

```text
(9081406001, 'LMH-14W', '14W-11', Decimal('0.75'), 'High', 'Nov 12 11:19', '-')
(9080906003, 'LMH-14W', '14W-05', Decimal('0.71'), 'High', 'Nov 08 16:27', '-')
(9081606001, 'LMH-14W', '14W-19', Decimal('0.71'), 'High', 'Nov 13 20:26', '-')
(9080807002, 'LMH-15W', '15W-05', Decimal('0.69'), 'High', 'Nov 06 06:06', '-')
(9080911002, 'LMH-10W', '10W-01', Decimal('0.68'), 'High', 'Nov 02 14:35', '-')
Point lookup: median 2.1 ms over 20 reads
Read back: (9081406001, 'LMH-14W', 'Confirmed', datetime.datetime(2026, 10, 31, 14, 0, tzinfo=zoneinfo.ZoneInfo(key='GMT')), 3) | new action_id 3
```


---
### Cell 9: Step 6 - Give the app's service principal access (optional)

```python
from databricks.sdk.service.postgres import Role, RoleRoleSpec, RoleIdentityType
from psycopg import sql

if not APP_SP:
    print("No app service principal entered: skipped. After creating the app (notebook 08), enter its application ID and re-run this cell.")
else:
    try:
        w.postgres.create_role(parent=BRANCH, role=Role(spec=RoleRoleSpec(postgres_role=APP_SP, identity_type=RoleIdentityType.SERVICE_PRINCIPAL)),
                               role_id="app-" + re.sub(r"[^a-z0-9]", "", APP_SP.lower())[:40]).wait()
        print("Created Postgres role for the app service principal")
    except (AlreadyExists, ResourceAlreadyExists):
        print("Postgres role for the app service principal already exists")
    with pg_conn() as conn, conn.cursor() as cur:
        role = sql.Identifier(APP_SP)
        for stmt in ["GRANT USAGE ON SCHEMA {s} TO {r}", "GRANT SELECT ON ALL TABLES IN SCHEMA {s} TO {r}"]:
            cur.execute(sql.SQL(stmt).format(s=sql.Identifier(SERVING), r=role))
        for stmt in ["GRANT USAGE ON SCHEMA capacity_ops TO {r}",
                     "GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA capacity_ops TO {r}",
                     "GRANT USAGE ON ALL SEQUENCES IN SCHEMA capacity_ops TO {r}"]:
            cur.execute(sql.SQL(stmt).format(r=role))
    print(f"App service principal can read {SERVING}.* and write capacity_ops.*")
```

**Output:**

```text
Created Postgres role for the app service principal
App service principal can read lsh_capacity_serving.* and write capacity_ops.*
```


---
### Cell 10: Step 7 - Check Lakebase matches the lakehouse, and show the app connection values

```python
checks = []
with pg_conn() as conn, conn.cursor() as cur:
    for table, key in [("bed_board", "encounter_id"), ("unit_status", "unit_code")]:
        cur.execute(f"SELECT count(*), count(DISTINCT {key}) FROM {SERVING}.{table}")
        pg_rows, pg_keys = cur.fetchone()
        uc_rows = spark.table(f"{GOLD_NAME}.{table}").count()
        checks.append((table, uc_rows, pg_rows, pg_rows == uc_rows == pg_keys))
    cur.execute(f"SELECT sum(patients_now), round(sum(expected_discharges_today)::numeric, 1) FROM {SERVING}.unit_status")
    pg_pat, pg_dis = cur.fetchone()
    cur.execute(f"SELECT encounter_id, long_stay_risk FROM {SERVING}.bed_board ORDER BY long_stay_risk DESC NULLS LAST, encounter_id LIMIT 1")
    pg_top = cur.fetchone()
uc = spark.sql(f"SELECT sum(patients_now) AS p, round(sum(expected_discharges_today), 1) AS d FROM {GOLD}.unit_status").first()
uc_top = spark.sql(f"SELECT encounter_id, long_stay_risk FROM {GOLD}.bed_board ORDER BY long_stay_risk DESC NULLS LAST, encounter_id LIMIT 1").first()
checks.append(("unit_status totals (patients, expected discharges)", f"{uc.p}, {uc.d}", f"{pg_pat}, {pg_dis}", int(pg_pat) == uc.p and float(pg_dis) == float(uc.d)))
checks.append(("top-risk patient and score", f"{uc_top.encounter_id}, {uc_top.long_stay_risk}", f"{pg_top[0]}, {pg_top[1]}",
               pg_top[0] == uc_top.encounter_id and abs(pg_top[1] - uc_top.long_stay_risk) < 1e-9))
display(spark.createDataFrame([(c[0], str(c[1]), str(c[2]), c[3]) for c in checks], "check string, lakehouse string, lakebase string, matches boolean"))
assert all(c[3] for c in checks), "Lakebase doesn't match the lakehouse"

print("App connection values (no secrets; the app gets OAuth credentials at runtime):")
for k, v in [("Lakebase project", project.name), ("Branch", BRANCH), ("Endpoint", EP_NAME), ("Host", HOST), ("Database", PG_DB),
             ("Read schema (synced)", SERVING), ("Write schema", "capacity_ops")]:
    print(f"  {k}: {v}")
```

**Output (table):**

| check | lakehouse | lakebase | matches |
| --- | --- | --- | --- |
| bed_board | 1064 | 1064 | True |
| unit_status | 52 | 52 | True |
| unit_status totals (patients, expected discharges) | 1064, 261.7 | 1064, 261.7 | True |
| top-risk patient and score | 9081406001, 0.7502 | 9081406001, 0.7502 | True |

**Output:**

```text
App connection values (no secrets; the app gets OAuth credentials at runtime):
  Lakebase project: projects/lsh-capacity
  Branch: projects/lsh-capacity/branches/production
  Endpoint: projects/lsh-capacity/branches/production/endpoints/primary
  Host: ep-round-grass-d1za0378.database.us-west-2.cloud.databricks.com
  Database: databricks_postgres
  Read schema (synced): lsh_capacity_serving
  Write schema: capacity_ops
```
