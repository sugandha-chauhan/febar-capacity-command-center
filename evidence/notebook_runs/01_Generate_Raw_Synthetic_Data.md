# 01 - Generate Raw Synthetic Data

Exported from the Databricks workspace on 2026-10-31 06:17 UTC with each cell's **saved output** from its last run. Synthetic data; organization and hospital names anonymized.


---
### Cell 1: Overview (markdown)

# 01 - Generate Raw Synthetic Data

Simulates the source systems behind the LSH Capacity Command Center (see *00 - Problem Statement*) and writes their **raw extracts** to a Unity Catalog volume, as they would arrive from the hospital systems. Everything is **synthetic**: patients, names, MRNs, providers, beds and outcomes are generated. No PHI.

**How to run (about 3 minutes):** run **Parameters**, check the boxes at the top, then **Run all**. If the landing volume already has files, the notebook stops and leaves them alone, unless *Overwrite raw files* = `yes`.

**What lands in** `/Volumes/<catalog>/<prefix>_raw/landing/`

| Folder | Source system | Format | Built-in data quality issues |
| --- | --- | --- | --- |
| `epic_adt` | Epic ADT feed (HL7 A01 admit, A02 transfer, A03 discharge, A06 obs-to-inpatient) | JSON, monthly files + 2 daily files | ISO timestamps with UTC offset vs HL7 `yyyyMMddHHmmss`, unit codes as `11E` / `LMH-11E` / `nmh 11e`, re-sent duplicates, late discharge messages |
| `epic_encounters` | Epic Clarity encounter / DRG extract | CSV, monthly + 2 nightly snapshots | facility code vs name, DRG `064` / `64` / `MS-DRG 064`, financial class / disposition / admit source spelling variants, Y/N vs 1/0 flags, the same encounter re-extracted after discharge |
| `careport_barriers` | CarePort avoidable-day barriers | JSON | category spelling variants, open barriers without a resolved date |
| `bed_capacity` | Bed management: staffed beds by unit, effective-dated | CSV | winter surge beds and a postpartum bed closure |
| `mdm_facilities`, `mdm_drg`, `mdm_code_xwalk` | Master data | CSV | reference data used to conform the extracts |

The notebook also saves an **answer key**, `<prefix>_raw._sim_truth_encounters`: the true simulated encounters, including the real discharge time of patients still in the hospital. The pipeline never reads it. Notebook 05 uses it only to show how well the model predicted stays that hadn't finished yet.

**Stories built into the data**
* LMH runs about 93% midnight occupancy, with Med/Surg and Oncology units often above 95%, and a winter surge in Dec-Feb. Winter surge beds open on LMH 13W from Dec 15, 2025 to Mar 15, 2026.
* LMH Oncology and Neurosciences, and WCH Medicine, run above the expected LOS. BSH runs below it.
* SNF placement barriers and avoidable days rise in FY2026 (from Sep 2025), as post-acute capacity tightens.
* PCH starts a discharge-before-noon program in Mar 2026; its before-noon rate rises from about 20% to about 33%.
* Complex discharges, age 75+, living alone, Medicaid / Medicare Advantage and admission from a SNF all raise the chance of a long stay. These are the signals the model in notebook 05 learns.

---
### Cell 2: Parameters

```python
# Run this cell, then check the input boxes at the top of the notebook. Use the same values in notebooks 01-08.
dbutils.widgets.text("catalog", "schauhan_workspace_catalog", "1. Catalog")
dbutils.widgets.text("prefix", "lsh_capacity", "2. Schema prefix")
dbutils.widgets.dropdown("overwrite_raw", "no", ["no", "yes"], "3. Overwrite raw files")
```

_No saved output._


---
### Cell 3: Step 1 - Check inputs, create the raw schema and landing volume

```python
import re

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("prefix").strip()
OVERWRITE = dbutils.widgets.get("overwrite_raw") == "yes"
for label, val in [("catalog", CATALOG), ("prefix", PREFIX)]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", val or ""):
        raise ValueError(f"Enter a {label} with only letters, numbers and underscores.")

spark.conf.set("spark.sql.session.timeZone", "UTC")   # all timestamps are hospital local wall-clock time
RAW_SCHEMA = f"{PREFIX}_raw"
RAW = f"`{CATALOG}`.`{RAW_SCHEMA}`"
LANDING = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/landing"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {RAW} COMMENT 'Raw landing zone: synthetic LSH capacity and LOS source extracts. No PHI.'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {RAW}.landing COMMENT 'Raw source files, ingested by Lakeflow Connect (Auto Loader) in notebook 02'")

try:
    EXISTING_FILES = dbutils.fs.ls(LANDING)
except Exception:
    EXISTING_FILES = []
if EXISTING_FILES and not OVERWRITE:
    dbutils.notebook.exit(f"Raw files already exist in {LANDING}, so nothing was changed. Go to notebook 02, or set "
                          "'Overwrite raw files' = yes to regenerate them (then run notebook 02 with 'Full refresh' = yes).")
print(f"Landing volume: {LANDING}")
```

**Output:**

```text
Landing volume: /Volumes/schauhan_workspace_catalog/lsh_capacity_raw/landing
```


---
### Cell 4: Step 1b - Overwrite only: clear the landing volume (does nothing when Overwrite = no)

```python
# Only reached when 'Overwrite raw files' = yes (Step 1 stops the notebook otherwise if files exist).
if OVERWRITE and EXISTING_FILES:
    for f in EXISTING_FILES:
        dbutils.fs.rm(f.path, True)
    print("Old raw files removed")
else:
    print("Landing volume is empty, nothing to clear")
```

_No saved output._


---
### Cell 5: Step 2 - Reference data: hospitals, units, DRGs, code spellings

```python
import math
from pyspark.sql import functions as F

AS_OF_TS = "2026-10-30 07:00:00"     # 7:00 AM capacity huddle
AS_OF_DATE = "2026-10-30"
SIM_START = "2024-08-01"             # warm-up month so the census is full from Sep 1, 2024
REPORT_START = "2024-09-01"          # first day of FY2025
FY2026_START = "2025-09-01"

# (code, name, city, tier, base occupancy, LOS multiplier, discharge-before-noon rate)
FACILITIES = [
    ("LMH", "Lakeshore Memorial Hospital", "Lakeshore City, ST", "Academic Medical Center", 0.92, 1.04, 0.17),
    ("PCH", "Prairie Central Hospital", "Prairie Town, ST", "Regional Flagship", 0.89, 1.05, 0.20),
    ("BSH", "Bluffside Hospital", "Bluffside, ST", "Community", 0.80, 0.90, 0.28),
    ("RBH", "Riverbend Hospital", "Riverbend, ST", "Community", 0.78, 1.02, 0.18),
    ("ORH", "Oak Ridge Hospital", "Oak Ridge, ST", "Community", 0.76, 1.01, 0.19),
    ("FVH", "Fox Valley Hospital", "Fox Valley, ST", "Community", 0.80, 1.04, 0.18),
    ("WCH", "Willow Creek Hospital", "Willow Creek, ST", "Community", 0.87, 1.08, 0.16),
    ("CFH", "Cornfield Hospital", "Cornfield, ST", "Community", 0.66, 1.03, 0.20),
]
# (facility, unit, unit name, unit type, staffed beds)
UNITS = [
    ("LMH","MICU","Medical ICU","ICU",34), ("LMH","SICU","Surgical ICU","Surgical ICU",30),
    ("LMH","CVICU","Cardiovascular ICU","CVICU",24), ("LMH","NSICU","Neuro ICU","Neuro ICU",20),
    ("LMH","8E","8 East Telemetry","Telemetry",36), ("LMH","9E","9 East Cardiology","Telemetry",32),
    ("LMH","14W","14 West Oncology","Oncology",36), ("LMH","15W","15 West Oncology","Oncology",36),
    ("LMH","11E","11 East Med/Surg","Med/Surg",36), ("LMH","12E","12 East Med/Surg","Med/Surg",36),
    ("LMH","13W","13 West Med/Surg","Med/Surg",32), ("LMH","10W","10 West Ortho/Neuro","Ortho/Neuro",32),
    ("LMH","16E","16 East Transplant","Transplant",24), ("LMH","9W","9 West Postpartum","Postpartum",30),
    ("LMH","4E","4 East Behavioral Health","Behavioral",24), ("LMH","OBS","Clinical Decision Unit","Observation",24),
    ("PCH","ICU","Intensive Care Unit","ICU",28), ("PCH","4S","4 South Telemetry","Telemetry",32),
    ("PCH","3N","3 North Med/Surg","Med/Surg",36), ("PCH","3S","3 South Med/Surg","Med/Surg",36),
    ("PCH","5N","5 North Oncology","Oncology",28), ("PCH","5S","5 South Ortho/Neuro","Ortho/Neuro",28),
    ("PCH","2N","2 North Postpartum","Postpartum",24), ("PCH","OBS","Observation Unit","Observation",20),
    ("BSH","ICU","Intensive Care Unit","ICU",16), ("BSH","3E","3 East Med/Surg","Med/Surg",32),
    ("BSH","3W","3 West Telemetry","Telemetry",24), ("BSH","PP","Postpartum","Postpartum",16),
    ("BSH","OBS","Observation Unit","Observation",12),
    ("RBH","ICU","Intensive Care Unit","ICU",12), ("RBH","2E","2 East Med/Surg","Med/Surg",32),
    ("RBH","2W","2 West Telemetry","Telemetry",20), ("RBH","PP","Postpartum","Postpartum",14),
    ("RBH","OBS","Observation Unit","Observation",10),
    ("ORH","ICU","Intensive Care Unit","ICU",12), ("ORH","MS","Med/Surg","Med/Surg",30),
    ("ORH","TELE","Telemetry","Telemetry",18), ("ORH","OBS","Observation Unit","Observation",10),
    ("FVH","ICU","Intensive Care Unit","ICU",12), ("FVH","MS","Med/Surg","Med/Surg",30),
    ("FVH","TELE","Telemetry","Telemetry",20), ("FVH","BH","Behavioral Health","Behavioral",20),
    ("FVH","OBS","Observation Unit","Observation",10),
    ("WCH","ICU","Intensive Care Unit","ICU",14), ("WCH","MS","Med/Surg","Med/Surg",34),
    ("WCH","TELE","Telemetry","Telemetry",24), ("WCH","PP","Postpartum","Postpartum",12),
    ("WCH","OBS","Observation Unit","Observation",10),
    ("CFH","ICU","Intensive Care Unit","ICU",8), ("CFH","MS","Med/Surg","Med/Surg",24),
    ("CFH","TELE","Telemetry","Telemetry",12), ("CFH","OBS","Observation Unit","Observation",8),
]
# Bed changes: (facility, unit, staffed beds, effective from, effective to)
BED_CHANGES = [("LMH", "13W", 40, "2025-12-15", "2026-03-15"),   # winter surge beds
               ("RBH", "PP", 10, "2025-07-01", None)]               # postpartum beds closed
# Occupancy offset by unit type, and by calendar month (winter surge, summer dip)
TYPE_OCC = {"ICU": -0.02, "Surgical ICU": -0.03, "CVICU": -0.03, "Neuro ICU": -0.02, "Telemetry": 0.0,
            "Med/Surg": 0.035, "Oncology": 0.04, "Ortho/Neuro": 0.0, "Transplant": -0.05, "Postpartum": -0.15,
            "Behavioral": 0.02, "Observation": -0.12}
SEASON = {12: 0.03, 1: 0.06, 2: 0.04, 3: 0.015, 6: -0.015, 7: -0.03, 8: -0.03}
DOW = {1: 0.82, 2: 1.12, 3: 1.08, 4: 1.04, 5: 1.02, 6: 0.98, 7: 0.84}      # Spark dayofweek: 1 = Sunday

# MS-DRGs: (code, description, geometric mean LOS, service line, surgical)
DRGS = [
    ("005","Liver transplant w MCC",11.5,"Transplant",1), ("652","Kidney transplant",4.0,"Transplant",1),
    ("023","Craniotomy w major device implant or acute complex CNS PDX w MCC",8.0,"Neurosciences",1),
    ("025","Craniotomy and endovascular intracranial procedures w MCC",7.0,"Neurosciences",1),
    ("064","Intracranial hemorrhage or cerebral infarction w MCC",4.3,"Neurosciences",0),
    ("207","Respiratory system diagnosis w ventilator support >96 hours",12.3,"Pulmonary & Critical Care",0),
    ("870","Septicemia or severe sepsis w MV >96 hours",12.0,"Pulmonary & Critical Care",0),
    ("871","Septicemia or severe sepsis w/o MV >96 hours w MCC",4.4,"Medicine",0),
    ("853","Infectious and parasitic diseases w O.R. procedure w MCC",9.2,"Medicine",1),
    ("193","Simple pneumonia and pleurisy w MCC",3.7,"Medicine",0),
    ("190","Chronic obstructive pulmonary disease w MCC",3.2,"Medicine",0),
    ("683","Renal failure w CC",3.3,"Medicine",0), ("603","Cellulitis w/o MCC",3.0,"Medicine",0),
    ("392","Esophagitis, gastroent and misc digestive disorders w/o MCC",2.4,"Digestive Health",0),
    ("329","Major small and large bowel procedures w MCC",9.0,"Digestive Health",1),
    ("330","Major small and large bowel procedures w CC",5.4,"Digestive Health",1),
    ("291","Heart failure and shock w MCC",3.9,"Cardiovascular",0),
    ("247","Percutaneous cardiovascular procedures w drug-eluting stent w/o MCC",1.8,"Cardiovascular",1),
    ("216","Cardiac valve and other major cardiothoracic procedures w cardiac cath w MCC",9.6,"Cardiovascular",1),
    ("470","Major hip and knee joint replacement w/o MCC",2.0,"Orthopedics",1),
    ("460","Spinal fusion except cervical w/o MCC",2.6,"Orthopedics",1),
    ("834","Acute leukemia w/o major O.R. procedure w MCC",10.5,"Oncology",0),
    ("846","Chemotherapy w/o acute leukemia as secondary diagnosis w MCC",4.7,"Oncology",0),
    ("775","Vaginal delivery w/o complicating diagnoses",1.9,"Women's Health",0),
    ("788","Cesarean section w/o sterilization w/o CC/MCC",2.5,"Women's Health",1),
    ("885","Psychoses",5.6,"Behavioral Health",0), ("897","Alcohol, drug abuse or dependence w/o rehab w/o MCC",3.2,"Behavioral Health",0),
    ("312","Syncope and collapse",2.2,"Cardiovascular",0), ("313","Chest pain",1.6,"Cardiovascular",0),
]
GMLOS = {d[0]: d[2] for d in DRGS}
POOLS = {   # DRG mix by unit type: (DRG, weight)
    "ICU": [("207",.22),("870",.18),("871",.25),("853",.10),("291",.15),("064",.10)],
    "Surgical ICU": [("329",.35),("853",.25),("460",.15),("005",.10),("330",.15)],
    "CVICU": [("216",.60),("291",.30),("247",.10)],
    "Neuro ICU": [("023",.35),("025",.35),("064",.30)],
    "Telemetry": [("291",.40),("247",.30),("312",.15),("313",.15)],
    "Med/Surg": [("871",.22),("193",.16),("190",.12),("683",.10),("603",.10),("392",.10),("330",.08),("329",.04),("470",.04),("460",.04)],
    "Oncology": [("846",.50),("834",.25),("871",.15),("329",.10)],
    "Ortho/Neuro": [("470",.45),("460",.30),("064",.25)],
    "Transplant": [("652",.65),("005",.35)],
    "Postpartum": [("775",.68),("788",.32)],
    "Behavioral": [("885",.55),("897",.45)],
    "Observation": [("313",.35),("312",.30),("392",.20),("603",.15)],
}
# Rough mean LOS per unit type, used only to size admissions before calibration in Step 3
ALOS_EST = {t: sum(w * GMLOS[d] for d, w in p) * 1.18 + 0.3 for t, p in POOLS.items()}
ALOS_EST["Observation"] = 1.0

# Raw spellings used by the source systems, canonical value first. Notebook 03 conforms them.
CODE_VARIANTS = {
    "fin_class": {"Medicare": ["MEDICARE", "Medicare FFS"], "Medicare Advantage": ["Mcare Adv", "MA - Medicare Advantage"],
                  "Commercial": ["COMM", "Commercial/Managed Care"], "Medicaid": ["IL Medicaid", "MEDICAID"],
                  "Self-Pay": ["SELF PAY", "Uninsured"], "Other": ["Workers Comp/Other", "OTHER"]},
    "admit_source": {"Emergency Department": ["ED", "EMERGENCY"], "Transfer from Hospital": ["TRANSFER", "Hosp Transfer"],
                     "Direct Admit": ["DIRECT", "Direct Admission"], "Elective / Scheduled": ["ELECTIVE", "Scheduled Surgery"]},
    "disposition": {"Home": ["01 - Home/Self Care", "HOME"], "Home Health": ["06 - Home Health", "Home w/ Home Health"],
                    "Skilled Nursing Facility": ["03 - SNF", "Skilled Nursing"], "Inpatient Rehab": ["62 - Inpatient Rehab", "IRF"],
                    "Hospice": ["50 - Hospice", "HOSPICE"], "Expired": ["20 - Expired", "EXPIRED"],
                    "Left AMA": ["07 - Left AMA", "AMA"], "Transfer to Acute Hospital": ["02 - Short-term Hospital", "Transfer Acute"]},
    "barrier_category": {"SNF placement": ["Post-Acute: SNF bed availability", "SNF Placement"],
                         "Insurance authorization": ["Payer auth pending", "INSURANCE AUTHORIZATION"],
                         "Home health / DME": ["Home Health/DME", "DME delivery"],
                         "Family / guardianship": ["Family/Guardianship decision", "Guardianship"],
                         "Pending test or procedure": ["Pending Test/Procedure", "Awaiting MRI"],
                         "Transportation": ["Transport", "Transportation - ambulette"]},
    "patient_class": {"Inpatient": ["IP", "I"], "Observation": ["OBS", "O"]},
}
FIRST_F = ["Mary","Patricia","Jennifer","Linda","Elizabeth","Susan","Sarah","Karen","Grace","Nina","Rachel","Sofia","Priya","Maria","Aisha","Hannah","Emily","Rosa","Mei","Laura"]
FIRST_M = ["James","Robert","John","Michael","David","William","Richard","Thomas","Daniel","Marcus","Omar","Anil","Wei","Ahmed","Carlos","Kevin","Brian","Luis","Samuel","Paul"]
LAST = ["Anderson","Brooks","Chen","Diaz","Ellis","Foster","Gupta","Harris","Ibrahim","Johnson","Kaplan","Lee","Morrison",
        "Nguyen","Okafor","Patel","Quinn","Reyes","Sharma","Tanaka","Underwood","Vega","Walsh","Xu","Young","Zimmerman","Novak","Rahimi","Kowalski","Murphy"]

# ---- Deterministic random helpers: every value is a hash of the row's keys, so every run is identical ----
def U(salt, *cols):
    """Uniform [0, 1) from a hash of the given columns."""
    return (F.abs(F.xxhash64(*[F.col(c) if isinstance(c, str) else c for c in cols], F.lit(salt))) % 1000003) / 1000003.0

def Z(salt, *cols):
    """Standard normal (Box-Muller) from two hashes."""
    u1 = U(salt + "a", *cols) * 0.9998 + 0.0001
    return F.sqrt(-2 * F.log(u1)) * F.cos(2 * math.pi * U(salt + "b", *cols))

def pick(u, options):
    """Weighted choice: options = [(value, weight), ...], weights sum to 1."""
    cum, expr = 0.0, None
    for v, w in options[:-1]:
        cum += w
        expr = F.when(u < cum, F.lit(v)) if expr is None else expr.when(u < cum, F.lit(v))
    return expr.otherwise(F.lit(options[-1][0]))

def lit_map(d):
    return F.create_map(*[F.lit(x) for kv in d.items() for x in kv])

def from_list(values, u):
    return F.element_at(F.array(*[F.lit(v) for v in values]), (F.floor(u * len(values)) + 1).cast("int"))

fac_df = spark.createDataFrame(FACILITIES, "facility_code string, facility_name string, city string, tier string, base_occ double, los_mult double, before_noon double")
unit_df = spark.createDataFrame([(i, *u) for i, u in enumerate(UNITS)], "unit_idx int, facility_code string, unit string, unit_name string, unit_type string, staffed_beds int") \
               .withColumn("unit_code", F.concat_ws("-", "facility_code", "unit"))
drg_df = spark.createDataFrame(DRGS, "drg string, drg_desc string, gmlos double, service_line string, surgical int")
pool_df = spark.createDataFrame(
    [(t, [(d, sum(w for _, w in p[:i + 1]) if i < len(p) - 1 else 1.01) for i, (d, _) in enumerate(p)]) for t, p in POOLS.items()],
    "unit_type string, pool array<struct<drg:string, cum:double>>")
print(f"{len(FACILITIES)} hospitals, {len(UNITS)} units, {sum(u[4] for u in UNITS)} staffed beds, {len(DRGS)} DRGs")
```

**Output:**

```text
8 hospitals, 52 units, 1214 staffed beds, 29 DRGs
```


---
### Cell 6: Step 3 - Simulate encounters (calibrated to target occupancy) and save the answer key

```python
# 3a. Admissions per unit per day, sized from the target occupancy (with headroom, thinned in 3c).
days = spark.sql(f"SELECT explode(sequence(DATE'{SIM_START}', DATE'{AS_OF_DATE}')) AS d")
grid = (unit_df.join(fac_df, "facility_code").crossJoin(days)
        .withColumn("occ_target", F.least(F.lit(0.99), F.greatest(F.lit(0.3),
                    F.col("base_occ") + lit_map(TYPE_OCC)[F.col("unit_type")]
                    + F.coalesce(lit_map(SEASON)[F.month("d")], F.lit(0.0)))))
        .withColumn("lam", F.col("staffed_beds") * F.col("occ_target") * lit_map(DOW)[F.dayofweek("d")]
                    / lit_map(ALOS_EST)[F.col("unit_type")] * 1.3)
        .withColumn("n", F.greatest(F.lit(0), F.round(F.col("lam") + F.sqrt("lam") * Z("n", "unit_code", "d"))).cast("int")))

e = (grid.filter("n > 0").withColumn("seq", F.explode(F.sequence(F.lit(1), F.col("n"))))
     .withColumn("csn", (F.lit(9000000000) + F.datediff("d", F.lit(SIM_START)) * 100000 + F.col("unit_idx") * 1000 + F.col("seq")).cast("long"))
     .join(pool_df, "unit_type"))
e = (e.withColumn("u_drg", U("drg", "csn"))
      .withColumn("drg", F.expr("element_at(filter(pool, x -> x.cum >= u_drg), 1).drg"))
      .join(drg_df, "drg"))

# 3b. Patient and encounter attributes
pool = F.when(F.col("unit_type") == "Postpartum", "OB").when(F.col("unit_type") == "Behavioral", "BH").otherwise("GEN")
e = (e.withColumn("pool", pool)
      .withColumn("mrn_num", F.when(F.col("pool") == "OB", 5000000 + F.abs(F.xxhash64("csn", F.lit("mrn"))) % 30000)
                              .when(F.col("pool") == "BH", 6000000 + F.abs(F.xxhash64("csn", F.lit("mrn"))) % 20000)
                              .otherwise(1000000 + F.abs(F.xxhash64("csn", F.lit("mrn"))) % 160000))
      .withColumn("mrn", F.concat(F.lit("NM"), F.lpad(F.col("mrn_num").cast("string"), 8, "0")))
      .withColumn("age_years", F.when(F.col("pool") == "OB", 18 + F.floor(24 * U("age", "mrn")))
                                .when(F.col("pool") == "BH", 18 + F.floor(52 * U("age", "mrn")))
                                .otherwise(18 + F.floor(80 * F.pow(U("age", "mrn"), 0.75))))
      .withColumn("birth_date", F.date_sub(F.lit(AS_OF_DATE).cast("date"),
                                           (F.col("age_years") * 365.25 + F.floor(U("bd", "mrn") * 364)).cast("int")))
      .withColumn("sex", F.when(F.col("pool") == "OB", "F").when(U("sex", "mrn") < 0.51, "F").otherwise("M"))
      .withColumn("first_name", F.when(F.col("sex") == "F", from_list(FIRST_F, U("fn", "mrn"))).otherwise(from_list(FIRST_M, U("fn", "mrn"))))
      .withColumn("last_name", from_list(LAST, U("ln", "mrn")))
      .withColumn("age", F.floor(F.months_between("d", "birth_date") / 12).cast("int"))
      .withColumn("is_75", (F.col("age") >= 75).cast("int")))

age65 = F.col("age") >= 65
e = (e.withColumn("fin_class", F.when(age65, pick(U("pay", "csn"), [("Medicare", .52), ("Medicare Advantage", .30), ("Commercial", .10), ("Medicaid", .05), ("Other", .03)]))
                                 .otherwise(pick(U("pay", "csn"), [("Commercial", .55), ("Medicaid", .25), ("Self-Pay", .07), ("Medicare", .05), ("Medicare Advantage", .03), ("Other", .05)])))
      .withColumn("lives_alone", (U("alone", "csn") < 0.10 + 0.18 * F.col("is_75")).cast("int"))
      .withColumn("admitted_from", F.when(F.col("is_75") == 1, pick(U("from", "csn"), [("Skilled Nursing Facility", .07), ("Assisted Living", .05), ("Home", .88)]))
                                    .otherwise(pick(U("from", "csn"), [("Skilled Nursing Facility", .01), ("Home", .99)])))
      .withColumn("complex_discharge", (U("cx", "csn") < 0.05 + 0.10 * F.col("is_75") + 0.08 * (F.col("fin_class") == "Medicaid").cast("int")
                                        + 0.12 * (F.col("admitted_from") != "Home").cast("int") + 0.06 * F.col("lives_alone")).cast("int"))
      .withColumn("admit_source", F.when(F.col("surgical") == 1, pick(U("src", "csn"), [("Elective / Scheduled", .62), ("Emergency Department", .26), ("Transfer from Hospital", .12)]))
                                   .when(F.col("unit_type") == "Postpartum", pick(U("src", "csn"), [("Elective / Scheduled", .35), ("Emergency Department", .65)]))
                                   .when(F.col("facility_code") == "LMH", pick(U("src", "csn"), [("Emergency Department", .64), ("Transfer from Hospital", .16), ("Direct Admit", .14), ("Elective / Scheduled", .06)]))
                                   .otherwise(pick(U("src", "csn"), [("Emergency Department", .74), ("Transfer from Hospital", .06), ("Direct Admit", .14), ("Elective / Scheduled", .06)])))
      .withColumn("admit_hour", F.when(F.col("admit_source") == "Elective / Scheduled", 6 + F.floor(3 * U("ah", "csn")))
                                 .when(F.col("admit_source") == "Emergency Department", 10 + F.floor(14 * F.pow(U("ah", "csn"), 0.8)))
                                 .when(F.col("admit_source") == "Direct Admit", 9 + F.floor(9 * U("ah", "csn")))
                                 .otherwise(F.floor(24 * U("ah", "csn"))))
      .withColumn("admit_ts", F.timestamp_seconds(F.unix_timestamp(F.col("d").cast("timestamp")) + F.col("admit_hour") * 3600
                                                  + F.floor(U("am", "csn") * 60) * 60))
      .withColumn("converted_obs", ((F.col("unit_type") == "Observation") & (U("conv", "csn") < 0.10)).cast("int"))
      .withColumn("patient_class", F.when((F.col("unit_type") == "Observation") & (F.col("converted_obs") == 0), "Observation").otherwise("Inpatient"))
      .withColumn("attending", F.concat(F.lit("Dr. "), from_list(FIRST_F + FIRST_M, U("doc1", "facility_code", "service_line", F.floor(U("doc", "csn") * 5))),
                                        F.lit(" "), from_list(LAST, U("doc2", "facility_code", "service_line", F.floor(U("doc", "csn") * 5))))))

# Length of stay: Vizient-style expected LOS, then observed clinical days, plus avoidable days from barriers
story = (F.when((F.col("facility_code") == "LMH") & F.col("service_line").isin("Oncology", "Neurosciences"), 1.18)
          .when((F.col("facility_code") == "WCH") & (F.col("service_line") == "Medicine"), 1.12).otherwise(1.0))
fy26 = F.col("d") >= F.lit(FY2026_START)
e = (e.withColumn("expected_los", F.round(F.greatest(F.lit(0.8), F.col("gmlos") * (1 + 0.006 * (F.col("age") - 60)) * (1 + 0.12 * F.col("complex_discharge"))), 2))
      .withColumn("clinical_days", F.when(F.col("patient_class") == "Observation", 0.35 + 1.15 * U("obs", "csn"))
                  .otherwise(F.col("expected_los") * F.exp(0.40 * Z("los", "csn") - 0.06) * F.col("los_mult") * story
                             * F.when(F.dayofweek("d").isin(6, 7), 1.06).otherwise(1.0)
                             * F.when(F.col("admit_source") == "Transfer from Hospital", 1.10).otherwise(1.0)
                             * (1 + 0.18 * F.col("complex_discharge"))))
      .withColumn("p_barrier", F.when(F.col("patient_class") == "Observation", 0.0).otherwise(
                  (0.05 + 0.20 * F.col("complex_discharge") + 0.08 * F.col("is_75") + 0.07 * F.col("lives_alone")
                   + 0.05 * F.col("fin_class").isin("Medicaid", "Medicare Advantage").cast("int")
                   + 0.10 * (F.col("admitted_from") != "Home").cast("int"))
                  * F.when(F.col("unit_type").isin("Postpartum", "Behavioral"), 0.3).otherwise(1.0)
                  * F.when(fy26, 1.25).otherwise(1.0)))
      .withColumn("has_barrier", (U("bar", "csn") < F.col("p_barrier")).cast("int"))
      .withColumn("barrier_category", F.when(F.col("has_barrier") == 0, None)
                  .when((F.col("is_75") == 1) | (F.col("admitted_from") != "Home"),
                        pick(U("cat", "csn"), [("SNF placement", .45), ("Insurance authorization", .18), ("Home health / DME", .12),
                                               ("Family / guardianship", .10), ("Pending test or procedure", .08), ("Transportation", .07)]))
                  .otherwise(pick(U("cat", "csn"), [("SNF placement", .20), ("Insurance authorization", .22), ("Home health / DME", .15),
                                                    ("Family / guardianship", .12), ("Pending test or procedure", .18), ("Transportation", .13)])))
      .withColumn("avoidable_days", F.when(F.col("has_barrier") == 0, 0).otherwise(
                  F.when(F.col("barrier_category") == "SNF placement", 1 + F.floor(5 * U("ad", "csn")) + F.when(fy26, 1).otherwise(0))
                   .when(F.col("barrier_category") == "Insurance authorization", 1 + F.floor(3 * U("ad", "csn")))
                   .when(F.col("barrier_category") == "Family / guardianship", 1 + F.floor(4 * U("ad", "csn")))
                   .when(F.col("barrier_category").isin("Home health / DME", "Pending test or procedure"), 1 + F.floor(2 * U("ad", "csn")))
                   .otherwise(1)).cast("int"))
      .withColumn("expired", ((F.col("pool") == "GEN") & (U("exp", "csn") < F.when(F.col("is_75") == 1, 0.03).otherwise(0.008))).cast("int"))
      .withColumn("disposition", F.when(F.col("expired") == 1, "Expired")
                  .when(F.col("barrier_category") == "SNF placement", "Skilled Nursing Facility")
                  .when(F.col("barrier_category") == "Home health / DME", "Home Health")
                  .when(F.col("is_75") == 1, pick(U("disp", "csn"), [("Home Health", .25), ("Skilled Nursing Facility", .12), ("Inpatient Rehab", .05),
                                                                    ("Hospice", .03), ("Transfer to Acute Hospital", .01), ("Home", .54)]))
                  .otherwise(pick(U("disp", "csn"), [("Home Health", .11), ("Skilled Nursing Facility", .03), ("Inpatient Rehab", .03),
                                                     ("Hospice", .01), ("Left AMA", .015), ("Transfer to Acute Hospital", .02), ("Home", .785)])))
      .withColumn("los_target", F.col("clinical_days") + F.col("avoidable_days"))
      .withColumn("disch_target", F.timestamp_seconds(F.unix_timestamp("admit_ts") + (F.col("los_target") * 86400).cast("long"))))

# Inpatient discharges snap to a discharge hour; PCH's before-noon program starts Mar 2026
before_noon_p = F.when((F.col("facility_code") == "PCH") & (F.col("disch_target") >= F.lit("2026-03-01")), 0.33).otherwise(F.col("before_noon"))
hour = F.when(U("bn", "csn") < before_noon_p, 8 + F.floor(4 * U("dh", "csn"))).otherwise(12 + F.floor(8 * F.pow(U("dh", "csn"), 1.4)))
snapped = F.timestamp_seconds(F.unix_timestamp(F.to_date("disch_target").cast("timestamp")) + hour * 3600 + F.floor(U("dm", "csn") * 60) * 60)
snapped = F.when(snapped < F.col("admit_ts") + F.expr("INTERVAL 12 HOURS"), snapped + F.expr("INTERVAL 1 DAY")).otherwise(snapped)
e = e.withColumn("discharge_ts", F.when(F.col("patient_class") == "Observation", F.col("disch_target")).otherwise(snapped))

# 3c. Calibrate: thin admissions so each unit's bed-day DEMAND per month is ~15% above its target occupancy.
# Step 3d then caps each unit at its beds, so busy units run full and the excess demand is diverted.
DEMAND = 1.15
bed_days = e.groupBy("unit_code", F.trunc("d", "MM").alias("m")).agg(
    F.sum((F.unix_timestamp("discharge_ts") - F.unix_timestamp("admit_ts")) / 86400).alias("gen_days"))
target = grid.groupBy("unit_code", F.trunc("d", "MM").alias("m")).agg(F.sum(F.col("staffed_beds") * F.col("occ_target")).alias("target_days"))
keep = bed_days.join(target, ["unit_code", "m"]).select("unit_code", "m", F.least(F.lit(1.0), F.col("target_days") * DEMAND / F.col("gen_days")).alias("keep_p"))
e = (e.withColumn("m", F.trunc("d", "MM")).join(keep, ["unit_code", "m"])
      .filter(U("keep", "csn") < F.col("keep_p"))
      .filter(F.col("admit_ts") <= F.lit(AS_OF_TS).cast("timestamp"))
      .filter(F.col("discharge_ts") >= F.lit(REPORT_START).cast("timestamp")))

cand = e.select(
    "csn", "mrn", "first_name", "last_name", "birth_date", "sex", "age", "facility_code", "facility_name", "unit_code", "unit",
    "unit_name", "unit_type", "staffed_beds", "drg", "drg_desc", "service_line", "surgical", "fin_class", "admit_source",
    "admitted_from", "lives_alone", "complex_discharge", "patient_class", "converted_obs", "attending", "admit_ts",
    "discharge_ts", "expected_los", "clinical_days", "has_barrier", "barrier_category", "avoidable_days", "disposition",
    ((F.unix_timestamp("discharge_ts") - F.unix_timestamp("admit_ts")) / 86400).alias("los_days"),
    (F.col("discharge_ts") > F.lit(AS_OF_TS).cast("timestamp")).alias("in_house_at_as_of"))

# 3d. Capacity: a unit can't hold more patients than its staffed beds (plus a few hallway / boarding beds
# on non-ICU units). Admissions that arrive when the unit is full are diverted, as they are in real life.
# This is a sequential queue simulation per unit, so it runs on the driver over a small 4-column extract.
import heapq
BEDS = {f"{u[0]}-{u[1]}": u[4] for u in UNITS}
TYPE = {f"{u[0]}-{u[1]}": u[3] for u in UNITS}
def capacity(unit_code, ts):
    beds = BEDS[unit_code]
    for fac, unit, new_beds, start, end in BED_CHANGES:
        if unit_code == f"{fac}-{unit}" and str(ts) >= start and (end is None or str(ts) < end):
            beds = new_beds
    return beds if "ICU" in TYPE[unit_code] else beds + max(1, round(beds * 0.04))

pdf = cand.select("csn", "unit_code", "admit_ts", "discharge_ts").orderBy("unit_code", "admit_ts", "csn").toPandas()
accepted, diverted = [], 0
for unit_code, g in pdf.groupby("unit_code", sort=False):
    in_bed = []                                   # discharge times of patients currently on the unit
    for csn, admit, disch in zip(g["csn"], g["admit_ts"], g["discharge_ts"]):
        while in_bed and in_bed[0] <= admit:
            heapq.heappop(in_bed)
        if len(in_bed) < capacity(unit_code, admit):
            heapq.heappush(in_bed, disch)
            accepted.append(int(csn))
        else:
            diverted += 1
accepted_df = spark.createDataFrame([(c,) for c in accepted], "csn long")
print(f"Admitted {len(accepted):,} encounters; {diverted:,} diverted because the unit was full")

truth = cand.join(accepted_df, "csn")
truth.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{RAW}._sim_truth_encounters")
spark.sql(f"COMMENT ON TABLE {RAW}._sim_truth_encounters IS 'Simulation answer key (true outcomes, incl. future discharges of current inpatients). Not read by the pipeline; used only to evaluate the model in notebook 05.'")
T = spark.table(f"{RAW}._sim_truth_encounters")
display(T.groupBy("facility_code").agg(F.count("*").alias("encounters"), F.sum(F.col("in_house_at_as_of").cast("int")).alias("in_house_now"),
                                       F.round(F.avg("los_days"), 2).alias("alos"), F.round(F.sum("los_days") / F.sum("expected_los"), 3).alias("o_e"),
                                       F.round(F.avg("has_barrier"), 3).alias("barrier_rate")).orderBy("facility_code"))
```

**Output:**

```text
Admitted 218,722 encounters; 27,288 diverted because the unit was full
```

**Output (table):**

| facility_code | encounters | in_house_now | alos | o_e | barrier_rate |
| --- | --- | --- | --- | --- | --- |
| PCH | 44925 | 211 | 3.56 | 1.101 | 0.097 |
| RBH | 17261 | 66 | 3.13 | 1.053 | 0.089 |
| ORH | 13505 | 52 | 3.3 | 1.022 | 0.101 |
| CFH | 8921 | 37 | 3.28 | 1.026 | 0.101 |
| BSH | 21608 | 85 | 2.83 | 0.956 | 0.093 |
| FVH | 16824 | 69 | 3.56 | 1.065 | 0.085 |
| LMH | 77381 | 459 | 4.43 | 1.174 | 0.106 |
| WCH | 18297 | 85 | 3.43 | 1.152 | 0.095 |


---
### Cell 7: Step 4 - Master data and staffed bed capacity (CSV)

```python
from pyspark.sql import Window

def write_csv(df, folder, files=1):
    df.coalesce(files).write.mode("append").option("header", True).csv(f"{LANDING}/{folder}")

def write_json(df, folder, files=1):
    df.coalesce(files).write.mode("append").json(f"{LANDING}/{folder}")

write_csv(fac_df.select(F.col("facility_code").alias("FACILITY_CODE"), F.col("facility_name").alias("FACILITY_NAME"),
                        F.col("city").alias("CITY"), F.col("tier").alias("TIER"), F.lit("Y").alias("ACTIVE")), "mdm_facilities")
write_csv(drg_df.select(F.col("drg").alias("MS_DRG"), F.col("drg_desc").alias("DRG_DESCRIPTION"), F.col("gmlos").cast("string").alias("GMLOS"),
                        F.col("service_line").alias("SERVICE_LINE"), F.col("surgical").cast("string").alias("SURGICAL_FLAG")), "mdm_drg")
xw = [(dom, v, canon, i) for dom, m in CODE_VARIANTS.items() for i, (canon, al) in enumerate(m.items(), 1) for v in [canon] + al]
write_csv(spark.createDataFrame(xw, "DOMAIN string, SOURCE_VALUE string, CANONICAL_VALUE string, SORT_ORDER int"), "mdm_code_xwalk")

# Staffed beds, effective-dated. A bed change splits the unit's row into before / during / after periods.
cap_rows = []
for fac, unit, name, utype, beds in UNITS:
    periods = [(SIM_START, None, beds)]
    for cfac, cunit, new_beds, start, end in BED_CHANGES:
        if (cfac, cunit) == (fac, unit):
            periods = [(SIM_START, start, beds), (start, end, new_beds)] + ([(end, None, beds)] if end else [])
    for start, end, b in periods:
        cap_rows.append((fac, unit, name, utype, str(b), str(round(b * 1.15)), start, end or ""))
write_csv(spark.createDataFrame(cap_rows, "FACILITY string, UNIT_CODE string, UNIT_NAME string, UNIT_TYPE string, STAFFED_BEDS string, "
                                          "LICENSED_BEDS string, EFFECTIVE_FROM string, EFFECTIVE_TO string"), "bed_capacity")
print(f"Master data: {len(FACILITIES)} facilities, {len(DRGS)} DRGs, {len(xw)} code spellings; bed capacity: {len(cap_rows)} rows")
```

**Output:**

```text
Master data: 8 facilities, 29 DRGs, 78 code spellings; bed capacity: 55 rows
```


---
### Cell 8: Step 5 - Epic encounter / DRG extract (CSV, monthly + nightly snapshots)

```python
AS_OF = F.lit(AS_OF_TS).cast("timestamp")
T = spark.table(f"{RAW}._sim_truth_encounters")

# Patients in the hospital now get real bed numbers (1..n per unit); past stays get a random bed on the unit.
ranked = (T.filter("in_house_at_as_of")
           .withColumn("bed_no", F.row_number().over(Window.partitionBy("unit_code").orderBy("admit_ts", "csn")))
           .select("csn", "bed_no"))
T2 = (T.join(ranked, "csn", "left")
       .withColumn("bed", F.concat(F.col("unit"), F.lit("-"), F.lpad(F.coalesce(F.col("bed_no"),
                   1 + F.abs(F.xxhash64("csn", F.lit("bed"))) % F.col("staffed_beds")).cast("string"), 2, "0"))))

def variant(colname, domain, salt, p=0.5):
    """Canonical value with probability p, otherwise one of the domain's source spellings."""
    m = F.create_map(*[x for k, al in CODE_VARIANTS[domain].items() for x in (F.lit(k), F.array(*[F.lit(a) for a in al]))])
    u = U(salt, "csn")
    return (F.when(F.col(colname).isNull(), F.lit(None)).when(u < p, F.col(colname))
             .otherwise(F.element_at(m[F.col(colname)], F.when(u < p + (1 - p) / 2, 1).otherwise(2))))

def encounter_extract(df, extract_ts, show_discharge=True):
    disch = F.col("discharge_ts") if show_discharge else F.lit(None).cast("timestamp")
    u = lambda s: U(s, "csn")
    return df.select(
        F.col("csn").cast("string").alias("PAT_ENC_CSN_ID"), F.col("mrn").alias("PAT_MRN_ID"),
        F.concat(F.upper("last_name"), F.lit(","), F.col("first_name")).alias("PAT_NAME"),
        F.date_format("birth_date", "MM/dd/yyyy").alias("BIRTH_DATE"), F.col("sex").alias("SEX"),
        F.when(u("fac") < 0.6, F.col("facility_code")).otherwise(F.col("facility_name")).alias("FACILITY"),
        F.col("unit").alias("UNIT"), F.col("unit_name").alias("DEPARTMENT_NAME"),
        F.when(disch.isNull(), F.col("bed")).alias("ROOM_BED"),
        F.date_format("admit_ts", "MM/dd/yyyy HH:mm").alias("ADMIT_DTTM"),
        F.date_format(disch, "MM/dd/yyyy HH:mm").alias("DISCH_DTTM"),
        variant("admit_source", "admit_source", "v_src").alias("ADMIT_SOURCE"),
        variant("patient_class", "patient_class", "v_cls").alias("PATIENT_CLASS"),
        F.col("service_line").alias("HOSP_SERVICE"),
        F.when(u("drgf") < 0.5, F.col("drg")).when(u("drgf") < 0.75, F.col("drg").cast("int").cast("string"))
         .otherwise(F.concat(F.lit("MS-DRG "), F.col("drg"))).alias("MSDRG"),
        variant("fin_class", "fin_class", "v_fin").alias("FIN_CLASS"),
        F.col("attending").alias("ATTENDING_PROV"),
        F.when(u("flag") < 0.5, F.when(F.col("complex_discharge") == 1, "Y").otherwise("N"))
         .otherwise(F.col("complex_discharge").cast("string")).alias("COMPLEX_DC_FLAG"),
        F.when(F.col("lives_alone") == 1, "Y").otherwise("N").alias("LIVES_ALONE"),
        F.col("admitted_from").alias("ADMITTED_FROM"),
        F.when(disch.isNotNull(), variant("disposition", "disposition", "v_disp")).alias("DISCH_DISP"),
        F.format_number("expected_los", 2).alias("EXP_LOS_DAYS"),
        (F.concat(F.date_add(F.to_date("discharge_ts"), 1).cast("string"), F.lit(" 05:00:00")) if extract_ts is None
         else F.lit(extract_ts)).alias("EXTRACT_TS"))

RECENT = "2026-10-27 07:00:00"
SNAP_1028 = "2026-10-28 06:00:00"
hist = T2.filter(F.col("discharge_ts") < F.lit(RECENT).cast("timestamp"))
write_csv(encounter_extract(hist, None).repartition(26, "EXTRACT_TS"), "epic_encounters", files=26)
# Nightly snapshot on Oct 28: everyone in the hospital at 6:00 AM, before discharge
snap = T2.filter((F.col("admit_ts") <= F.lit(SNAP_1028).cast("timestamp")) & (F.col("discharge_ts") > F.lit(SNAP_1028).cast("timestamp")))
write_csv(encounter_extract(snap, SNAP_1028, show_discharge=False), "epic_encounters")
# Oct 30 snapshot: recent discharges (now complete) and everyone in the hospital now
final = T2.filter(F.col("discharge_ts") >= F.lit(RECENT).cast("timestamp"))
final_ex = encounter_extract(final.filter("NOT in_house_at_as_of"), AS_OF_TS).unionByName(
           encounter_extract(final.filter("in_house_at_as_of"), AS_OF_TS, show_discharge=False))
write_csv(final_ex, "epic_encounters")
print(f"Encounters: {hist.count():,} historical, {snap.count():,} in the Oct 28 snapshot, {final.count():,} in the Oct 30 snapshot")
```

**Output:**

```text
Encounters: 216,873 historical, 1,009 in the Oct 28 snapshot, 1,849 in the Oct 30 snapshot
```


---
### Cell 9: Step 6 - Epic ADT events (HL7-style JSON: monthly files + 2 daily files)

```python
admit_cls = F.when(F.col("patient_class") == "Observation", "O").when(F.col("converted_obs") == 1, "O").otherwise("I")
los_sec = F.unix_timestamp("discharge_ts") - F.unix_timestamp("admit_ts")
events = (
    T2.select("csn", "mrn", "facility_code", "unit", "bed", F.lit("A01").alias("typ"), F.col("admit_ts").alias("event_ts"), admit_cls.alias("cls"))
    .unionByName(T2.filter("converted_obs = 1 AND discharge_ts > admit_ts + INTERVAL 20 HOURS")   # obs -> inpatient
                 .select("csn", "mrn", "facility_code", "unit", "bed", F.lit("A06").alias("typ"),
                         (F.col("admit_ts") + F.expr("INTERVAL 20 HOURS")).alias("event_ts"), F.lit("I").alias("cls")))
    .unionByName(T2.filter((U("xfer", "csn") < 0.2) & (los_sec > 86400) & ~F.col("in_house_at_as_of"))
                 .select("csn", "mrn", "facility_code", "unit",
                         F.concat(F.col("unit"), F.lit("-"), F.lpad((1 + F.abs(F.xxhash64("csn", F.lit("bed2"))) % F.col("staffed_beds")).cast("string"), 2, "0")).alias("bed"),
                         F.lit("A02").alias("typ"), F.timestamp_seconds(F.unix_timestamp("admit_ts") + (los_sec * 0.4).cast("long")).alias("event_ts"),
                         F.when(F.col("patient_class") == "Observation", "O").otherwise("I").alias("cls")))
    .unionByName(T2.filter(~F.col("in_house_at_as_of"))
                 .select("csn", "mrn", "facility_code", "unit", "bed", F.lit("A03").alias("typ"), F.col("discharge_ts").alias("event_ts"),
                         F.when(F.col("patient_class") == "Observation", "O").otherwise("I").alias("cls")))
    .filter(F.col("event_ts") <= AS_OF))

# Lakeshore City local time; the ISO form carries the UTC offset, the HL7 form doesn't
cdt = (F.month("event_ts").between(4, 10) | ((F.month("event_ts") == 3) & (F.dayofmonth("event_ts") >= 9))
       | ((F.month("event_ts") == 11) & (F.dayofmonth("event_ts") < 2)))
key = F.concat_ws("-", F.col("csn").cast("string"), "typ")
msg = events.select(
    key.alias("control_id"), F.concat(F.lit("ADT^"), "typ").alias("message_type"),
    F.when(U("tsf", key) < 0.6, F.concat(F.date_format("event_ts", "yyyy-MM-dd'T'HH:mm:ss"), F.when(cdt, "-05:00").otherwise("-06:00")))
     .otherwise(F.date_format("event_ts", "yyyyMMddHHmmss")).alias("event_ts"),
    F.struct(F.col("mrn"), F.col("csn").cast("string").alias("csn")).alias("patient"),
    F.struct(F.col("facility_code").alias("facility"),
             F.when(U("unitf", key) < 0.4, F.col("unit"))
              .when(U("unitf", key) < 0.7, F.concat_ws("-", "facility_code", "unit"))
              .otherwise(F.lower(F.concat_ws(" ", "facility_code", "unit"))).alias("unit"),
             F.col("bed")).alias("location"),
    F.col("cls").alias("patient_class"), F.lit("EPIC").alias("sending_app"),
    F.col("event_ts").alias("_ts"), F.col("typ").alias("_typ"), key.alias("_key"))

DAY1, DAY2 = "2026-10-29", "2026-10-30"
late = (F.col("_typ") == "A03") & (F.col("_ts") >= F.lit("2026-09-01").cast("timestamp")) & (U("late", "_key") < 0.004)
hist_msg = msg.filter((F.col("_ts") < F.lit(DAY1).cast("timestamp")) & ~late)
resent = msg.filter((F.col("_ts") >= F.lit("2026-09-29").cast("timestamp")) & (F.col("_ts") < F.lit(DAY1).cast("timestamp"))
                    & (U("dup", "_key") < 0.015))
day1 = msg.filter(F.to_date("_ts") == F.lit(DAY1)).unionByName(resent)
day2 = msg.filter(F.to_date("_ts") == F.lit(DAY2)).unionByName(msg.filter((F.col("_ts") < F.lit(DAY1).cast("timestamp")) & late))
OUT = ["control_id", "message_type", "event_ts", "patient", "location", "patient_class", "sending_app"]
write_json(hist_msg.repartition(26, F.trunc("_ts", "MM")).select(OUT), "epic_adt", files=26)
write_json(day1.select(OUT), "epic_adt")
write_json(day2.select(OUT), "epic_adt")
print(f"ADT messages: {hist_msg.count():,} historical, {day1.count():,} on Oct 29 (incl. re-sent), {day2.count():,} on Oct 30 (incl. late)")
```

**Output:**

```text
ADT messages: 477,716 historical, 813 on Oct 29 (incl. re-sent), 115 on Oct 30 (incl. late)
```


---
### Cell 10: Step 7 - CarePort avoidable-day barriers (JSON)

```python
b = (T2.filter("has_barrier = 1")
       .withColumn("opened", F.date_add(F.to_date("admit_ts"), F.greatest(F.lit(1), F.floor("clinical_days")).cast("int")))
       .withColumn("resolved", F.date_add("opened", F.col("avoidable_days")))
       .filter(F.col("opened") <= F.lit(AS_OF_DATE).cast("date")))                 # not yet raised = not in CarePort
open_now = F.col("in_house_at_as_of") & (F.col("resolved") > F.lit(AS_OF_DATE).cast("date"))
cm = F.create_map(*[x for k, al in CODE_VARIANTS["barrier_category"].items() for x in (F.lit(k), F.array(*[F.lit(a) for a in al]))])
bar = b.select(
    F.concat(F.lit("CP"), F.col("csn").cast("string")).alias("barrier_id"), F.col("csn").cast("string").alias("csn"),
    F.when(U("bc", "csn") < 0.4, F.col("barrier_category"))
     .otherwise(F.element_at(cm[F.col("barrier_category")], F.when(U("bc", "csn") < 0.7, 1).otherwise(2))).alias("category"),
    F.col("opened").cast("string").alias("opened_dt"),
    F.when(open_now, None).otherwise(F.col("resolved").cast("string")).alias("resolved_dt"),
    F.when(open_now, F.datediff(F.lit(AS_OF_DATE).cast("date"), "opened")).otherwise(F.col("avoidable_days")).alias("avoidable_days"),
    F.when(open_now, "Open").otherwise("Resolved").alias("status"),
    F.when(open_now, F.lit(AS_OF_TS)).otherwise(F.concat(F.col("resolved").cast("string"), F.lit(" 18:00:00"))).alias("updated_at"))
hist_bar = bar.filter(F.col("updated_at") < F.lit(DAY1))
write_json(hist_bar, "careport_barriers", files=4)
write_json(bar.filter(F.col("updated_at") >= F.lit(DAY1)), "careport_barriers")
print(f"Barriers: {hist_bar.count():,} historical, {bar.filter(F.col('updated_at') >= F.lit(DAY1)).count():,} updated Oct 29-30 "
      f"({bar.filter(F.col('status') == 'Open').count()} open now)")
```

**Output:**

```text
Barriers: 21,325 historical, 126 updated Oct 29-30 (74 open now)
```


---
### Cell 11: Step 8 - Inventory of the landing volume

```python
FOLDERS = [("epic_adt", "json"), ("epic_encounters", "csv"), ("careport_barriers", "json"), ("bed_capacity", "csv"),
           ("mdm_facilities", "csv"), ("mdm_drg", "csv"), ("mdm_code_xwalk", "csv")]
rows = []
for folder, fmt in FOLDERS:
    path = f"{LANDING}/{folder}"
    files = [f for f in dbutils.fs.ls(path) if f.name.endswith(f".{fmt}")]
    reader = spark.read.option("header", True) if fmt == "csv" else spark.read
    rows.append((folder, fmt, len(files), getattr(reader, fmt)(f"{path}/*.{fmt}").count()))
display(spark.createDataFrame(rows, "folder string, format string, data_files int, rows long"))
print("Sample raw ADT messages (note the timestamp and unit formats):")
display(spark.read.json(f"{LANDING}/epic_adt/*.json").limit(5))
print(f"Next: open notebook 02 and use catalog = {CATALOG}, prefix = {PREFIX}")
```

**Output (table):**

| folder | format | data_files | rows |
| --- | --- | --- | --- |
| epic_adt | json | 20 | 478644 |
| epic_encounters | csv | 28 | 219731 |
| careport_barriers | json | 2 | 21451 |
| bed_capacity | csv | 1 | 55 |
| mdm_facilities | csv | 1 | 8 |
| mdm_drg | csv | 1 | 29 |
| mdm_code_xwalk | csv | 1 | 78 |

**Output:**

```text
Sample raw ADT messages (note the timestamp and unit formats):
```

**Output (table):**

| control_id | event_ts | location | message_type | patient | patient_class | sending_app |
| --- | --- | --- | --- | --- | --- | --- |
| 9081605015-A01 | 2026-10-26T08:53:00-05:00 | ['9E-28', 'LMH', 'LMH-9E'] | ADT^A01 | ['9081605015', 'NM01088311'] | I | EPIC |
| 9081504007-A01 | 2026-10-25T08:17:00-05:00 | ['8E-10', 'LMH', '8E'] | ADT^A01 | ['9081504007', 'NM01113261'] | I | EPIC |
| 9027005011-A01 | 2025-04-28T18:15:00-05:00 | ['9E-32', 'LMH', 'nmh 9e'] | ADT^A01 | ['9027005011', 'NM01128705'] | I | EPIC |
| 9054700007-A01 | 2026-01-30T13:27:00-06:00 | ['MICU-33', 'LMH', 'nmh micu'] | ADT^A01 | ['9054700007', 'NM01128331'] | I | EPIC |
| 9054302003-A01 | 20260126063100 | ['CVICU-02', 'LMH', 'LMH-CVICU'] | ADT^A01 | ['9054302003', 'NM01098811'] | I | EPIC |

**Output:**

```text
Next: open notebook 02 and use catalog = schauhan_workspace_catalog, prefix = lsh_capacity
```
