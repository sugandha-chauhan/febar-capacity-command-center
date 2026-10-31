# 00 - Problem Statement

Exported from the Databricks workspace on 2026-10-31 06:17 UTC with each cell's **saved output** from its last run. Synthetic data; organization and hospital names anonymized.


---
### Cell 1: Problem statement (markdown)

# LSH Capacity Command Center: bed utilization and length of stay

An end-to-end Databricks prototype for a multi-hospital academic health system, modeled on Lakeshore Health (LSH). It runs on **synthetic data only**: no PHI and no real LSH volumes, beds or outcomes. All data is fixed as of the **7:00 AM capacity huddle on Oct 30, 2026**.

## 1. Industry and customer
* **Industry:** Healthcare providers (acute care hospitals). Outcome area: *capacity management and patient flow*.
* **Customer:** an academic medical center (LMH) plus 7 community hospitals (PCH, BSH, RBH, ORH, FVH, WCH, CFH), 1,214 staffed beds on 52 units. In the synthetic data, about 27,000 admissions over two years are diverted because the unit was full: the ED boarding and declined transfers this prototype aims to reduce.
* **Who uses it:**

| Persona | Question they need answered | Today |
| --- | --- | --- |
| VP Patient Flow / CNO | Where will we run out of beds today and tomorrow? | Phone calls and a bed board snapshot |
| Unit case manager / physician advisor | Which of my patients will stay past their expected discharge date, and what's blocking them? | Manual huddle lists in spreadsheets |
| Service line and quality leaders | Which service lines, DRGs and units drive excess days against the Vizient expected LOS? | A monthly Power BI LOS report, about 6 weeks behind |

## 2. The problem
LSH's flagship and larger community hospitals run at **90%+ midnight occupancy** on many days. When beds are full:
* ED patients board for hours waiting for an inpatient bed, and transfers from other hospitals are declined.
* Elective surgical cases are delayed, which costs margin.
* Patients who are medically ready to leave stay because of **avoidable days**: waiting for a skilled nursing facility (SNF) bed, insurance authorization, transport or home services.

Length of stay runs **above the Vizient expected LOS** (observed / expected index above 1.0) for several service lines. Every excess day is a bed nobody else can use. The data to act on this exists (ADT events, encounter and DRG data, CarePort barriers, staffed bed counts), but it lives in separate systems and is reported **after** the patient has left, not while there is still time to act.

**Problem statement:** *Case managers and patient flow leaders can't see, early in the stay, which current inpatients are likely to exceed their expected length of stay and why. So excess and avoidable days pile up, beds stay full, and the system loses ED throughput, transfers and surgical volume.*

## 3. Goals and success measures
Targets are illustrative for the prototype. The FY2026 baseline (Sep 2025 - Aug 2026) is calculated from the synthetic data in notebook 03 and matches the metric views in notebook 04.

| KPI | Definition | FY2026 synthetic baseline | Prototype target |
| --- | --- | --- | --- |
| LOS O/E index | Observed LOS / Vizient expected LOS (inpatient discharges) | 1.10 (BSH) to 1.34 (WCH); LMH 1.26 | 1.00 or below |
| Excess days | Sum of max(0, observed - expected LOS) | ~96,000 days (LMH 42,369) | -15% |
| Avoidable days | CarePort avoidable days | ~27,000 days; SNF placement is the top barrier | -20% |
| Midnight occupancy | Midnight census / staffed beds | LMH 88% average, many LMH units above 95% | 85-90% band, fewer days above 95% |
| Discharges before noon | Share of inpatient discharges before 12:00 | LMH 17%, PCH 27% (up after its Mar 2026 initiative) | 30%+ |
| Bed days freed | Excess days avoided, expressed as equivalent staffed beds | ~96,000 excess days = ~263 beds all year | -15% excess days frees ~39 staffed beds |

## 4. The solution: one integrated data journey
Each stage feeds the next. The app and the Genie agent read the same governed gold data, model scores and Lakebase tables, so there is one version of the truth.

| # | Notebook | Stage | What it does |
| --- | --- | --- | --- |
| 01 | Generate Raw Synthetic Data | Source systems | Creates messy source extracts in a UC volume: Epic ADT events (JSON, daily files, duplicates, late events), encounter / DRG extract (CSV), CarePort avoidable-day barriers (JSON), staffed bed capacity (CSV), and master data |
| 02 | Lakeflow Connect Ingestion | **Lakeflow** | Lakeflow Connect file connector (Auto Loader) loads every source into bronze tables incrementally, with lineage columns and rescued data |
| 03 | Silver and Gold Star Schema | **Lakeflow** | Cleans and conforms the data, then builds dims (date with LSH fiscal year, facility, unit, DRG, service, disposition, barrier) and facts (encounter, unit census by day, barriers, current inpatients) |
| 04 | Unity Catalog Governance and Metric Views | **Unity Catalog** | Comments, classification and PHI tags, PHI column masks, a facility row filter, least-privilege group grants, lineage, and the `los_metrics`, `capacity_metrics` and `barrier_metrics` metric views |
| 05 | ML Long Stay Risk and Discharge Prediction | **ML + GenAI** | Trains a long-stay risk classifier and an LOS regressor on admission-time features, tracks them in MLflow, registers both in UC (`@champion`), scores every current inpatient, and writes AI huddle notes with `ai_query` from de-identified facts |
| 06 | Lakebase Operational Serving | **Lakebase** | Syncs the scored bed board and unit status into Lakebase Postgres for millisecond app reads, and creates the write-back tables where case managers confirm discharge plans and log actions |
| 07 | Genie Agent for Capacity and LOS | **Genie** | A capacity and LOS analyst over the metric views and the de-identified bed board, with instructions, sample questions, example queries and 14 benchmarks |
| 08 | Databricks App Build Guide | **App** | Checks every app resource and prints a ready-to-paste AppKit prompt for the Capacity Command Center app: Command Center, Discharge Worklist (Lakebase write-back), LOS Performance and Ask (Genie) pages, in the LSH theme |

## 5. Demo storyline (10 minutes)
1. **7:00 AM huddle:** the app shows LMH at ~94% of staffed beds, the units with 2 or fewer beds forecast open tonight, and ~262 discharges expected today system-wide.
2. **Who to work first:** the at-risk list ranks current inpatients by long-stay risk. The top patient is past their expected discharge date with a SNF placement barrier.
3. **Act:** the case manager records an action (SNF referral escalated). It writes to Lakebase and appears immediately on the bed board.
4. **Ask why:** "Which service lines had the most excess days last quarter, and what are the top barriers?" Genie answers from the governed metric views.
5. **Trust:** show lineage from the raw ADT files to the app, the PHI masks, and the row filter that limits a community hospital leader to their own facility.

## 6. Scope
* **In scope:** 8 acute hospitals, inpatient and observation stays, Oct 2024 to Oct 30, 2026; batch / daily refresh with incremental ingestion; one long-stay model; one app.
* **Out of scope:** real-time HL7 streaming, OR scheduling, staffing optimization, writing back to Epic.
* **Path to production:** swap the file connector for a managed Lakeflow Connect connector or HL7 feed; schedule notebooks 02, 03, 05 and 06 as one Lakeflow Job; move silver and gold into a Lakeflow Spark Declarative Pipeline.

## 7. How to run
Run the notebooks in order (01 through 08) with the same **catalog** and **prefix** values. Defaults: catalog `schauhan_workspace_catalog`, prefix `lsh_capacity`. Objects go to `<prefix>_raw`, `<prefix>_bronze`, `<prefix>_silver` and `<prefix>_gold`.