# Live checks

Run by notebook 09 on 2026-10-31 06:17 UTC against the workspace. Synthetic data; names anonymized.


## Row counts by layer



| layer | table | rows |
| --- | --- | --- |
| lsh_capacity_bronze | bed_capacity | 55 |
| lsh_capacity_bronze | careport_barriers | 21451 |
| lsh_capacity_bronze | epic_adt_events | 478644 |
| lsh_capacity_bronze | epic_encounters | 219731 |
| lsh_capacity_bronze | mdm_code_xwalk | 78 |
| lsh_capacity_bronze | mdm_drg | 29 |
| lsh_capacity_bronze | mdm_facilities | 8 |
| lsh_capacity_gold | bed_board | 1064 |
| lsh_capacity_gold | current_inpatients | 1064 |
| lsh_capacity_gold | dim_date | 1248 |
| lsh_capacity_gold | dim_drg | 29 |
| lsh_capacity_gold | dim_facility | 8 |
| lsh_capacity_gold | dim_patient | 132657 |
| lsh_capacity_gold | dim_unit | 52 |
| lsh_capacity_gold | fact_barrier | 21451 |
| lsh_capacity_gold | fact_encounter | 218722 |
| lsh_capacity_gold | fact_unit_census_day | 41028 |
| lsh_capacity_gold | huddle_notes | 95 |
| lsh_capacity_gold | inpatient_predictions | 1064 |
| lsh_capacity_gold | unit_status | 52 |
| lsh_capacity_serving | __0ca310fc_2824_4302_984a_130de2b52c53_databricks_postgres_lsh_capacity_serving_bed_board_f5353589_f038_482c_bab8_d2bcd3e81d33 | not readable here: AnalysisException |
| lsh_capacity_serving | __1c8cba9d_25e4_49eb_9491_fe063969e84e_databricks_postgres_lsh_capacity_serving_unit_status_15008ced_6958_4f02_a049_0e5655f3256e | not readable here: AnalysisException |
| lsh_capacity_serving | bed_board | 1064 |
| lsh_capacity_serving | event_log_0ca310fc_2824_4302_984a_130de2b52c53 | 99 |
| lsh_capacity_serving | event_log_1c8cba9d_25e4_49eb_9491_fe063969e84e | 176 |
| lsh_capacity_serving | unit_status | 52 |
| lsh_capacity_silver | adt_events | 478357 |
| lsh_capacity_silver | adt_stays | 218722 |
| lsh_capacity_silver | barriers | 21451 |
| lsh_capacity_silver | bed_capacity | 55 |
| lsh_capacity_silver | encounters | 218722 |
| lsh_capacity_silver | ref_code_match | 68 |
| lsh_capacity_silver | ref_drg | 29 |
| lsh_capacity_silver | ref_facility | 8 |
| lsh_capacity_silver | ref_facility_match | 16 |


## Reconciliation: encounters and bed-days

Gold matches silver; the bed board, current inpatients and unit status agree on the census.

```sql
SELECT (SELECT count(*) FROM schauhan_workspace_catalog.lsh_capacity_gold.fact_encounter) AS gold_encounters,
         (SELECT count(DISTINCT csn) FROM schauhan_workspace_catalog.lsh_capacity_silver.encounters) AS silver_encounters,
         (SELECT count(*) FROM schauhan_workspace_catalog.lsh_capacity_gold.current_inpatients) AS current_inpatients,
         (SELECT count(*) FROM schauhan_workspace_catalog.lsh_capacity_gold.bed_board) AS bed_board_rows,
         (SELECT sum(patients_now) FROM schauhan_workspace_catalog.lsh_capacity_gold.unit_status) AS unit_status_patients
```

| gold_encounters | silver_encounters | current_inpatients | bed_board_rows | unit_status_patients |
| --- | --- | --- | --- | --- |
| 218722 | 218722 | 1064 | 1064 | 1064 |


## FY2026 KPIs by hospital (metric view `los_metrics`)



```sql
SELECT `Facility Code`, `Facility`, MEASURE(`Inpatient Discharges`) AS discharges, ROUND(MEASURE(`ALOS`), 2) AS alos,
         ROUND(MEASURE(`LOS O/E Index`), 3) AS los_oe, ROUND(MEASURE(`Excess Days`), 0) AS excess_days,
         ROUND(MEASURE(`Discharges Before Noon %`), 3) AS before_noon
  FROM schauhan_workspace_catalog.lsh_capacity_gold.los_metrics WHERE `Fiscal Year` = 'FY2026' GROUP BY ALL ORDER BY los_oe DESC
```

| Facility Code | Facility | discharges | alos | los_oe | excess_days | before_noon |
| --- | --- | --- | --- | --- | --- | --- |
| WCH | Willow Creek Hospital | 6112 | 4.37 | 1.339 | 8396.0 | 0.168 |
| LMH | Lakeshore Memorial Hospital | 29449 | 5.14 | 1.264 | 42369.0 | 0.17 |
| PCH | Prairie Central Hospital | 15779 | 4.37 | 1.237 | 18732.0 | 0.269 |
| CFH | Cornfield Hospital | 2648 | 4.54 | 1.225 | 3253.0 | 0.209 |
| ORH | Oak Ridge Hospital | 4069 | 4.54 | 1.219 | 4895.0 | 0.188 |
| FVH | Fox Valley Hospital | 5532 | 4.6 | 1.219 | 6677.0 | 0.182 |
| RBH | Riverbend Hospital | 5806 | 3.92 | 1.218 | 6036.0 | 0.178 |
| BSH | Bluffside Hospital | 7200 | 3.56 | 1.1 | 5683.0 | 0.288 |


## FY2026 avoidable days by barrier (metric view `barrier_metrics`)



```sql
SELECT `Barrier Category`, MEASURE(`Barriers`) AS barriers, MEASURE(`Avoidable Days`) AS avoidable_days
  FROM schauhan_workspace_catalog.lsh_capacity_gold.barrier_metrics WHERE `Fiscal Year` = 'FY2026' GROUP BY ALL ORDER BY avoidable_days DESC
```

| Barrier Category | barriers | avoidable_days |
| --- | --- | --- |
| SNF placement | 3673 | 14791 |
| Insurance authorization | 2114 | 4259 |
| Family / guardianship | 1204 | 2941 |
| Home health / DME | 1406 | 2111 |
| Pending test or procedure | 1323 | 1979 |
| Transportation | 1058 | 1058 |


## FY2026 occupancy by hospital (metric view `capacity_metrics`)



```sql
SELECT `Facility Code`, ROUND(MEASURE(`Occupancy %`), 3) AS occupancy FROM schauhan_workspace_catalog.lsh_capacity_gold.capacity_metrics
  WHERE `Fiscal Year` = 'FY2026' GROUP BY ALL ORDER BY occupancy DESC
```

| Facility Code | occupancy |
| --- | --- |
| LMH | 0.884 |
| PCH | 0.881 |
| WCH | 0.853 |
| FVH | 0.836 |
| ORH | 0.819 |
| RBH | 0.818 |
| BSH | 0.791 |
| CFH | 0.722 |


## Registered models (Unity Catalog) and their MLflow run metrics

| model | version | alias | status | mlflow_run_id | metrics |
| --- | --- | --- | --- | --- | --- |
| schauhan_workspace_catalog.lsh_capacity_gold.long_stay_risk_model | 1 | champion | READY | 62ebfe70d2e9482ba32a45ac9c5030d9 | {"baseline_auc_complex_flag_only": 0.5645, "test_accuracy": 0.7925, "test_brier": 0.1501, "test_f1": 0.1718, "test_log_loss": 0.4652, "test_pr_auc": 0.4097, "test_roc_auc": 0.7215} |
| schauhan_workspace_catalog.lsh_capacity_gold.los_days_model | 1 | champion | READY | 25319a2fb8a54eb78484340be19ed4e1 | {"baseline_mae_expected_los": 1.6283, "test_mae_days": 1.5304, "test_median_ae_days": 0.9761, "test_within_1_day": 0.5097} |


## Model output: risk tiers for current inpatients



```sql
SELECT risk_tier, count(*) AS patients, ROUND(avg(long_stay_risk), 3) AS avg_risk,
         ROUND(sum(p_discharge_today), 1) AS expected_discharges_today
  FROM schauhan_workspace_catalog.lsh_capacity_gold.bed_board GROUP BY 1 ORDER BY avg_risk DESC NULLS LAST
```

| risk_tier | patients | avg_risk | expected_discharges_today |
| --- | --- | --- | --- |
| High | 95 | 0.594 | 8.8 |
| Medium | 301 | 0.381 | 35.2 |
| Low | 613 | 0.181 | 171.6 |
| Observation | 55 | nan | 46.2 |


## Model output: 10 highest-risk patients (no names, MRNs or encounter IDs; with AI huddle notes)

Identity columns are left out; the huddle notes were generated by `ai_query` from de-identified facts (notebook 05).

```sql
SELECT facility_code, unit_name, bed, drg_description, ROUND(los_so_far_days, 1) AS day_of_stay, expected_los,
         ROUND(long_stay_risk, 3) AS risk, risk_tier, predicted_discharge_ts, open_barrier_category, risk_drivers, huddle_note
  FROM schauhan_workspace_catalog.lsh_capacity_gold.bed_board ORDER BY long_stay_risk DESC NULLS LAST LIMIT 10
```

| facility_code | unit_name | bed | drg_description | day_of_stay | expected_los | risk | risk_tier | predicted_discharge_ts | open_barrier_category | risk_drivers | huddle_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LMH | 14 West Oncology | 14W-11 | Acute leukemia w/o major O.R. procedure w MCC | 6.0 | 12.96 | 0.75 | High | 2026-11-12 11:19:11.825030 | None | complex discharge screen; age 75+; lives alone | This 77-year-old patient with acute leukemia has a 75% long-stay risk due to complex discharge needs and living alone, which may delay placement planning beyond the predicted November 12 discharge dat |
| LMH | 14 West Oncology | 14W-05 | Acute leukemia w/o major O.R. procedure w MCC | 10.6 | 14.37 | 0.713 | High | 2026-11-08 16:27:27.106502 | None | complex discharge screen; age 75+ | This 97-year-old patient with acute leukemia is at risk of exceeding the expected 14.37-day length of stay due to a complex discharge screen and advanced age. The case manager should initiate discharg |
| PCH | 5 North Oncology | 5N-02 | Acute leukemia w/o major O.R. procedure w MCC | 16.6 | 12.89 | 0.71 | High | 2026-11-05 05:09:49.003318 | None | complex discharge screen; age 75+; lives alone; transfer admission; 3.7 days past expected discharge | This 76-year-old patient is 3.7 days past expected discharge with a 71% long-stay risk driven by complex discharge needs, advanced age, and living alone. The case manager should complete a comprehensi |
| LMH | 14 West Oncology | 14W-19 | Acute leukemia w/o major O.R. procedure w MCC | 3.8 | 13.45 | 0.707 | High | 2026-11-13 20:26:14.917501 | None | complex discharge screen; age 75+; admitted from assisted living | This 84-year-old patient admitted from assisted living with complex discharge needs and multiple comorbidities has a 71% long-stay risk and is currently tracking below the expected length of stay benc |
| LMH | 15 West Oncology | 15W-05 | Acute leukemia w/o major O.R. procedure w MCC | 11.4 | 11.48 | 0.688 | High | 2026-11-06 06:06:10.469238 | None | complex discharge screen; Medicaid payer | This patient is at risk of staying longer than expected due to a complex discharge screen requiring additional coordination and a Medicaid payer status that may delay post-acute placement approval. Th |
| FVH | Intensive Care Unit | ICU-01 | Septicemia or severe sepsis w MV >96 hours | 13.6 | 14.97 | 0.687 | High | 2026-11-08 19:59:49.586133 | None | complex discharge screen; age 75+; admitted from skilled nursing facility | This 79-year-old patient is at risk of exceeding the expected length of stay due to a complex discharge screen combined with advanced age and admission from a skilled nursing facility. The case manage |
| LMH | 10 West Ortho/Neuro | 10W-01 | Intracranial hemorrhage or cerebral infarction w MCC | 10.5 | 5.54 | 0.685 | High | 2026-11-02 14:35:55.871065 | None | complex discharge screen; age 75+; transfer admission; 5.0 days past expected discharge | This 85-year-old patient is 5 days past the expected length of stay with a 68% long-stay risk due to complex discharge requirements and transfer admission status. The case manager should complete a co |
| LMH | Neuro ICU | NSICU-06 | Craniotomy w major device implant or acute complex CNS PDX w MCC | 8.9 | 10.95 | 0.682 | High | 2026-11-07 09:35:47.862088 | None | complex discharge screen; age 75+; admitted from skilled nursing facility; Medicaid payer | This 97-year-old patient has a 68% long-stay risk due to complex discharge needs, advanced age, Medicaid coverage, and admission from skilled nursing facility. The case manager should complete the com |
| LMH | 14 West Oncology | 14W-18 | Chemotherapy w/o acute leukemia as secondary diagnosis w MCC | 4.2 | 6.09 | 0.677 | High | 2026-11-05 01:26:47.427532 | None | complex discharge screen; age 75+; Medicare Advantage payer; transfer admission | This 86-year-old transfer admission with complex discharge needs and Medicare Advantage coverage presents a 68% long-stay risk that may extend beyond the predicted November 5 discharge. The case manag |
| LMH | 10 West Ortho/Neuro | 10W-08 | Intracranial hemorrhage or cerebral infarction w MCC | 5.7 | 5.37 | 0.675 | High | 2026-11-03 04:17:34.942132 | None | complex discharge screen; age 75+; admitted from assisted living; Medicaid payer; 0.4 days past expected discharge | This 79-year-old patient with intracranial hemorrhage is at risk of extended length of stay due to complex discharge planning needs, Medicaid payer status, and admission from assisted living. The case |


## Lakebase synced tables

| synced table | state | message |
| --- | --- | --- |
| lsh_capacity_serving.bed_board | SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE | Synced table creation succeeded using Delta Live Tables: https://fevm-schauhan-workspace.cloud.databricks.com#joblist/pipelines/0ca310fc-2824-4302-984 |
| lsh_capacity_serving.unit_status | SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE | Synced table creation succeeded using Delta Live Tables: https://fevm-schauhan-workspace.cloud.databricks.com#joblist/pipelines/1c8cba9d-25e4-49eb-949 |


## Genie space answers (Conversation API)

| question | answer | first result rows | check |
| --- | --- | --- | --- |
| Which hospital had the highest LOS O/E index in FY2026? |  [SQL: WITH hospital_los AS (   SELECT     `Facility`,     MEASURE(`LOS O/E Index`) AS `los_oe_index`   FROM `schauhan_workspace_catalog`.`lsh_capacity_gold`.`los_metrics`   WHERE `Fiscal Year` = 'FY2 | [["Willow Creek Hospital", "1.339034935201206"]] | matches SQL |
| How many high-risk patients are in each hospital right now? Use the bed board. |  [SQL: SELECT   `facility_name` AS `hospital`,   COUNT(*) AS `high_risk_patients` FROM `schauhan_workspace_catalog`.`lsh_capacity_gold`.`bed_board_deid` WHERE `risk_tier` = 'High'   AND `facility_name | [["Lakeshore Memorial Hospital", "60"], ["Prairie Central Hospital", "17"], ["Willow Creek Hospital", "6"]] | answered |
