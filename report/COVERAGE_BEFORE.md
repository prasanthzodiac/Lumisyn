# Baseline Test Coverage Report (Before Fixes)

**Timestamp**: 2026-09-20T21:16:10+05:30
**Overall Project Coverage**: 46% (6697 statements, 3637 missed)

## Release-Path Modules with Zero or Weak Coverage (< 50%)

| Module | Statements | Missed | Coverage | Stage Association |
| :--- | :---: | :---: | :---: | :--- |
| `lumisyn\backend\app_backend.py` | 334 | 226 | **32%** | Platform API / Service |
| `lumisyn\backend\benchmark_1000_variations.py` | 126 | 126 | **0%** | Platform API / Service |
| `lumisyn\backend\compare_original_vs_synthetic.py` | 73 | 73 | **0%** | Platform API / Service |
| `lumisyn\backend\download_and_test_real_nhanes.py` | 130 | 130 | **0%** | Platform API / Service |
| `lumisyn\backend\full_audit_acfmv5.py` | 171 | 171 | **0%** | Platform API / Service |
| `lumisyn\backend\ollama_evaluator.py` | 49 | 49 | **0%** | Platform API / Service |
| `lumisyn\backend\run_adversarial_audit.py` | 98 | 98 | **0%** | Platform API / Service |
| `lumisyn\backend\run_hard_stress_test.py` | 102 | 102 | **0%** | Platform API / Service |
| `lumisyn\backend\run_kaggle_benchmark.py` | 71 | 71 | **0%** | Platform API / Service |
| `lumisyn\backend\run_multi_dataset_benchmark.py` | 147 | 147 | **0%** | Platform API / Service |
| `lumisyn\backend\run_nhanes_benchmark.py` | 106 | 106 | **0%** | Platform API / Service |
| `lumisyn\backend\test_local_nhanes.py` | 157 | 157 | **0%** | Platform API / Service |
| `lumisyn\backend\tests\test_backend_api.py` | 61 | 51 | **16%** | Platform API / Service |
| `lumisyn\backend\tests\test_full_platform_e2e.py` | 52 | 42 | **19%** | Platform API / Service |
| `lumisyn\framework\acfm_net\evaluation\__init__.py` | 43 | 27 | **37%** | Stage 9 (Fidelity / Utility) |
| `lumisyn\framework\acfm_net\evaluation\clinical.py` | 99 | 88 | **11%** | Stage 9 (Fidelity / Utility) |
| `lumisyn\framework\acfm_net\evaluation\composite_score.py` | 41 | 39 | **5%** | Stage 9 (Fidelity / Utility) |
| `lumisyn\framework\acfm_net\evaluation\fairness.py` | 46 | 35 | **24%** | Stage 9 (Fidelity / Utility) |
| `lumisyn\framework\acfm_net\evaluation\fidelity.py` | 313 | 208 | **34%** | Stage 9 (Fidelity / Utility) |
| `lumisyn\framework\acfm_net\evaluation\floor_baselines.py` | 73 | 73 | **0%** | Stage 9 (Fidelity / Utility) |
| `lumisyn\framework\acfm_net\evaluation\harness.py` | 81 | 74 | **9%** | Stage 9 (Fidelity / Utility) |
| `lumisyn\framework\acfm_net\evaluation\module_proof.py` | 13 | 10 | **23%** | Stage 9 (Fidelity / Utility) |
| `lumisyn\framework\acfm_net\evaluation\privacy.py` | 121 | 105 | **13%** | Stage 9 (Fidelity / Utility) |
| `lumisyn\framework\acfm_net\evaluation\privacy_eval.py` | 166 | 113 | **32%** | Stage 9 (Fidelity / Utility) |
| `lumisyn\framework\acfm_net\evaluation\report.py` | 107 | 93 | **13%** | Stage 9 (Fidelity / Utility) |
| `lumisyn\framework\acfm_net\evaluation\statistical.py` | 103 | 87 | **16%** | Stage 9 (Fidelity / Utility) |
| `lumisyn\framework\acfm_net\evaluation\utility.py` | 105 | 72 | **31%** | Stage 9 (Fidelity / Utility) |
| `lumisyn\framework\acfm_net\generative\discrete_flow.py` | 33 | 24 | **27%** | Stage 6 & 8 (Flow) |
| `lumisyn\framework\acfm_net\generative\gated_fusion.py` | 23 | 17 | **26%** | Stage 6 & 8 (Flow) |
| `lumisyn\framework\acfm_net\graph\refinement.py` | 33 | 25 | **24%** | Stage 3 (DCGB) |
| `lumisyn\framework\acfm_net\intelligence\generator_1000.py` | 71 | 71 | **0%** | Stage 1 (UDIE) |
| `lumisyn\framework\acfm_net\intelligence\kaggle_loader.py` | 35 | 35 | **0%** | Stage 1 (UDIE) |
| `lumisyn\framework\acfm_net\optimization\curriculum.py` | 67 | 53 | **21%** | Other |
| `lumisyn\framework\acfm_net\optimization\gradnorm.py` | 16 | 9 | **44%** | Other |
| `lumisyn\framework\acfm_net\optimization\model_selection.py` | 18 | 10 | **44%** | Other |
| `lumisyn\framework\acfm_net\optimization\pcgrad.py` | 48 | 41 | **15%** | Other |
| `lumisyn\framework\acfm_net\preparation\balancing.py` | 24 | 20 | **17%** | Stage 2 (ADPE) |
| `lumisyn\framework\acfm_net\repair\constraint_repair.py` | 155 | 100 | **35%** | Stage 7 (Constraint Repair) |
| `lumisyn\framework\acfm_net\representation\contrastive.py` | 26 | 20 | **23%** | Stage 4 (HCARLN) |
| `lumisyn\framework\acfm_net\representation\tabular_vae.py` | 78 | 66 | **15%** | Stage 4 (HCARLN) |

## Complete Module Coverage Table

| Module | Stmts | Miss | Cover |
| :--- | :---: | :---: | :---: |
| `lumisyn\backend\__init__.py` | 0 | 0 | 100% |
| `lumisyn\backend\app_backend.py` | 334 | 226 | 32% |
| `lumisyn\backend\benchmark_1000_variations.py` | 126 | 126 | 0% |
| `lumisyn\backend\compare_original_vs_synthetic.py` | 73 | 73 | 0% |
| `lumisyn\backend\db_store.py` | 48 | 20 | 58% |
| `lumisyn\backend\download_and_test_real_nhanes.py` | 130 | 130 | 0% |
| `lumisyn\backend\full_audit_acfmv5.py` | 171 | 171 | 0% |
| `lumisyn\backend\ollama_evaluator.py` | 49 | 49 | 0% |
| `lumisyn\backend\run_adversarial_audit.py` | 98 | 98 | 0% |
| `lumisyn\backend\run_hard_stress_test.py` | 102 | 102 | 0% |
| `lumisyn\backend\run_kaggle_benchmark.py` | 71 | 71 | 0% |
| `lumisyn\backend\run_multi_dataset_benchmark.py` | 147 | 147 | 0% |
| `lumisyn\backend\run_nhanes_benchmark.py` | 106 | 106 | 0% |
| `lumisyn\backend\test_local_nhanes.py` | 157 | 157 | 0% |
| `lumisyn\backend\tests\test_adpe.py` | 13 | 0 | 100% |
| `lumisyn\backend\tests\test_apc.py` | 16 | 0 | 100% |
| `lumisyn\backend\tests\test_api_contract.py` | 15 | 0 | 100% |
| `lumisyn\backend\tests\test_backend_api.py` | 61 | 51 | 16% |
| `lumisyn\backend\tests\test_cdgl.py` | 18 | 0 | 100% |
| `lumisyn\backend\tests\test_dp_copula_accounting.py` | 45 | 0 | 100% |
| `lumisyn\backend\tests\test_dp_recalibration_accounting.py` | 76 | 0 | 100% |
| `lumisyn\backend\tests\test_dpsgd_chunked.py` | 36 | 0 | 100% |
| `lumisyn\backend\tests\test_full_platform_e2e.py` | 52 | 42 | 19% |
| `lumisyn\backend\tests\test_gate4b_metrics.py` | 41 | 0 | 100% |
| `lumisyn\backend\tests\test_noisy_copy_controls.py` | 22 | 0 | 100% |
| `lumisyn\backend\tests\test_per_group_sensitivity.py` | 42 | 0 | 100% |
| `lumisyn\backend\tests\test_phase1_privacy.py` | 319 | 1 | 99% |
| `lumisyn\backend\tests\test_pipeline.py` | 20 | 8 | 60% |
| `lumisyn\backend\tests\test_reflow_isolation.py` | 51 | 22 | 57% |
| `lumisyn\backend\tests\test_schema_invariance.py` | 33 | 0 | 100% |
| `lumisyn\backend\tests\test_udie.py` | 22 | 0 | 100% |
| `lumisyn\framework\acfm_net\__init__.py` | 1 | 0 | 100% |
| `lumisyn\framework\acfm_net\evaluation\__init__.py` | 43 | 27 | 37% |
| `lumisyn\framework\acfm_net\evaluation\clinical.py` | 99 | 88 | 11% |
| `lumisyn\framework\acfm_net\evaluation\composite_score.py` | 41 | 39 | 5% |
| `lumisyn\framework\acfm_net\evaluation\dp_gaussian_copula.py` | 113 | 3 | 97% |
| `lumisyn\framework\acfm_net\evaluation\fairness.py` | 46 | 35 | 24% |
| `lumisyn\framework\acfm_net\evaluation\fidelity.py` | 313 | 208 | 34% |
| `lumisyn\framework\acfm_net\evaluation\floor_baselines.py` | 73 | 73 | 0% |
| `lumisyn\framework\acfm_net\evaluation\harness.py` | 81 | 74 | 9% |
| `lumisyn\framework\acfm_net\evaluation\module_proof.py` | 13 | 10 | 23% |
| `lumisyn\framework\acfm_net\evaluation\privacy.py` | 121 | 105 | 13% |
| `lumisyn\framework\acfm_net\evaluation\privacy_eval.py` | 166 | 113 | 32% |
| `lumisyn\framework\acfm_net\evaluation\report.py` | 107 | 93 | 13% |
| `lumisyn\framework\acfm_net\evaluation\statistical.py` | 103 | 87 | 16% |
| `lumisyn\framework\acfm_net\evaluation\utility.py` | 105 | 72 | 31% |
| `lumisyn\framework\acfm_net\generative\__init__.py` | 4 | 0 | 100% |
| `lumisyn\framework\acfm_net\generative\discrete_flow.py` | 33 | 24 | 27% |
| `lumisyn\framework\acfm_net\generative\gated_fusion.py` | 23 | 17 | 26% |
| `lumisyn\framework\acfm_net\generative\ot_cfm.py` | 106 | 25 | 76% |
| `lumisyn\framework\acfm_net\governance\__init__.py` | 2 | 0 | 100% |
| `lumisyn\framework\acfm_net\governance\cdgl.py` | 134 | 22 | 84% |
| `lumisyn\framework\acfm_net\graph\__init__.py` | 3 | 0 | 100% |
| `lumisyn\framework\acfm_net\graph\builder.py` | 93 | 36 | 61% |
| `lumisyn\framework\acfm_net\graph\refinement.py` | 33 | 25 | 24% |
| `lumisyn\framework\acfm_net\intelligence\__init__.py` | 6 | 0 | 100% |
| `lumisyn\framework\acfm_net\intelligence\dataset2vec.py` | 65 | 18 | 72% |
| `lumisyn\framework\acfm_net\intelligence\dependencies.py` | 52 | 13 | 75% |
| `lumisyn\framework\acfm_net\intelligence\generator_1000.py` | 71 | 71 | 0% |
| `lumisyn\framework\acfm_net\intelligence\kaggle_loader.py` | 35 | 35 | 0% |
| `lumisyn\framework\acfm_net\intelligence\missingness.py` | 53 | 17 | 68% |
| `lumisyn\framework\acfm_net\intelligence\outliers.py` | 55 | 7 | 87% |
| `lumisyn\framework\acfm_net\intelligence\sensitive.py` | 39 | 5 | 87% |
| `lumisyn\framework\acfm_net\optimization\__init__.py` | 5 | 0 | 100% |
| `lumisyn\framework\acfm_net\optimization\curriculum.py` | 67 | 53 | 21% |
| `lumisyn\framework\acfm_net\optimization\gradnorm.py` | 16 | 9 | 44% |
| `lumisyn\framework\acfm_net\optimization\model_selection.py` | 18 | 10 | 44% |
| `lumisyn\framework\acfm_net\optimization\pcgrad.py` | 48 | 41 | 15% |
| `lumisyn\framework\acfm_net\pipeline.py` | 560 | 158 | 72% |
| `lumisyn\framework\acfm_net\postprocessing\dp_recalibration.py` | 145 | 45 | 69% |
| `lumisyn\framework\acfm_net\preparation\__init__.py` | 5 | 0 | 100% |
| `lumisyn\framework\acfm_net\preparation\balancing.py` | 24 | 20 | 17% |
| `lumisyn\framework\acfm_net\preparation\encoding.py` | 139 | 21 | 85% |
| `lumisyn\framework\acfm_net\preparation\imputation.py` | 60 | 19 | 68% |
| `lumisyn\framework\acfm_net\preparation\scaling.py` | 70 | 30 | 57% |
| `lumisyn\framework\acfm_net\privacy\__init__.py` | 6 | 0 | 100% |
| `lumisyn\framework\acfm_net\privacy\budget_allocator.py` | 97 | 27 | 72% |
| `lumisyn\framework\acfm_net\privacy\checkpoint_audit.py` | 44 | 4 | 91% |
| `lumisyn\framework\acfm_net\privacy\contract.py` | 65 | 21 | 68% |
| `lumisyn\framework\acfm_net\privacy\dpsgd.py` | 108 | 27 | 75% |
| `lumisyn\framework\acfm_net\privacy\prv_accountant.py` | 67 | 11 | 84% |
| `lumisyn\framework\acfm_net\privacy\rdp_accountant.py` | 21 | 4 | 81% |
| `lumisyn\framework\acfm_net\repair\__init__.py` | 2 | 0 | 100% |
| `lumisyn\framework\acfm_net\repair\constraint_repair.py` | 155 | 100 | 35% |
| `lumisyn\framework\acfm_net\representation\__init__.py` | 6 | 0 | 100% |
| `lumisyn\framework\acfm_net\representation\contrastive.py` | 26 | 20 | 23% |
| `lumisyn\framework\acfm_net\representation\cross_attention.py` | 30 | 1 | 97% |
| `lumisyn\framework\acfm_net\representation\gat_encoder.py` | 34 | 2 | 94% |
| `lumisyn\framework\acfm_net\representation\hvae.py` | 75 | 6 | 92% |
| `lumisyn\framework\acfm_net\representation\tabular_vae.py` | 78 | 66 | 15% |
