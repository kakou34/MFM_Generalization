# analysis_v2 - MedIA resubmission analyses

    python run_all.py [--skip-features]     # outputs in <repo>/local/output_v2/ ; needs R (lme4, lmerTest, performance) + rpy2

All statistics use one row per (FM, source, target) cell = mean over the 20 iterations (180 rows, mean pooling).

| Script | Content |
|---|---|
| `01_main_lmm.py` | Main LMM `auc_disease ~ auc_cohort + (1\|FM)+(1\|train)+(1\|test)`, all cohorts and without Nigerian Brain; scatter figure (marker = FM, colour = adaptation cohort, shade = test cohort) |
| `02_covariate_adjustment.py` | Same LMM + demographic (JSD sex, W1 age) and imaging (JSD field strength, JSD manufacturer) distances |
| `03_mantel_test.py` | Exact Mantel test (720 cohort relabellings) on the FM-averaged within-source Spearman correlation, as in permutation_test_steps.ipynb |
| `04_combat_comparison.py` | Original vs ComBat(sex) vs ComBat(sex+age): mean AUCs and main-LMM slope |
| `05_pooling_ablation.py` | Mean vs max vs GeM (5 FMs; 3DINO has no pooling): mean AUCs and main-LMM slope, external AUC per FM |
| `06_feature_space_figures.py` | t-SNE grids and 2-Wasserstein heatmaps (+ cohort-vs-disease distance ratio) for the 6 FMs |

