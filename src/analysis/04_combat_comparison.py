"""04 - Original vs ComBat(sex) vs ComBat(sex+age) features (mean pooling): mean cohort / external AUC and the
cohort-AUC slope of the main LMM for each condition.

Usage: python 04_combat_comparison.py
"""
import pandas as pd

import common as C
import lmm
import plots

folder = C.outdir("04_combat_comparison")
rows, conds = [], {}
for cond in ["pooling_mean", "combat_sex", "combat_sex_age"]:
    d = C.load_cells(cond)
    rows.append({"features": C.COND_LABELS[cond], "mean_cohort_auc": d["auc_cohort_mean"].mean(),
                 "mean_internal_auc": d["auc_disease_internal_mean"].mean(),
                 "mean_external_auc": d["auc_disease_mean"].mean(), **(r := lmm.main_lmm(d))})
    conds[C.COND_LABELS[cond]] = (d, r["estimate"])
table = pd.DataFrame(rows)
C.save_table(table, folder, "combat_comparison")
plots.condition_figure(conds, folder / "combat_comparison")

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print(table.round(4).to_string(index=False))
