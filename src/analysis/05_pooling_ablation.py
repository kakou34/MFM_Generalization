"""05 - Pooling ablation: mean (used throughout) vs max vs GeM, on the 5 FMs that need pooling (3DINO has no pooling
step). Overall: mean AUCs and the cohort-AUC slope of the main LMM; per FM: mean external AUC.

Usage: python 05_pooling_ablation.py
"""
import pandas as pd

import common as C
import lmm
import plots

POOLS = {"pooling_mean": "mean", "pooling_max": "max", "pooling_gem": "GeM"}
folder = C.outdir("05_pooling_ablation")
rows, per_fm, conds = [], [], {}
for cond, name in POOLS.items():
    d = C.load_cells(cond, C.POOLED_FMS)
    rows.append({"pooling": name, "mean_cohort_auc": d["auc_cohort_mean"].mean(),
                 "mean_internal_auc": d["auc_disease_internal_mean"].mean(),
                 "mean_external_auc": d["auc_disease_mean"].mean(), **(r := lmm.main_lmm(d))})
    conds[C.COND_LABELS[cond]] = (d, r["estimate"])  # legend labels: "Original (mean pooling)", "Max pooling", ...
    g = d.groupby("model")["auc_disease_mean"].mean().rename(name)
    per_fm.append(g)
overall = pd.DataFrame(rows)
by_fm = pd.concat(per_fm, axis=1).loc[C.POOLED_FMS].rename(index=C.FM_LABELS).rename_axis("FM").reset_index()
C.save_table(overall, folder, "pooling_ablation")
C.save_table(by_fm, folder, "pooling_ablation_external_auc_by_fm")
plots.condition_figure(conds, folder / "pooling_ablation")

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print(overall.round(4).to_string(index=False))
    print(by_fm.round(3).to_string(index=False))
