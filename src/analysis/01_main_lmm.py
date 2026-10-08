"""01 - Main result: LMM  auc_disease ~ auc_cohort + (1|FM) + (1|train) + (1|test)   (mean pooling, 6 FMs)
on all cohorts and with Nigerian Brain excluded (sanity check: the slope stays negative and significant),
plus the scatter figure of the first submission.

Usage: python 01_main_lmm.py
"""
import pandas as pd

import common as C
import lmm
import plots

folder = C.outdir("01_main_lmm")
d = C.load_cells("pooling_mean")

rows = []
for subset, dd in [("all cohorts", d), ("without Nigerian Brain", C.drop_cohort(d))]:
    rows.append({"subset": subset, **lmm.main_lmm(dd)})
table = pd.DataFrame(rows)
C.save_table(table, folder, "main_lmm")
C.save_table(d, folder, "cell_means")
fixed = lmm.fit_lmm(d, f"auc_disease_mean ~ auc_cohort_mean + {lmm.RE}")["fixed"].set_index("term")["estimate"]
plots.pair_scatter(d, folder / "scatter_auc_disease_vs_auc_cohort",
                   line=(fixed["(Intercept)"], fixed["auc_cohort_mean"]))  # dashed line = LMM fixed effect
main = table.iloc[0]
plots.pair_scatter_abstract(d, folder / "scatter_auc_disease_vs_auc_cohort_abstract",
                            annotation=f"LMM slope = {main['estimate']:.2f}\n"
                                       + ("p < 0.001" if main["p"] < 0.001 else f"p = {main['p']:.3f}"))

# internal (adaptation cohort) vs external (unseen cohorts) AD-vs-CN AUC per FM, averaged over adaptation cohorts first
per_src = d.groupby(["model", "train"]).agg(internal=("auc_disease_internal_mean", "first"),
                                            external=("auc_disease_mean", "mean"))
drop = per_src.groupby("model").mean().reindex([m for m in C.FMS if m in set(d["model"])]).reset_index()
drop["drop"] = drop["external"] - drop["internal"]
C.save_table(drop, folder, "internal_vs_external_auc")
plots.internal_external(d, folder / "internal_vs_external_auc")

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
