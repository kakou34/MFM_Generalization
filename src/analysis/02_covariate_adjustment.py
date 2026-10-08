"""02 - Does the cohort-AUC slope survive adding demographic (JSD sex, W1 age) and imaging (JSD field strength,
JSD manufacturer) distances?  Same LMM as 01 (all cohorts) with covariates z-scored.

Usage: python 02_covariate_adjustment.py
"""
import pandas as pd

import common as C
import lmm

DEMO = ["jsd_sex_mean", "wasserstein_age_mean"]
IMG = ["jsd_field_strength_mean", "jsd_manufacturer_mean"]
SPECS = {"cohort AUC only": [], "+ demographics (sex, age)": DEMO, "+ imaging (field strength, manufacturer)": IMG,
         "+ demographics + imaging": DEMO + IMG}

folder = C.outdir("02_covariate_adjustment")
d = C.load_cells("pooling_mean")
for c in DEMO + IMG:
    d[c] = (d[c] - d[c].mean()) / d[c].std(ddof=1)

rows, full = [], []
for name, covs in SPECS.items():
    res = lmm.fit_lmm(d, " + ".join(["auc_disease_mean ~ auc_cohort_mean"] + covs) + f" + {lmm.RE}")
    rows.append({"model": name, **lmm.term_row(res, "auc_cohort_mean")})
    full.append(res["fixed"].assign(model=name))
table = pd.DataFrame(rows)
C.save_table(table, folder, "covariate_adjustment")
C.save_table(pd.concat(full), folder, "covariate_adjustment_all_coefficients")

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
