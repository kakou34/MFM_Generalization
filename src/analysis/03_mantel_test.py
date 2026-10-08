"""03 - Mantel (cohort-relabelling) permutation test of the main result, on the FM-averaged correlation.
Follows permutation_test_steps.ipynb (steps 1-5):

1. average the 20 splits -> per FM a 6x6 separability table X (cohort-probe AUC) and a 6x6 external-AUC table Y
2. rho(m, s) = Spearman between row s of X and row s of Y over the 5 targets of source s
3. statistic: rho_bar = mean of the 36 rho(m, s)   (6 FMs x 6 sources)
4. null: relabel the 6 cohorts with a permutation pi on rows AND columns of X (same pi for every FM), Y untouched
5. all 6! = 720 permutations -> exact p-value (the identity is one of them, so the smallest possible p is 1/720)

Usage: python 03_mantel_test.py
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import common as C
import plots

folder = C.outdir("03_mantel_test")
cells = C.load_cells("pooling_mean")
K = len(C.COHORTS)
OFF = np.array([[j for j in range(K) if j != i] for i in range(K)])  # off-diagonal columns of every row


def table(col):
    """(FM, source, target) array of the cell means, diagonal NaN."""
    out = []
    for m in C.FMS:
        g = cells[cells["model"] == m].pivot(index="train", columns="test", values=col)
        out.append(g.reindex(index=C.COHORTS, columns=C.COHORTS).to_numpy())
    return np.stack(out)


def row_spearman(A, B):
    """Spearman across the 5 off-diagonal entries of every (FM, row): (F, K, K) x (F, K, K) -> (F, K)."""
    r = np.arange(K)[:, None]
    a, b = rankdata(A[:, r, OFF], axis=-1), rankdata(B[:, r, OFF], axis=-1)
    a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1))


X, Y = table("auc_cohort_mean"), table("auc_disease_mean")
rho_ms = row_spearman(X, Y)
rho_bar = rho_ms.mean()

perms = np.array(list(itertools.permutations(range(K))))  # 720 x 6, identity first
null = np.array([row_spearman(X[:, p][:, :, p], Y).mean() for p in perms])
eps = 1e-12
n_extreme = int((np.abs(null) >= abs(rho_bar) - eps).sum())
res = pd.DataFrame([{"rho_bar": rho_bar, "n_permutations": len(perms), "n_as_extreme_two_sided": n_extreme,
                     "p_two_sided": n_extreme / len(perms), "p_one_sided_negative": float((null <= rho_bar + eps).mean()),
                     "n_rho_negative": int((rho_ms < 0).sum()), "n_rho_total": rho_ms.size}])
C.save_table(res, folder, "mantel_test")
C.save_table(pd.DataFrame(rho_ms, index=[C.FM_LABELS[m] for m in C.FMS], columns=C.COHORTS)
             .rename_axis("FM").reset_index(), folder, "rho_per_fm_and_source")

plots.mantel_null(null, rho_bar, folder / "mantel_null")

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
