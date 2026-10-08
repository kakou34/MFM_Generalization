"""Feature-distribution shift and cohort-metadata divergence per cohort pair.

For every ``(train cohort, test cohort, split)`` row of ``classify_disease.py``
this rebuilds the exact cognitively-normal subsets its paired cohort probe was
trained on -- up to ``--cohort_n`` controls from the split's training fold and
up to ``--cohort_n`` from the full test cohort, drawn with the same seed -- and
measures how far apart the two feature distributions are. Rows share the keys
``train, test, split`` with the disease results, so the two tables can be
joined 1:1. No probe is trained; the script is CPU-only.

Feature-space measures (directed ones are ``train -> test`` as ``_tr_te`` and
the reverse as ``_te_tr``):

- ``wasserstein_dist``  exact optimal-transport cost with squared-Euclidean
  ground cost (W2^2), the same quantity ``classify_disease.py`` computes on
  all controls.
- ``sliced_wasserstein``  sliced W2 over random 1-D projections
  (Bonneel et al., 2015).
- ``mmd2`` / ``mmd_pvalue``  unbiased squared MMD with an RBF kernel at the
  median-heuristic bandwidth, and its permutation p-value (Gretton et al.,
  2012; Rabanser et al., 2019).
- ``energy_dist``  energy distance (Szekely & Rizzo, 2013).
- ``frechet_dist``  Frechet distance between Gaussian fits, as in FID
  (Heusel et al., 2017).
- ``kl_gauss_*``  closed-form KL between Gaussian fits.
- ``kl_knn_*``  k-nearest-neighbour KL estimator (Wang, Kulkarni & Verdu,
  2009). It is badly biased when the dimension approaches the sample size, so
  it is computed in a PCA space of ``--knn_pca_dim`` components fitted on the
  pooled pair.
- ``coral_dist``  squared Frobenius distance between covariances over 4 d^2,
  the CORAL loss (Sun & Saenko, 2016).
- ``ks_mean`` / ``ks_frac_shifted`` / ``ks_bonferroni_p``  per-feature
  two-sample Kolmogorov-Smirnov tests: mean statistic, fraction of features
  shifted after Benjamini-Hochberg at 0.05, and the Bonferroni-aggregated
  p-value (Rabanser et al., 2019).

The measures above grow with the feature scale and, since they sum over
features, with the feature dimension, so they cannot be compared across FMs.
Z-scoring fixes the scale but not the dimension. The normalised measures below
are comparable across FMs, except the ``_z`` columns, as noted:

- ``mean_shift_rel``  squared distance between the cohort means, debiased
  for sampling noise, over the summed per-feature variance of both sides.
  Numerator and denominator both sum over features, so it is a
  variance-weighted average of per-feature mean shift: invariant to rescaling
  and to the number of features, and ~0 when there is no shift.
- ``wasserstein_rel``  ``wasserstein_dist`` minus the mean of its permutation
  null (``--n_ot_permutations``), over the same summed variance. Without the
  subtraction the ratio sits near 0.7 even for identical distributions, a
  floor set by dimension over sample size. Fewer permutations than the MMD
  test, since each one is an exact OT solve.
- ``energy_coef``  energy distance over 2E|A-B|, bounded in [0, 1].
- ``mmd_z`` / ``energy_z``  each statistic standardised against its
  permutation null (same permutations as ``mmd_pvalue``). Scale-free, but
  these measure how detectable the shift is, not how large: for the same
  per-feature shift they grow with the number of independently shifted
  features (roughly 3x from 64 to 1024 dimensions on Gaussian data), and with
  the sample size. Use them within an FM, not across FMs.
- ``mmd2`` itself is bounded by the median-heuristic kernel, and the KS
  summaries average over features.

Remaining differences in intrinsic dimension and feature correlation still
shape these values, so cross-FM gaps partly reflect feature geometry. There is
no normalised Frechet distance: its covariance term depends on the Ledoit-Wolf
shrinkage, which itself depends on the dimension, so dividing by the variance
does not make it dimension-free.

Gaussian fits use Ledoit-Wolf shrinkage, since with at most 150 scans per side
the sample covariance of a 256-1024-d feature vector is singular.

Metadata measures (identical for every FM, since all feature CSVs hold the same
scans in the same order) are Jensen-Shannon divergences, base 2 as in
``src/analysis/compute_img_diff.py``, between the same two control subsets:
scanner manufacturer, field strength, sex, and age at scan in 5-year bins,
plus the 1-D Wasserstein distance of age at scan in years. Missing values are
dropped per variable. ``jsd_label`` is the exception: the subsets are
controls only, so it compares the AD/CN mix of the whole training fold with
that of the whole test cohort, i.e. the label shift the disease probe faces.
"""

from pathlib import Path

import numpy as np
import ot
import pandas as pd
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import false_discovery_control, ks_2samp, wasserstein_distance
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


CATEGORICAL_META = {
    'jsd_manufacturer': 'scanner_manufacturer',
    'jsd_field_strength': 'field_strength_tesla',
    'jsd_sex': 'sex',
}
# Age at the scan itself; the plain ``age`` column is baseline/entry age for
# ADNI and OASIS-3 and is missing altogether for OASIS-4.
AGE_COL = 'age_at_scan'
AGE_BIN_WIDTH = 5


# ---- Feature-space measures ----------------------------------------------

def wasserstein_ot(X_a, X_b, rng, n_permutations=20):
    """Exact OT cost (squared-Euclidean ground cost, uniform weights) and the
    mean of its permutation null.

    Between two finite samples the cost is mostly sampling noise, whose size
    depends on the dimension relative to the sample size; the null mean
    estimates that floor. A few permutations suffice for a mean.
    """
    X = np.concatenate([X_a, X_b])
    n_a = len(X_a)
    M = ot.dist(X, X, metric='sqeuclidean')
    a = np.full(n_a, 1 / n_a)
    b = np.full(len(X) - n_a, 1 / (len(X) - n_a))

    def cost(order):
        return ot.emd2(a, b, M[np.ix_(order[:n_a], order[n_a:])])

    observed = cost(np.arange(len(X)))
    if n_permutations == 0:
        return observed, np.nan
    null_mean = np.mean([cost(rng.permutation(len(X))) for _ in range(n_permutations)])
    return observed, null_mean


def pair_means(M, S, n_a, n_b):
    """Mean within-A, within-B and between entries of a zero-diagonal matrix.

    Each row of ``S`` is a 0/1 side-A indicator over the pooled samples, so a
    whole batch of permuted labellings comes out of one matrix product. The
    within-side means skip the diagonal (U-statistics).
    """
    T = 1 - S
    MS = S @ M
    within_a = np.einsum('ij,ij->i', MS, S) / (n_a * (n_a - 1))
    within_b = np.einsum('ij,ij->i', T @ M, T) / (n_b * (n_b - 1))
    between = np.einsum('ij,ij->i', MS, T) / (n_a * n_b)
    return within_a, within_b, between


def permutation_tests(X_a, X_b, rng, n_permutations=200):
    """MMD and energy distance, each with a permutation null.

    Squared MMD uses an RBF kernel at the median-heuristic bandwidth. Energy
    distance is 2E|A-B| - E|A-A'| - E|B-B'|; ``energy_coef`` divides it by
    2E|A-B|, which bounds it in [0, 1] (Szekely & Rizzo, 2013). The ``_z``
    columns standardise each statistic against its own permutation null; they
    are scale-free but still grow with dimension and sample size (see the
    module docstring).
    """
    X = np.concatenate([X_a, X_b])
    n_a, n_b = len(X_a), len(X_b)
    D2 = cdist(X, X, 'sqeuclidean')
    bandwidth = np.median(D2[np.triu_indices_from(D2, k=1)])
    K = np.exp(-D2 / bandwidth)
    np.fill_diagonal(K, 0.0)
    D = np.sqrt(D2)

    def stats(S):
        k_aa, k_bb, k_ab = pair_means(K, S, n_a, n_b)
        d_aa, d_bb, d_ab = pair_means(D, S, n_a, n_b)
        return k_aa + k_bb - 2 * k_ab, 2 * d_ab - d_aa - d_bb, d_ab

    is_a = (np.arange(len(X)) < n_a).astype(np.float64)
    (mmd2,), (energy,), (d_ab,) = stats(is_a[None])
    out = {
        'mmd2': mmd2,
        'energy_dist': energy,
        'energy_coef': energy / (2 * d_ab),
    }
    if n_permutations == 0:
        out.update(mmd_pvalue=np.nan, mmd_z=np.nan, energy_z=np.nan)
        return out

    null_mmd, null_energy, _ = stats(
        np.stack([rng.permutation(is_a) for _ in range(n_permutations)])
    )
    out['mmd_pvalue'] = (1 + np.sum(null_mmd >= mmd2)) / (1 + n_permutations)
    out['mmd_z'] = (mmd2 - null_mmd.mean()) / null_mmd.std(ddof=1)
    out['energy_z'] = (energy - null_energy.mean()) / null_energy.std(ddof=1)
    return out


def gaussian_fit(X):
    """Mean and Ledoit-Wolf shrunk covariance."""
    return X.mean(axis=0), LedoitWolf().fit(X).covariance_


def frechet_distance(mu_a, cov_a, mu_b, cov_b):
    """||mu_a - mu_b||^2 + Tr(C_a + C_b - 2 (C_a C_b)^(1/2))."""
    # Tr((C_a C_b)^(1/2)) = sum of sqrt eigenvalues of C_a^(1/2) C_b C_a^(1/2),
    # which is symmetric, so it can go through eigh instead of sqrtm.
    w, V = np.linalg.eigh(cov_a)
    sqrt_a = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    cross = np.sqrt(np.clip(np.linalg.eigvalsh(sqrt_a @ cov_b @ sqrt_a), 0, None)).sum()
    return np.sum((mu_a - mu_b) ** 2) + np.trace(cov_a) + np.trace(cov_b) - 2 * cross


def kl_gaussian(mu_p, cov_p, mu_q, cov_q):
    """KL(N(mu_p, cov_p) || N(mu_q, cov_q)) in nats."""
    d = len(mu_p)
    diff = mu_q - mu_p
    trace_term = np.trace(np.linalg.solve(cov_q, cov_p))
    maha_term = diff @ np.linalg.solve(cov_q, diff)
    _, logdet_p = np.linalg.slogdet(cov_p)
    _, logdet_q = np.linalg.slogdet(cov_q)
    return 0.5 * (trace_term + maha_term - d + logdet_q - logdet_p)


def kl_knn(X_p, X_q, k=5):
    """k-NN estimate of KL(P || Q) in nats (Wang, Kulkarni & Verdu, 2009)."""
    n, d = X_p.shape
    m = len(X_q)
    k = min(k, n - 1, m)
    # k + 1 neighbours within P, since each point is its own nearest one.
    rho = NearestNeighbors(n_neighbors=k + 1).fit(X_p).kneighbors(X_p)[0][:, k]
    nu = NearestNeighbors(n_neighbors=k).fit(X_q).kneighbors(X_p)[0][:, k - 1]
    # Duplicate scans give zero distances; floor them so the log stays finite.
    eps = np.finfo(np.float64).tiny
    rho, nu = np.maximum(rho, eps), np.maximum(nu, eps)
    return d * np.mean(np.log(nu / rho)) + np.log(m / (n - 1))


def coral_distance(X_a, X_b):
    """Squared Frobenius distance between covariances over 4 d^2 (CORAL loss)."""
    d = X_a.shape[1]
    return np.sum((np.cov(X_a, rowvar=False) - np.cov(X_b, rowvar=False)) ** 2) / (4 * d ** 2)


def ks_features(X_a, X_b, alpha=0.05):
    """Per-feature KS tests: mean statistic, fraction shifted (BH), Bonferroni p."""
    res = ks_2samp(X_a, X_b, axis=0)
    # Constant features (e.g. dead channels) give NaN p-values; ignore them.
    stat, pvals = res.statistic, res.pvalue
    valid = ~np.isnan(pvals)
    stat, pvals = stat[valid], pvals[valid]
    shifted = false_discovery_control(pvals) < alpha
    return stat.mean(), shifted.mean(), min(1.0, pvals.min() * len(pvals))


def feature_shift(X_a, X_b, rng, n_permutations=200, n_ot_permutations=20,
                  n_projections=500, knn_k=5, knn_pca_dim=10, standardize=False):
    """All feature-space shift measures between two control subsets."""
    X_a = X_a.astype(np.float64)
    X_b = X_b.astype(np.float64)
    if standardize:
        # Pooled over the pair, so neither side is treated as the reference.
        scaler = StandardScaler().fit(np.concatenate([X_a, X_b]))
        X_a, X_b = scaler.transform(X_a), scaler.transform(X_b)

    mu_a, cov_a = gaussian_fit(X_a)
    mu_b, cov_b = gaussian_fit(X_b)
    ks_mean, ks_frac, ks_p = ks_features(X_a, X_b)

    n_components = min(knn_pca_dim, X_a.shape[1], len(X_a) + len(X_b) - 1)
    pca = PCA(n_components=n_components).fit(np.concatenate([X_a, X_b]))
    Z_a, Z_b = pca.transform(X_a), pca.transform(X_b)

    wasserstein, wasserstein_null = wasserstein_ot(X_a, X_b, rng, n_ot_permutations)
    # Summed per-feature variance of both sides, the normaliser for the
    # relative measures.
    var_a = X_a.var(axis=0, ddof=1).sum()
    var_b = X_b.var(axis=0, ddof=1).sum()
    total_var = var_a + var_b
    # E||mean_a - mean_b||^2 exceeds the true squared mean shift by
    # var_a / n_a + var_b / n_b; subtracting that makes it unbiased.
    mean_shift = (np.sum((mu_a - mu_b) ** 2)
                  - var_a / len(X_a) - var_b / len(X_b))

    return {
        'wasserstein_dist': wasserstein,
        'wasserstein_rel': (wasserstein - wasserstein_null) / total_var,
        'sliced_wasserstein': ot.sliced_wasserstein_distance(
            X_a, X_b, n_projections=n_projections, p=2,
            seed=int(rng.integers(2**31)),
        ),
        **permutation_tests(X_a, X_b, rng, n_permutations),
        'frechet_dist': frechet_distance(mu_a, cov_a, mu_b, cov_b),
        'mean_shift_rel': mean_shift / total_var,
        'kl_gauss_tr_te': kl_gaussian(mu_a, cov_a, mu_b, cov_b),
        'kl_gauss_te_tr': kl_gaussian(mu_b, cov_b, mu_a, cov_a),
        'kl_knn_tr_te': kl_knn(Z_a, Z_b, knn_k),
        'kl_knn_te_tr': kl_knn(Z_b, Z_a, knn_k),
        'coral_dist': coral_distance(X_a, X_b),
        'ks_mean': ks_mean,
        'ks_frac_shifted': ks_frac,
        'ks_bonferroni_p': ks_p,
    }


FEATURE_METRICS = [
    'wasserstein_dist', 'wasserstein_rel', 'sliced_wasserstein',
    'mmd2', 'energy_dist', 'energy_coef', 'mmd_pvalue', 'mmd_z', 'energy_z',
    'frechet_dist', 'mean_shift_rel', 'kl_gauss_tr_te', 'kl_gauss_te_tr',
    'kl_knn_tr_te', 'kl_knn_te_tr', 'coral_dist', 'ks_mean',
    'ks_frac_shifted', 'ks_bonferroni_p',
]


# ---- Metadata measures ---------------------------------------------------

def jsd_categorical(a, b):
    """Base-2 Jensen-Shannon divergence between two categorical samples."""
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    categories = sorted(set(a) | set(b))
    p = a.value_counts(normalize=True).reindex(categories, fill_value=0).values
    q = b.value_counts(normalize=True).reindex(categories, fill_value=0).values
    # scipy returns the JS distance, the square root of the divergence.
    return jensenshannon(p, q, base=2) ** 2


def metadata_shift(meta_a, meta_b, age_bins):
    """Metadata divergences between two control subsets (rows of the metadata)."""
    out = {
        name: jsd_categorical(meta_a[col], meta_b[col])
        for name, col in CATEGORICAL_META.items()
    }
    age_a, age_b = meta_a[AGE_COL].dropna().values, meta_b[AGE_COL].dropna().values
    if len(age_a) and len(age_b):
        out['jsd_age'] = jsd_categorical(
            pd.cut(age_a, age_bins).astype(str), pd.cut(age_b, age_bins).astype(str)
        )
        out['wasserstein_age'] = wasserstein_distance(age_a, age_b)
    else:
        out['jsd_age'] = out['wasserstein_age'] = np.nan
    return out


META_METRICS = list(CATEGORICAL_META) + ['jsd_age', 'wasserstein_age']


def load_metadata(metadata_csv, ids):
    """Metadata rows for ``ids`` in that order, with field strength rounded."""
    meta = pd.read_csv(metadata_csv, low_memory=False).set_index('id')
    missing = pd.Index(ids).difference(meta.index)
    if len(missing):
        raise ValueError(f"{len(missing)} feature ids missing from {metadata_csv}, e.g. {missing[:3].tolist()}")
    meta = meta.loc[ids]
    # 1.494 T (a few OASIS-3 scans) is a 1.5 T scanner.
    meta['field_strength_tesla'] = meta['field_strength_tesla'].round(1)
    return meta


# ---- Main loop -----------------------------------------------------------

def shift_per_pair(df, meta=None, n_splits=20, cohort_n=150, **shift_kwargs):
    # Imported here rather than at the top: both modules import torch, which
    # the metric functions above do not need, so src/analysis notebooks can
    # import those without it.
    from classify_disease import disease_splits
    from linear_probe import MIN_PER_SIDE, subsample_pair

    cohorts = df['cohort'].unique().tolist()
    FEATURE_COLS = [c for c in df.columns if c.startswith("Feature")]

    if meta is not None:
        age = meta[AGE_COL].dropna()
        lo = np.floor(age.min() / AGE_BIN_WIDTH) * AGE_BIN_WIDTH
        hi = np.ceil(age.max() / AGE_BIN_WIDTH) * AGE_BIN_WIDTH + AGE_BIN_WIDTH
        age_bins = np.arange(lo, hi, AGE_BIN_WIDTH)

    results = []

    for train_cohort in cohorts:
        print(f"\n=== Train cohort {train_cohort} ===")

        train_df = df[df['cohort'] == train_cohort]
        X_np = train_df[FEATURE_COLS].values.astype(np.float32)
        y_np = train_df['label'].values.astype(np.float32)
        ids_np = train_df['id'].values

        for split_idx, (train_idx, _val_idx) in enumerate(disease_splits(train_df, n_splits)):
            y_train_np = y_np[train_idx]
            train_label_0_mask = (y_train_np == 0)
            X_train_label0_np = X_np[train_idx][train_label_0_mask]
            ids_train_label0 = ids_np[train_idx][train_label_0_mask]

            for test_cohort in cohorts:
                test_df = df[df['cohort'] == test_cohort]
                y_test_np = test_df['label'].values.astype(np.float32)
                test_label_0_mask = (y_test_np == 0)
                X_test_label0_np = test_df[FEATURE_COLS].values.astype(np.float32)[test_label_0_mask]
                ids_test_label0 = test_df['id'].values[test_label_0_mask]

                row = {'train': train_cohort, 'test': test_cohort, 'split': split_idx}
                n_a = min(cohort_n, len(X_train_label0_np))
                n_b = min(cohort_n, len(X_test_label0_np))
                row['n_cohort_train'], row['n_cohort_test'] = n_a, n_b

                # Same skip rules as the paired cohort probe: nothing to compare
                # within one cohort, and too few scans to say anything.
                if test_cohort == train_cohort or n_a < MIN_PER_SIDE or n_b < MIN_PER_SIDE:
                    row.update({m: np.nan for m in FEATURE_METRICS})
                    if meta is not None:
                        row.update({m: np.nan for m in META_METRICS})
                        row['jsd_label'] = np.nan
                    results.append(row)
                    continue

                # Same generator and draw order as cohort_separability, so these
                # are exactly the controls the paired cohort probe saw.
                rng = np.random.default_rng(split_idx)
                idx_a, idx_b = subsample_pair(
                    len(X_train_label0_np), len(X_test_label0_np), cohort_n, rng
                )
                # A separate stream for the permutation test and projections,
                # so they cannot disturb the subsample.
                metric_rng = np.random.default_rng([split_idx, 1])
                row.update(feature_shift(
                    X_train_label0_np[idx_a], X_test_label0_np[idx_b],
                    metric_rng, **shift_kwargs,
                ))

                if meta is not None:
                    row.update(metadata_shift(
                        meta.loc[ids_train_label0[idx_a]],
                        meta.loc[ids_test_label0[idx_b]],
                        age_bins,
                    ))
                    row['jsd_label'] = jsd_categorical(y_train_np, y_test_np)

                results.append(row)

            print(f"Split {split_idx+1} done")

    return pd.DataFrame(results)


def main(args):
    df = pd.read_csv(args.features_csv)
    meta = load_metadata(args.metadata_csv, df['id']) if args.metadata_csv else None
    df_results = shift_per_pair(
        df, meta,
        n_splits=args.num_splits,
        cohort_n=args.cohort_n,
        n_permutations=args.n_permutations,
        n_ot_permutations=args.n_ot_permutations,
        n_projections=args.n_projections,
        knn_k=args.knn_k,
        knn_pca_dim=args.knn_pca_dim,
        standardize=args.standardize,
    )
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved distribution-shift table to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature-distribution shift and metadata divergence on the paired cohort-probe control subsets")
    parser.add_argument('--features_csv', type=Path, required=True, help='Path to CSV file containing features and labels')
    parser.add_argument('--metadata_csv', type=Path, default=None, help='Per-scan metadata keyed by id (e.g. csvs/metadata/meta_info_used_samples.csv); omit to skip the metadata divergences')
    parser.add_argument('--output_csv', type=Path, default=Path('results_distribution_shift.csv'), help='Path to save the results CSV')
    parser.add_argument('--num_splits', type=int, default=20, help='Must match the --num_splits given to classify_disease.py')
    parser.add_argument('--cohort_n', type=int, default=150, help='Must match the --cohort_n given to classify_disease.py')
    parser.add_argument('--n_permutations', type=int, default=200, help='Permutations for the MMD p-value (0 to skip)')
    parser.add_argument('--n_ot_permutations', type=int, default=20, help='Permutations for the null mean subtracted in wasserstein_rel (0 to skip)')
    parser.add_argument('--n_projections', type=int, default=500, help='Random projections for the sliced Wasserstein distance')
    parser.add_argument('--knn_k', type=int, default=5, help='Neighbour rank for the k-NN KL estimator')
    parser.add_argument('--knn_pca_dim', type=int, default=10, help='PCA components the k-NN KL estimator is computed in')
    parser.add_argument('--standardize', action='store_true', help='z-score each pair on its pooled subset first (off by default, matching wasserstein_dist in classify_disease.py)')

    args = parser.parse_args()
    main(args)
