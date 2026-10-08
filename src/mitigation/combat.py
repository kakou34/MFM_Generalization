"""ComBat harmonization of pooled foundation-model features across cohorts.

Implements the parametric empirical-Bayes ComBat of Johnson et al. (2007), as
used for neuroimaging by Fortin et al. (2017/2018) and the ``neuroCombat``
package, on the ``Feature1..FeatureN`` columns of a pooled feature CSV. The
batch variable is the cohort, so ComBat removes the per-cohort, per-feature
shift (location) and scaling (scale) that the cohort-separability probes pick
up, while keeping the effects of the biological covariates it is given.

Covariates are biological variables only (sex, optionally age). The diagnosis
label is deliberately never used: ComBat adds each scan's covariate effect back
after harmonizing, so passing the label would write every test scan's own
diagnosis into its harmonized features and inflate the disease AUCs.

The fit is label-free but transductive: it uses every scan of every cohort,
including the cohorts the disease probe is later tested on (their features,
cohort and covariates -- never their labels), which is how ComBat is normally
applied to a multi-site dataset.

The output CSV has the same schema as the input, so it is a drop-in for
``src/adaptation/classify_cohort.py`` / ``classify_disease.py``.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


COVARIATE_CHOICES = ("sex", "age")

DEFAULT_METADATA_CSV = (
    Path(__file__).resolve().parents[2] / "csvs" / "metadata" / "meta_info_used_samples.csv"
)


class ComBat:
    """Parametric empirical-Bayes ComBat with a fit / transform split.

    ``fit`` estimates the covariate effects, the pooled per-feature mean and
    variance, and the per-batch location/scale parameters; ``transform`` maps
    scans from any of the fitted batches onto the pooled distribution. Keeping
    the two apart lets the same class be fit on one subset of scans (e.g. a
    training fold) and applied to another.
    """

    def __init__(self, eb=True, mean_only=False, tol=1e-4, max_iter=1000):
        self.eb = eb
        self.mean_only = mean_only
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, batch, covariates=None):
        X = np.asarray(X, dtype=np.float64)
        batch = np.asarray(batch)
        C = self._covariate_matrix(covariates, len(X))

        self.batches_ = np.unique(batch)
        batch_idx = [np.flatnonzero(batch == b) for b in self.batches_]
        n_per_batch = np.array([len(idx) for idx in batch_idx])
        if (n_per_batch < 2).any():
            raise ValueError("every batch needs at least 2 scans to estimate its scale")

        # Regress each feature on batch indicators plus covariates. The batch
        # coefficients act as per-batch intercepts; their size-weighted
        # average is the pooled (grand) mean.
        D_batch = (batch[:, None] == self.batches_[None, :]).astype(np.float64)
        design = np.hstack([D_batch, C])
        coef, *_ = np.linalg.lstsq(design, X, rcond=None)

        n_batches = len(self.batches_)
        self.grand_mean_ = (n_per_batch / len(X)) @ coef[:n_batches]
        self.beta_ = coef[n_batches:]
        self.var_pooled_ = ((X - design @ coef) ** 2).mean(axis=0)

        # Features that do not vary at all carry no batch effect to remove;
        # they are passed through unchanged in ``transform``.
        self.constant_ = self.var_pooled_ <= 0
        var_pooled = np.where(self.constant_, 1.0, self.var_pooled_)

        Z = (X - self.grand_mean_ - C @ self.beta_) / np.sqrt(var_pooled)

        gamma_hat = np.stack([Z[idx].mean(axis=0) for idx in batch_idx])
        if self.mean_only:
            delta_hat = np.ones_like(gamma_hat)
        else:
            delta_hat = np.stack([Z[idx].var(axis=0, ddof=1) for idx in batch_idx])

        if not self.eb:
            self.gamma_star_, self.delta_star_ = gamma_hat, delta_hat
            return self

        gamma_star, delta_star = [], []
        for i, idx in enumerate(batch_idx):
            g_star, d_star = self._shrink(Z[idx], gamma_hat[i], delta_hat[i])
            gamma_star.append(g_star)
            delta_star.append(d_star)
        self.gamma_star_ = np.stack(gamma_star)
        self.delta_star_ = np.stack(delta_star)
        return self

    def transform(self, X, batch, covariates=None):
        X = np.asarray(X, dtype=np.float64)
        batch = np.asarray(batch)
        C = self._covariate_matrix(covariates, len(X))

        unknown = set(np.unique(batch)) - set(self.batches_)
        if unknown:
            raise ValueError(f"batches not seen during fit: {sorted(unknown)}")

        var_pooled = np.where(self.constant_, 1.0, self.var_pooled_)
        stand_mean = self.grand_mean_ + C @ self.beta_
        Z = (X - stand_mean) / np.sqrt(var_pooled)

        for i, b in enumerate(self.batches_):
            rows = batch == b
            Z[rows] = (Z[rows] - self.gamma_star_[i]) / np.sqrt(self.delta_star_[i])

        X_adj = Z * np.sqrt(var_pooled) + stand_mean
        X_adj[:, self.constant_] = X[:, self.constant_]
        return X_adj

    def fit_transform(self, X, batch, covariates=None):
        return self.fit(X, batch, covariates).transform(X, batch, covariates)

    def _shrink(self, Z_b, gamma_hat, delta_hat):
        """Empirical-Bayes posterior location/scale for one batch.

        Location gets a normal prior and scale an inverse-gamma prior, both
        fit across features; the two posteriors depend on each other, so they
        are iterated to a fixed point.
        """
        n = len(Z_b)
        gamma_bar = gamma_hat.mean()
        tau2 = gamma_hat.var(ddof=1)

        def post_gamma(delta):
            return (tau2 * n * gamma_hat + delta * gamma_bar) / (tau2 * n + delta)

        if self.mean_only:
            return post_gamma(1.0), np.ones_like(delta_hat)

        m = delta_hat.mean()
        s2 = delta_hat.var(ddof=1)
        a_prior = (2 * s2 + m ** 2) / s2
        b_prior = (m * s2 + m ** 3) / s2

        g_old, d_old = gamma_hat, delta_hat
        for _ in range(self.max_iter):
            g_new = post_gamma(d_old)
            sum2 = ((Z_b - g_new) ** 2).sum(axis=0)
            d_new = (0.5 * sum2 + b_prior) / (n / 2 + a_prior - 1)
            change = max(
                np.max(np.abs(g_new - g_old) / np.abs(g_old)),
                np.max(np.abs(d_new - d_old) / d_old),
            )
            g_old, d_old = g_new, d_new
            if change <= self.tol:
                break
        return g_old, d_old

    @staticmethod
    def _covariate_matrix(covariates, n):
        if covariates is None:
            return np.zeros((n, 0))
        C = np.asarray(covariates, dtype=np.float64)
        return C.reshape(n, -1)


def build_covariates(meta, covariates):
    """Numeric covariate matrix for ComBat from the metadata columns.

    ``sex`` becomes a male indicator. ``age`` is ``age_at_scan``; the few
    scans without one get their cohort's median age, which uses nothing but
    the cohort itself (never the diagnosis).
    """
    cols = []
    for name in covariates:
        if name == "sex":
            sex = meta["sex"].str.lower()
            if not sex.isin(["male", "female"]).all():
                raise ValueError(f"unexpected sex values: {sorted(sex.unique())}")
            cols.append((sex == "male").astype(np.float64).to_numpy())
        elif name == "age":
            age = meta["age_at_scan"].astype(np.float64)
            missing = age.isna()
            if missing.any():
                print(f"Imputing age_at_scan with the cohort median for {missing.sum()} scans: "
                      f"{meta.loc[missing, 'cohort'].value_counts().to_dict()}")
                age = age.fillna(meta.groupby("cohort")["age_at_scan"].transform("median"))
            cols.append(age.to_numpy())
        else:
            raise ValueError(f"unknown covariate {name!r}")
    return np.column_stack(cols) if cols else None


def harmonize_csv(features_csv, metadata_csv, output_csv, covariates=("sex",),
                  eb=True, mean_only=False):
    df = pd.read_csv(features_csv)
    meta = pd.read_csv(metadata_csv)

    feature_cols = [c for c in df.columns if c.startswith("Feature")]

    # Align metadata to the feature rows by scan id, and make sure both files
    # agree on each scan's cohort and label.
    meta = df[["id", "cohort", "label"]].merge(
        meta.drop(columns=["cohort", "label"]).assign(
            meta_cohort=meta["cohort"], meta_label=meta["label"]
        ),
        on="id", how="left", validate="one_to_one",
    )
    if meta["meta_cohort"].isna().any():
        missing = meta.loc[meta["meta_cohort"].isna(), "id"].tolist()
        raise ValueError(f"{len(missing)} scans have no metadata, e.g. {missing[:5]}")
    if not ((meta["cohort"] == meta["meta_cohort"]).all()
            and (meta["label"] == meta["meta_label"]).all()):
        raise ValueError("cohort/label disagree between the feature and metadata CSVs")

    C = build_covariates(meta, covariates)
    X = df[feature_cols].to_numpy(dtype=np.float64)

    combat = ComBat(eb=eb, mean_only=mean_only)
    X_adj = combat.fit_transform(X, df["cohort"].to_numpy(), C)

    out = df.copy()
    out[feature_cols] = X_adj.astype(np.float32)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"ComBat (covariates={list(covariates)}, eb={eb}, mean_only={mean_only}) on "
          f"{len(df)} scans x {len(feature_cols)} features "
          f"({combat.constant_.sum()} constant features left as-is) -> {output_csv}")
    return out


def main(args):
    harmonize_csv(
        args.features_csv,
        args.metadata_csv,
        args.output_csv,
        covariates=args.covariates,
        eb=not args.no_eb,
        mean_only=args.mean_only,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonize pooled FM features across cohorts with ComBat")
    parser.add_argument('--features_csv', type=Path, required=True, help='Pooled feature CSV (id, label, cohort, Feature1..N)')
    parser.add_argument('--output_csv', type=Path, required=True, help='Where to write the harmonized feature CSV (same schema)')
    parser.add_argument('--metadata_csv', type=Path, default=DEFAULT_METADATA_CSV, help='Per-scan metadata with sex / age_at_scan, keyed by id')
    parser.add_argument('--covariates', nargs='*', default=['sex'], choices=COVARIATE_CHOICES, help='Biological covariates whose effects ComBat preserves (never the diagnosis label)')
    parser.add_argument('--no_eb', action='store_true', help='Skip the empirical-Bayes shrinkage and use the per-cohort estimates directly')
    parser.add_argument('--mean_only', action='store_true', help='Only remove per-cohort location shifts, not scale differences')

    args = parser.parse_args()
    main(args)
