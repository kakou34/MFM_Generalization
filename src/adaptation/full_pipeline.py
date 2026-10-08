"""Cohort separability and cross-cohort disease classification, end to end.

Python implementation of Algorithm 1 in ``pseudo_code.tex``. Each iteration
``r`` first draws every random choice -- the control samples of each cohort
pair, all splits, the oversampling and the probe seeds -- and then, for every
foundation model ``f``, runs the steps below on those same draws, so within an
iteration the models differ only in their features:

1. Cohort classification -- for each unordered cohort pair ``(i, j)``, sample
   up to ``n_max`` cognitively-normal scans per cohort, split them 50/50
   stratified by cohort, oversample the smaller cohort of the training half
   with repetition, train a linear probe to tell the two cohorts apart and
   record its test performance in the symmetric matrix
   ``P_coh[i, j] = P_coh[j, i]``.
   A subset of the distribution-shift measures of ``baseline_preds.py`` is
   computed on the same sampled controls, all of them rather than one half of
   the split: ``wasserstein_dist``, ``wasserstein_rel``, ``mmd2`` and
   ``frechet_dist``, plus, given ``--metadata_csv``, the metadata
   divergences.
2. Disease classification -- for each cohort ``i``, split it 70/30 stratified
   by diagnosis, train one AD-vs-CN linear probe per learning rate on the 70%,
   keep the one with the best AUC on the held-out 30%, and record its
   performance on that 30% (``P_dis[i, i]``, internal validation) and on every
   other full cohort (``P_dis[i, j]``, external test).
3. Flatten both matrices into one row per (iteration, model, ordered pair).

Performance is reported as ROC-AUC, balanced accuracy and accuracy
(``METRICS``); the two accuracies threshold the probe's probability at 0.5.
All three are NaN when the evaluation set holds a single class.

Both steps use the linear probe from ``linear_probe.py``. The cohort probe
runs at a fixed learning rate; only the disease probe sweeps ``LR_GRID``. The
cohort probe can also get its own iteration budget (``--cohort_iterations``),
since it trains on far fewer scans than the disease probe. Each probe has
its own standardization switch (``--no_cohort_standardize``,
``--no_disease_standardize``). The disease
sweep is scored on the 30% held-out fold itself, which keeps 70% of a small
cohort for training instead of carving a further inner split out of it. The
price is that ``P_dis[i, i]`` is the best of ``len(LR_GRID)`` scores on that
fold and so optimistically biased; the external ``P_dis[i, j]`` are not.

All shift measures are symmetric, so both rows of a cohort pair carry the
same values. ``jsd_label`` is the exception to the shared subset: the subset
holds controls only, so it compares the AD/CN mix of the two full cohorts.

When a ``subject_id`` column is present, every split is group-aware so scans of
the same subject never land on both sides.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from baseline_preds import (
    AGE_BIN_WIDTH,
    AGE_COL,
    META_METRICS,
    frechet_distance,
    gaussian_fit,
    jsd_categorical,
    load_metadata,
    metadata_shift,
    permutation_tests,
    wasserstein_ot,
)
from linear_probe import (
    DEVICE,
    LR_GRID,
    MIN_PER_SIDE,
    LinearClassifier,
    predict_scores,
    safe_auc,
    set_seed,
    train_linear_probe,
    train_linear_probes,
)

set_seed()


def stratified_split(y, test_size, seed, groups=None):
    """Train/test row indices, stratified by ``y`` and group-aware if ``groups``.

    With groups, ``StratifiedGroupKFold`` cannot hit an arbitrary fraction, so
    the first of ``round(1 / test_size)`` folds is used as the test set (e.g.
    3 folds -> ~33% for a nominal 30%).
    """
    idx = np.arange(len(y))
    if groups is not None:
        n_splits = max(2, int(round(1 / test_size)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return next(splitter.split(idx, y, groups))
    return train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)


def fit_scaler(X_train, standardize):
    """A function mapping a numpy array to a (scaled) float32 tensor on DEVICE.

    The scaler is fit on ``X_train`` only, so nothing leaks from held-out data.
    """
    scaler = StandardScaler().fit(X_train) if standardize else None

    def to_device(X):
        if scaler is not None:
            X = scaler.transform(X)
        return torch.from_numpy(X.astype(np.float32)).to(DEVICE)

    return to_device


def labels_to_device(y):
    return torch.from_numpy(y.astype(np.float32)).to(DEVICE).unsqueeze(1)


METRICS = ('auc', 'bacc', 'acc')


def evaluate(y, scores):
    """ROC-AUC, balanced accuracy and accuracy, in the order of ``METRICS``."""
    if len(np.unique(y)) < 2:
        return np.full(len(METRICS), np.nan)
    preds = (scores > 0.5).astype(np.float32)
    return np.array([
        safe_auc(y, scores),
        balanced_accuracy_score(y, preds),
        accuracy_score(y, preds),
    ])


def new_seed(rng):
    return int(rng.integers(2**31 - 1))


FEATURE_METRICS = ['wasserstein_dist', 'wasserstein_rel', 'mmd2', 'frechet_dist']


def feature_shift(X_a, X_b, rng, n_ot_permutations=20, standardize=False):
    """The feature-space shift measures kept from ``baseline_preds.feature_shift``.

    Computed the same way, but without the measures this pipeline does not
    use; in particular the MMD permutation test is skipped, since only the
    statistic itself is kept.
    """
    X_a = X_a.astype(np.float64)
    X_b = X_b.astype(np.float64)
    if standardize:
        # Pooled over the pair, so neither side is treated as the reference.
        scaler = StandardScaler().fit(np.concatenate([X_a, X_b]))
        X_a, X_b = scaler.transform(X_a), scaler.transform(X_b)

    wasserstein, wasserstein_null = wasserstein_ot(X_a, X_b, rng, n_ot_permutations)
    total_var = X_a.var(axis=0, ddof=1).sum() + X_b.var(axis=0, ddof=1).sum()
    mu_a, cov_a = gaussian_fit(X_a)
    mu_b, cov_b = gaussian_fit(X_b)

    return {
        'wasserstein_dist': wasserstein,
        'wasserstein_rel': (wasserstein - wasserstein_null) / total_var,
        'mmd2': permutation_tests(X_a, X_b, rng, n_permutations=0)['mmd2'],
        'frechet_dist': frechet_distance(mu_a, cov_a, mu_b, cov_b),
    }


def age_bins_for(meta):
    """5-year age bins spanning every scan in the metadata, as in baseline_preds."""
    age = meta[AGE_COL].dropna()
    lo = np.floor(age.min() / AGE_BIN_WIDTH) * AGE_BIN_WIDTH
    hi = np.ceil(age.max() / AGE_BIN_WIDTH) * AGE_BIN_WIDTH + AGE_BIN_WIDTH
    return np.arange(lo, hi, AGE_BIN_WIDTH)


def draw_pair_plan(y_i, y_j, groups_i, groups_j, n_max, rng):
    """Step 1 draws for one cohort pair, independent of the foundation model.

    Samples up to ``n_max`` controls per cohort, splits them 50/50 stratified by
    cohort and oversamples the smaller cohort of the training half. Indices
    refer to the pooled pair ``concat(controls_i[sel_i], controls_j[sel_j])``.
    Returns None when a side is too small to split.
    """
    cn_i, cn_j = np.flatnonzero(y_i == 0), np.flatnonzero(y_j == 0)
    n_i, n_j = min(len(cn_i), n_max), min(len(cn_j), n_max)
    if n_i < MIN_PER_SIDE or n_j < MIN_PER_SIDE:
        return None

    # Sample n_i / n_j controls without replacement.
    sel_i = rng.choice(cn_i, size=n_i, replace=False)
    sel_j = rng.choice(cn_j, size=n_j, replace=False)
    c = np.concatenate([np.ones(n_i), np.zeros(n_j)]).astype(np.float32)
    groups = None
    if groups_i is not None:
        groups = np.concatenate([groups_i[sel_i], groups_j[sel_j]])

    # 50/50 split, stratified by cohort.
    tr, te = stratified_split(c, 0.5, new_seed(rng), groups)

    # Oversample the smaller cohort of the training half by repetition, adding
    # Delta = |S_M| - |S_m| draws with replacement. The test half is unchanged.
    pos, neg = tr[c[tr] == 1], tr[c[tr] == 0]
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    extra = rng.choice(minority, size=len(majority) - len(minority), replace=True)
    train = rng.permutation(np.concatenate([majority, minority, extra]))

    return {
        'sel_i': sel_i, 'sel_j': sel_j, 'c': c,
        'tr': tr, 'train': train, 'te': te,
        'torch_seed': new_seed(rng),
        'metric_seed': new_seed(rng),
    }


def draw_disease_plan(y, groups, val_size, rng):
    """Step 2 draws for one training cohort, independent of the foundation model.

    The 70/30 split, whose 30% both selects the learning rate and gives the
    internal performance, and the probe seed. Returns None, with the reason,
    when the cohort cannot be split.
    """
    if min((y == 0).sum(), (y == 1).sum()) < 2:
        return None, "too few scans of a class"
    try:
        tr, val = stratified_split(y, val_size, new_seed(rng), groups)
    except ValueError as e:
        return None, f"cannot split ({e})"

    plan = {
        'tr': tr, 'val': val,
        'torch_seed': new_seed(rng),
    }
    return plan, None


def draw_iteration_plan(cohort_data, r, n_max, val_size):
    """All random draws of iteration ``r``, shared by every foundation model.

    ``cohort_data[k]`` is ``(labels, subject ids or None)`` of cohort ``k``.
    Every model is then trained and evaluated on the same cohort-pair samples,
    the same splits and the same probe seeds, so within an iteration the
    models differ only in their features.
    """
    pairs = {}
    for i, j in itertools.combinations(range(len(cohort_data)), 2):
        (y_i, g_i), (y_j, g_j) = cohort_data[i], cohort_data[j]
        rng = np.random.default_rng([r, 0, i, j])
        pairs[(i, j)] = draw_pair_plan(y_i, y_j, g_i, g_j, n_max, rng)

    disease = {}
    for i, (y, groups) in enumerate(cohort_data):
        rng = np.random.default_rng([r, 1, i])
        disease[i] = draw_disease_plan(y, groups, val_size, rng)

    return pairs, disease


def cohort_performance(X, plan, iterations, lr, batch_size, standardize):
    """Step 1 for one cohort pair: test ``METRICS`` of a probe separating the two cohorts.

    ``X`` holds the pooled sampled controls of the pair, in the order of
    ``plan``. Cohort ``i`` is the positive class.
    """
    c = plan['c']
    # Fit on the (non-oversampled) training half only.
    to_device = fit_scaler(X[plan['tr']], standardize)
    train, te = plan['train'], plan['te']

    torch.manual_seed(plan['torch_seed'])
    g = train_linear_probe(
        to_device(X[train]), labels_to_device(c[train]),
        lr=lr, iterations=iterations, batch_size=batch_size,
    )
    return evaluate(c[te], predict_scores(g, to_device(X[te])))


def cohort_classification(data, cohorts, pair_plans, iterations, lr, batch_size, standardize):
    """Step 1's probe for one iteration and one foundation model.

    ``data`` is from ``cohort_arrays`` and ``pair_plans`` from
    ``draw_iteration_plan``. Returns the symmetric cohort performance array, of
    shape ``(K, K, len(METRICS))`` and indexed by position in ``cohorts``, NaN
    for pairs with too few controls on a side to split them in two.
    """
    K = len(cohorts)
    P_coh = np.full((K, K, len(METRICS)), np.nan)
    for i, j in itertools.combinations(range(K), 2):
        plan = pair_plans[(i, j)]
        if plan is not None:
            X = np.concatenate([data[i][0][plan['sel_i']], data[j][0][plan['sel_j']]])
            P_coh[i, j] = P_coh[j, i] = cohort_performance(
                X, plan, iterations, lr, batch_size, standardize,
            )
        perf = P_coh[i, j]
        print(f"  cohort {cohorts[i]} vs {cohorts[j]}: AUC={perf[0]:.3f}, BACC={perf[1]:.3f}")
    return P_coh


def train_disease_probe(X_tr, y_tr, X_val, y_val, plan, lr_grid, iterations, batch_size,
                        standardize, head=LinearClassifier):
    """Step 2 training: one probe per rate of ``lr_grid`` on X_tr, the best on X_val kept.

    The probes are ranked by AUC on ``(X_val, y_val)``, the same fold the
    caller then reports as internal validation. Returns
    ``(model, to_device, best_lr)``, where ``to_device`` applies the scaler fit
    on ``X_tr`` so the caller can score held-out and external data. ``head`` is
    the probe architecture.
    """
    to_device = fit_scaler(X_tr, standardize)
    torch.manual_seed(plan['torch_seed'])
    model, best_lr, _auc = train_linear_probes(
        to_device(X_tr), labels_to_device(y_tr),
        to_device(X_val), labels_to_device(y_val),
        lr_grid=lr_grid, iterations=iterations, batch_size=batch_size, head=head,
    )
    return model, to_device, best_lr


def cohort_arrays(df, cohorts, with_ids=False):
    """``data[k] = (features, labels, ids or None)`` of cohort ``cohorts[k]``."""
    feature_cols = [c for c in df.columns if c.startswith("Feature")]
    data = {}
    for k, cohort in enumerate(cohorts):
        sub = df[df['cohort'] == cohort]
        data[k] = (
            sub[feature_cols].values.astype(np.float32),
            sub['label'].values.astype(np.float32),
            sub['id'].values if with_ids else None,
        )
    return data


def disease_classification(data, cohorts, disease_plans, lr_grid, iterations, batch_size,
                           standardize, head=LinearClassifier):
    """Step 2 for one iteration and one foundation model.

    ``data`` is from ``cohort_arrays`` and ``disease_plans`` from
    ``draw_iteration_plan``. Returns the disease performance array, of shape
    ``(K, K, len(METRICS))`` and indexed by position in ``cohorts``, and the
    learning rate picked for each disease probe. ``head`` is the probe
    architecture (linear here; ``disease_mlp.py`` passes an MLP).
    """
    K = len(cohorts)
    P_dis = np.full((K, K, len(METRICS)), np.nan)
    best_lrs = np.full(K, np.nan)
    for i in range(K):
        X, y, _ids = data[i]
        plan, reason = disease_plans[i]
        if plan is None:
            print(f"  disease train={cohorts[i]}: skipped, {reason}")
            continue
        tr, val = plan['tr'], plan['val']

        h, to_device, best_lrs[i] = train_disease_probe(
            X[tr], y[tr], X[val], y[val], plan, lr_grid, iterations, batch_size,
            standardize, head=head,
        )

        # Internal validation on the held-out fold (the one that picked the LR).
        P_dis[i, i] = evaluate(y[val], predict_scores(h, to_device(X[val])))
        # External test on every other full cohort.
        for j in range(K):
            if j != i:
                X_j, y_j, _ids = data[j]
                P_dis[i, j] = evaluate(y_j, predict_scores(h, to_device(X_j)))

        print(f"  disease train={cohorts[i]}: lr={best_lrs[i]}, "
              f"internal AUC={P_dis[i, i, 0]:.3f}, BACC={P_dis[i, i, 1]:.3f}")

    return P_dis, best_lrs


def run_model(df, cohorts, pair_plans, disease_plans, cohort_lr, lr_grid,
              iterations, batch_size, cohort_standardize, disease_standardize,
              meta=None, age_bins=None, shift_kwargs=None, cohort_iterations=None):
    """Steps 1 and 2 for one iteration and one foundation model.

    ``pair_plans`` and ``disease_plans`` are the iteration's draws from
    ``draw_iteration_plan``. Returns the cohort and disease performance
    arrays, of shape ``(K, K, len(METRICS))`` and indexed by position in
    ``cohorts``, the learning rate picked for each
    disease probe, and a dict ``shift[(k, l)]`` of distribution-shift measures
    oriented ``k -> l``. The cohort probe trains for ``cohort_iterations``
    (default: ``iterations``). Each probe standardizes its features if its
    ``cohort_standardize`` / ``disease_standardize`` is set.
    """
    K = len(cohorts)
    shift_kwargs = shift_kwargs or {}
    data = cohort_arrays(df, cohorts, with_ids=meta is not None)

    empty = {m: np.nan for m in FEATURE_METRICS}
    if meta is not None:
        empty.update({m: np.nan for m in META_METRICS})

    # ---- Step 1: cohort classification ----
    P_coh = cohort_classification(
        data, cohorts, pair_plans, cohort_iterations or iterations,
        cohort_lr, batch_size, cohort_standardize,
    )

    # Shift measures, on the same controls the cohort probe used.
    shift = {}
    for i, j in itertools.combinations(range(K), 2):
        X_i, y_i, id_i = data[i]
        X_j, y_j, id_j = data[j]
        plan = pair_plans[(i, j)]

        metrics = dict(empty)
        if plan is not None:
            A, B = X_i[plan['sel_i']], X_j[plan['sel_j']]
            metric_rng = np.random.default_rng(plan['metric_seed'])
            metrics.update(feature_shift(A, B, metric_rng, **shift_kwargs))
            if meta is not None:
                metrics.update(metadata_shift(
                    meta.loc[id_i[plan['sel_i']]], meta.loc[id_j[plan['sel_j']]], age_bins,
                ))

        # Label shift between the two full cohorts (the subset is controls only).
        metrics['jsd_label'] = jsd_categorical(y_i, y_j)

        shift[(i, j)] = shift[(j, i)] = metrics

    # ---- Step 2: disease classification ----
    P_dis, best_lrs = disease_classification(
        data, cohorts, disease_plans, lr_grid, iterations, batch_size, disease_standardize,
    )

    return P_coh, P_dis, best_lrs, shift


def align_features(features):
    """Put every model's feature frame in the same scan order, or fail.

    The shared draws index scans by position within each cohort, so all
    frames must hold the same scans with the same cohort and label. With an
    ``id`` column the frames are reordered to match the first one; without it
    they must already be in the same order.
    """
    names = list(features)
    ref = features[names[0]]
    keys = [c for c in ('id', 'cohort', 'label', 'subject_id') if c in ref.columns]
    aligned = {names[0]: ref}
    for name in names[1:]:
        df = features[name]
        if 'id' in ref.columns and 'id' in df.columns:
            if set(df['id']) != set(ref['id']) or df['id'].duplicated().any():
                raise ValueError(f"{name} does not hold the same scans as {names[0]}")
            df = df.set_index('id').loc[ref['id']].reset_index()
        elif len(df) != len(ref):
            raise ValueError(f"{name} has {len(df)} scans, {names[0]} has {len(ref)}")
        for key in keys:
            if key in df.columns and not (df[key].values == ref[key].values).all():
                raise ValueError(f"{name} and {names[0]} disagree on '{key}'")
        aligned[name] = df.reset_index(drop=True)
    return aligned


def cohort_draw_inputs(ref):
    """The sorted cohorts and, per cohort, ``(labels, subject ids or None)``.

    This is everything the random draws of ``draw_iteration_plan`` depend on
    besides the iteration and the split settings.
    """
    cohorts = sorted(ref['cohort'].unique().tolist())
    has_groups = 'subject_id' in ref.columns
    cohort_data = []
    for cohort in cohorts:
        sub = ref[ref['cohort'] == cohort]
        cohort_data.append((
            sub['label'].values.astype(np.float32),
            sub['subject_id'].values if has_groups else None,
        ))
    return cohorts, cohort_data


def run_pipeline(features, n_iterations=20, n_max=150, cohort_lr=1e-3,
                 lr_grid=LR_GRID, val_size=0.3,
                 iterations=2000, batch_size=64, cohort_standardize=True,
                 disease_standardize=True, meta=None, shift_kwargs=None,
                 cohort_iterations=None):
    """Run the full algorithm and return the data frame T.

    ``features`` maps a foundation-model name to its feature data frame
    (columns ``cohort``, ``label``, ``Feature1..FeatureN`` and optionally
    ``id`` and ``subject_id``); all frames must hold the same scans.
    ``meta`` is per-scan metadata indexed by ``id``, from
    ``baseline_preds.load_metadata``; without it the metadata divergences are
    skipped. ``shift_kwargs`` go to ``feature_shift``.
    """
    features = align_features(features)
    cohorts, cohort_data = cohort_draw_inputs(next(iter(features.values())))

    age_bins = age_bins_for(meta) if meta is not None else None
    rows = []
    for r in range(n_iterations):
        pair_plans, disease_plans = draw_iteration_plan(cohort_data, r, n_max, val_size)
        for model_name, df in features.items():
            print(f"\n=== Iteration {r + 1}/{n_iterations}, model {model_name} ===")
            P_coh, P_dis, best_lrs, shift = run_model(
                df, cohorts, pair_plans, disease_plans,
                cohort_lr=cohort_lr,
                lr_grid=lr_grid,
                iterations=iterations,
                batch_size=batch_size,
                cohort_standardize=cohort_standardize,
                disease_standardize=disease_standardize,
                meta=meta,
                age_bins=age_bins,
                shift_kwargs=shift_kwargs,
                cohort_iterations=cohort_iterations,
            )

            # ---- Step 3: one row per ordered (train, test) pair ----
            for k, l in itertools.permutations(range(len(cohorts)), 2):
                rows.append({
                    'i': r,
                    'model': model_name,
                    'train': cohorts[k],
                    'test': cohorts[l],
                    **{f'{m}_cohort': P_coh[k, l, n] for n, m in enumerate(METRICS)},
                    **{f'{m}_disease': P_dis[k, l, n] for n, m in enumerate(METRICS)},
                    **{f'{m}_disease_internal': P_dis[k, k, n] for n, m in enumerate(METRICS)},
                    'disease_lr': best_lrs[k],
                    **shift[(k, l)],
                })

    return pd.DataFrame(rows)


ROW_KEY = ['i', 'model', 'train', 'test']


def replace_columns(saved, new):
    """A saved frame T with the columns of ``new`` replaced by ``new``'s values.

    Used to rerun one step and keep the other columns from an earlier run.
    Rows are matched on ``ROW_KEY``, and every row of ``new`` must match exactly
    one saved row. Only the saved rows matched are returned, in the saved
    column order.
    """
    extra = sorted(set(new.columns) - set(saved.columns))
    if extra:
        raise ValueError(f"columns {extra} are not in the saved frame")
    saved = saved[saved['model'].isin(new['model'].unique())]
    kept = saved.drop(columns=[c for c in new.columns if c not in ROW_KEY])
    merged = new.merge(kept, on=ROW_KEY, how='left', validate='one_to_one', indicator=True)
    unmatched = merged['_merge'] == 'left_only'
    if unmatched.any():
        raise ValueError(f"{unmatched.sum()} rows have no counterpart in the saved frame, "
                         f"e.g. {merged.loc[unmatched, ROW_KEY].iloc[0].to_dict()}")
    if len(merged) < len(saved):
        print(f"Note: {len(saved) - len(merged)} saved rows were not recomputed and are left out")
    return merged[saved.columns]


def parse_features(specs):
    """Parse ``name=path`` pairs into an ordered {name: data frame} dict."""
    features = {}
    for spec in specs:
        name, sep, path = spec.partition('=')
        if not sep:
            path = spec
            name = Path(spec).parent.name
        features[name] = pd.read_csv(Path(path))
    return features


def main(args):
    features = parse_features(args.features)
    meta = None
    if args.metadata_csv:
        ids = pd.unique(pd.concat([df['id'] for df in features.values()]))
        meta = load_metadata(args.metadata_csv, ids)
    T = run_pipeline(
        features,
        n_iterations=args.n_iterations,
        n_max=args.n_max,
        cohort_lr=args.cohort_lr,
        val_size=args.val_size,
        iterations=args.iterations,
        cohort_iterations=args.cohort_iterations,
        batch_size=args.batch_size,
        cohort_standardize=args.cohort_standardize,
        disease_standardize=args.disease_standardize,
        meta=meta,
        shift_kwargs=dict(
            n_ot_permutations=args.n_ot_permutations,
            standardize=args.shift_standardize,
        ),
    )
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(out_path, index=False)
    print(f"\nSaved {len(T)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cohort separability and cross-cohort disease classification (Algorithm 1)"
    )
    parser.add_argument('--features', nargs='+', required=True,
                        help='One feature CSV per foundation model, as name=path '
                             '(or just path, in which case the parent folder name is used)')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to save the data frame T')
    parser.add_argument('--n_iterations', type=int, default=20,
                        help='Number of repetitions R of the whole procedure')
    parser.add_argument('--n_max', type=int, default=150,
                        help='Maximum number of CN scans sampled per cohort for the cohort probe')
    parser.add_argument('--cohort_lr', type=float, default=1e-3,
                        help='Fixed learning rate for the cohort probe')
    parser.add_argument('--val_size', type=float, default=0.3,
                        help='Fraction of a cohort held out to select the learning rate and '
                             'report internal disease validation')
    parser.add_argument('--iterations', type=int, default=2000,
                        help='Number of optimizer iterations used to train each linear head')
    parser.add_argument('--cohort_iterations', type=int, default=None,
                        help='Iterations for the cohort probe only (default: --iterations)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training the linear heads')
    parser.add_argument('--no_cohort_standardize', dest='cohort_standardize', action='store_false',
                        help='Feed raw features to the cohort probe instead of standardizing them')
    parser.add_argument('--no_disease_standardize', dest='disease_standardize', action='store_false',
                        help='Feed raw features to the disease probe instead of standardizing them')
    parser.add_argument('--metadata_csv', type=Path, default=None,
                        help='Per-scan metadata keyed by id (e.g. csvs/metadata/meta_info_used_samples.csv); '
                             'omit to skip the metadata divergences')
    parser.add_argument('--n_ot_permutations', type=int, default=20,
                        help='Permutations for the null mean subtracted in wasserstein_rel (0 to skip)')
    parser.add_argument('--shift_standardize', action='store_true',
                        help='z-score each pair on its pooled subset before the shift measures '
                             '(off by default, as in baseline_preds.py)')

    main(parser.parse_args())
