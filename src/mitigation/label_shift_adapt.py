"""Label shift adaptation of the disease probe (Alexandari et al., ICML 2020).

"Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label
Shift Adaptation" (code: ``labelshiftadaptation/``; the core routines live in
the ``abstention`` package, ``TempScaling(bias_positions='all')`` +
``EMImbalanceAdapter``). The cohorts differ strongly in AD prevalence (from
~15% in AIBL to ~84% in OASIS4), so a probe trained on one cohort carries that
cohort's prior into every test cohort. The method corrects for this, assuming
p(x | y) is shared across cohorts and only p(y) changes:

1. BCTS -- bias-corrected temperature scaling. A temperature and per-class
   biases are fit on the probe's logits on the held-out validation fold of the
   training cohort, by minimising the NLL, so its posteriors are calibrated.
2. MLLS -- the EM algorithm of Saerens et al. (2002). Starting from the
   (calibrated) source prior, it re-estimates the test cohort's prior from the
   probe's posteriors on that cohort's scans alone, then reweights each
   posterior by ``pi_target / pi_source`` and renormalises.

Only the disease probe is adapted; cohort classification is not touched. The
adaptation is label-free but transductive: the target prior is estimated from
all scans of the test cohort (their predictions, never their labels).

The probes themselves are trained exactly as in
``src/adaptation/classify_disease.py`` (same splits, LR sweep and seed), so
``disease_auc`` matches the baseline run. For a binary task both BCTS and the
prior reweighting are monotonic in the logit, so **AUC cannot change**; the
adaptation only moves the decision threshold and therefore affects accuracy,
balanced accuracy, F1 and NLL. Each row reports the metrics for four variants:

- ``raw``    the probe's own sigmoid output (the baseline)
- ``bcts``   after calibration only
- ``mlls``   after calibration + EM prior adaptation (the method)
- ``oracle`` after calibration + reweighting with the *true* test prior (an
             upper bound on what prior correction alone can give)

``disease_bacc`` / ``disease_f1`` hold the ``mlls`` values so the output drops
into the same analyses as the baseline CSVs. The paired cohort-separability
probe and the Wasserstein distance of ``classify_disease.py`` are not
recomputed: they depend on the features only, which this method leaves
unchanged, so they are identical to the baseline results.
"""

from pathlib import Path
import argparse
import sys

from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adaptation"))
from linear_probe import (
    DEVICE,
    safe_auc,
    set_seed,
    train_linear_probes,
)

set_seed()

VARIANTS = ("raw", "bcts", "mlls", "oracle")


def predict_logits(model, X):
    """The probe's pre-sigmoid outputs for X, as a 1-D numpy array."""
    model.eval()
    with torch.no_grad():
        return model(X).cpu().numpy().ravel().astype(np.float64)


def to_preacts(logits):
    """Binary logits as two-class softmax pre-activations ``[-z/2, z/2]``.

    Softmax of these equals ``[1 - sigmoid(z), sigmoid(z)]``; it is also what
    ``abstention``'s ``inverse_softmax`` recovers from those probabilities, but
    built from the logits it never saturates at probabilities of 0 or 1.
    """
    return np.stack([-logits / 2, logits / 2], axis=1)


def one_hot(y, n_classes=2):
    return np.eye(n_classes)[np.asarray(y, dtype=int)]


class BCTS:
    """Bias-corrected temperature scaling (``abstention.calibration.TempScaling``
    with ``bias_positions='all'``).

    Fits ``softmax(preacts / T + b)`` to the labels by minimising the mean NLL
    with L-BFGS-B, starting from ``T = 1, b = 0``.
    """

    def fit(self, preacts, labels):
        labels = one_hot(labels, preacts.shape[1])
        n_classes = preacts.shape[1]

        def nll_and_grad(x):
            t, b = x[0], x[1:]
            log_p = log_softmax(preacts / t + b, axis=1)
            nll = -np.mean(np.sum(labels * log_p, axis=1))
            # d NLL / d z, with z = preacts / t + b
            grad_z = (np.exp(log_p) - labels) / len(preacts)
            grad_t = np.sum(grad_z * (-preacts / t ** 2))
            grad_b = grad_z.sum(axis=0)
            return nll, np.concatenate([[grad_t], grad_b])

        result = minimize(
            nll_and_grad,
            x0=np.concatenate([[1.0], np.zeros(n_classes)]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(1e-6, None)] + [(None, None)] * n_classes,
            tol=1e-7,
        )
        self.converged_ = bool(result.success)
        self.temperature_ = result.x[0]
        self.biases_ = result.x[1:]
        return self

    def __call__(self, preacts):
        return np.exp(log_softmax(preacts / self.temperature_ + self.biases_, axis=1))


def reweight(probs, multipliers):
    """Posteriors under a new prior: ``p(y|x) * w_y``, renormalised over y."""
    unnorm = probs * multipliers[None, :]
    return unnorm / unnorm.sum(axis=1, keepdims=True)


def em_prior(probs, source_prior, tol=1e-6, max_iter=100):
    """Maximum-likelihood target prior by EM (``abstention.label_shift.EMImbalanceAdapter``).

    ``probs`` are the calibrated source posteriors on the unlabeled target
    scans. Each step reweights them by the current prior estimate and takes
    the mean adapted posterior as the next estimate. Returns the estimated
    target prior, the number of iterations run, and whether it converged
    within ``max_iter``.
    """
    prior = source_prior
    for it in range(1, max_iter + 1):
        new_prior = reweight(probs, prior / source_prior).mean(axis=0)
        done = np.max(np.abs(new_prior - prior)) <= tol
        prior = new_prior
        if done:
            break
    return prior, it, bool(done)


def binary_metrics(y_true, p_pos):
    """Threshold-dependent metrics at 0.5 plus NLL, for positive-class probabilities."""
    preds = (p_pos > 0.5).astype(np.float32)
    p_true = np.where(y_true == 1, p_pos, 1 - p_pos)
    return {
        "acc": accuracy_score(y_true, preds),
        "bacc": balanced_accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, zero_division=0.0),
        "nll": -np.mean(np.log(np.clip(p_true, 1e-12, None))),
    }


def train_each_test_all(df, n_splits=20, iterations=2000, batch_size=64,
                        standardize=True, class_weight=False,
                        em_tol=1e-6, em_max_iter=100,
                        out_path="results_label_shift.csv"):
    cohorts = df['cohort'].unique().tolist()
    FEATURE_COLS = [c for c in df.columns if c.startswith("Feature")]

    results = []

    for train_cohort in cohorts:
        print(f"\n=== Training on {train_cohort} ===")

        train_df = df[df['cohort'] == train_cohort]
        X_np = train_df[FEATURE_COLS].values.astype(np.float32)
        y_np = train_df['label'].values.astype(np.float32)

        # ---- Splitter (group-aware if subject_id exists), as in classify_disease.py ----
        if 'subject_id' in train_df.columns:
            groups = train_df['subject_id'].values
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42
            )
            split_iterator = splitter.split(X_np, y_np, groups)
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=n_splits,
                test_size=0.3,
                random_state=42
            )
            split_iterator = splitter.split(X_np, y_np)

        for split_idx, (train_idx, val_idx) in enumerate(split_iterator):

            X_train_np, X_val_np = X_np[train_idx], X_np[val_idx]
            y_train_np, y_val_np = y_np[train_idx], y_np[val_idx]

            if standardize:
                scaler = StandardScaler().fit(X_train_np)
            else:
                scaler = None

            def to_device(X_arr):
                if scaler is not None:
                    X_arr = scaler.transform(X_arr).astype(np.float32)
                return torch.from_numpy(X_arr).to(DEVICE)

            X_train = to_device(X_train_np)
            y_train = torch.from_numpy(y_train_np).to(DEVICE).unsqueeze(1)
            X_val = to_device(X_val_np)
            y_val = torch.from_numpy(y_val_np).to(DEVICE).unsqueeze(1)

            if class_weight:
                n_pos = float((y_train_np == 1).sum())
                n_neg = float((y_train_np == 0).sum())
                pos_weight = (
                    torch.tensor([n_neg / n_pos], device=DEVICE)
                    if n_pos > 0 and n_neg > 0 else None
                )
            else:
                pos_weight = None

            model, best_lr, best_auc = train_linear_probes(
                X_train, y_train, X_val, y_val,
                iterations=iterations,
                batch_size=batch_size,
                pos_weight=pos_weight,
            )

            # ---- BCTS on the held-out validation fold of the training cohort ----
            # A single-class fold cannot pin down a calibration, so fall back to
            # the identity (T = 1, b = 0) and flag it in the output.
            val_preacts = to_preacts(predict_logits(model, X_val))
            calibrator = BCTS()
            if len(np.unique(y_val_np)) == 2:
                calibrator.fit(val_preacts, y_val_np)
            else:
                calibrator.converged_ = False
                calibrator.temperature_, calibrator.biases_ = 1.0, np.zeros(2)

            # Source prior as the mean calibrated validation posterior, as in
            # abstention (so that adapting the validation fold to itself
            # estimates no shift), rather than the validation label frequency.
            source_prior = calibrator(val_preacts).mean(axis=0)

            print(f"Split {split_idx+1}: best lr={best_lr}, val auc={best_auc:.3f}, "
                  f"BCTS T={calibrator.temperature_:.3f} b={np.round(calibrator.biases_, 3)}, "
                  f"source prior={source_prior[1]:.3f}")

            # ---- Evaluate on all cohorts ----
            for test_cohort in cohorts:
                if test_cohort == train_cohort:
                    # Internal performance: held-out validation fold only,
                    # never the scans the probe was trained on.
                    X_test_np, y_test_np = X_val_np, y_val_np
                else:
                    test_df = df[df['cohort'] == test_cohort]
                    X_test_np = test_df[FEATURE_COLS].values.astype(np.float32)
                    y_test_np = test_df['label'].values.astype(np.float32)

                logits = predict_logits(model, to_device(X_test_np))
                calib_probs = calibrator(to_preacts(logits))

                est_prior, em_iters, em_converged = em_prior(
                    calib_probs, source_prior, tol=em_tol, max_iter=em_max_iter
                )
                true_prior = one_hot(y_test_np).mean(axis=0)

                p_pos = {
                    "raw": 1 / (1 + np.exp(-logits)),
                    "bcts": calib_probs[:, 1],
                    "mlls": reweight(calib_probs, est_prior / source_prior)[:, 1],
                    "oracle": reweight(calib_probs, true_prior / source_prior)[:, 1],
                }

                row = {
                    'train': train_cohort,
                    'test': test_cohort,
                    'split': split_idx,
                    'disease_auc': safe_auc(y_test_np, logits),
                }
                for variant in VARIANTS:
                    for name, value in binary_metrics(y_test_np, p_pos[variant]).items():
                        row[f'disease_{name}_{variant}'] = value
                row.update({
                    # Drop-in columns: the adapted (MLLS) threshold metrics.
                    'disease_bacc': row['disease_bacc_mlls'],
                    'disease_f1': row['disease_f1_mlls'],
                    'source_prior': source_prior[1],
                    'est_prior': est_prior[1],
                    'true_prior': true_prior[1],
                    'prior_abs_err': abs(est_prior[1] - true_prior[1]),
                    'em_iters': em_iters,
                    'em_converged': em_converged,
                    'bcts_temperature': calibrator.temperature_,
                    'bcts_bias_diff': calibrator.biases_[1] - calibrator.biases_[0],
                    'bcts_converged': calibrator.converged_,
                    'n_test': len(y_test_np),
                    'best_lr': best_lr,
                })
                results.append(row)

    df_results = pd.DataFrame(results)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved label-shift-adapted cross-cohort matrix to {out_path}")

    return results, df_results


def main(args):
    df = pd.read_csv(args.features_csv)
    train_each_test_all(
        df,
        n_splits=args.num_splits,
        iterations=args.iterations,
        batch_size=args.batch_size,
        standardize=not args.no_standardize,
        class_weight=args.class_weight,
        em_tol=args.em_tol,
        em_max_iter=args.em_max_iter,
        out_path=args.output_csv,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disease classification with label shift adaptation (BCTS + EM/MLLS, Alexandari et al. 2020)")
    parser.add_argument('--features_csv', type=Path, required=True, help='Path to CSV file containing features and labels')
    parser.add_argument('--output_csv', type=Path, default=Path('results_label_shift.csv'), help='Path to save the results CSV')
    parser.add_argument('--num_splits', type=int, default=20, help='Number of splits for StratifiedShuffleSplit')
    parser.add_argument('--iterations', type=int, default=2000, help='Number of SGD iterations used to train each linear head')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training the linear heads')
    parser.add_argument('--no_standardize', action='store_true', help='Feed raw features to the probe instead of standardizing them')
    parser.add_argument('--class_weight', action='store_true', help='Re-weight the positive class in the loss (off by default)')
    parser.add_argument('--em_tol', type=float, default=1e-6, help='EM stops once the prior estimate moves by less than this')
    parser.add_argument('--em_max_iter', type=int, default=100, help='Maximum number of EM iterations')

    args = parser.parse_args()
    main(args)
