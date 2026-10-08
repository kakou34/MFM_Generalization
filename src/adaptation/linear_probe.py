"""Linear probing on frozen foundation-model features.

Shared by ``classify_cohort.py`` and ``classify_disease.py`` so both evaluate
the features with exactly the same head and optimizer. The setup follows the
DINOv2 linear-evaluation protocol: a single linear layer trained on top of the
frozen features, cosine-annealed over a fixed iteration budget. Unlike DINOv2,
which uses SGD with momentum, the head is optimized with Adam.

The training functions take an optional ``head`` architecture (default
``LinearClassifier``), so ``disease_mlp.py`` can run the exact same protocol
with a non-linear probe.
"""

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Learning-rate grid swept when no fixed rate is given. It follows the DINOv2
# linear-evaluation protocol's 1/5 spacing, shifted down by a decade because
# the head is trained with Adam rather than SGD, whose useful range is lower.
# The other two axes of their grid search (how many output blocks to use,
# whether to concatenate average-pooled patch tokens with the class token) do
# not apply here: the features are already pooled into a single vector per
# sample upstream, so the learning rate is the only parameter left to sweep.
LR_GRID = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

# A cohort side with fewer than this many samples cannot be split into a
# training and a test half that both contain more than one scan.
MIN_PER_SIDE = 4


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, num_classes=1):
        super().__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(x)


def infinite_batches(loader):
    """Yield batches forever, reshuffling each time the loader is exhausted."""
    while True:
        for batch in loader:
            yield batch


def train_heads(X_train, y_train, lrs, iterations=2000, batch_size=64, pos_weight=None,
                head=LinearClassifier):
    """Train one head per learning rate and return them as (lr, model).

    Every head is stepped on the same stream of batches, mirroring the DINOv2
    protocol: the (frozen) features are computed once and fed to every linear
    classifier, so sweeping the grid costs little more than a single run.
    ``head`` builds a model from the input dimension.
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    heads = []
    for lr in lrs:
        model = head(X_train.shape[1]).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
        heads.append((lr, model, optimizer, scheduler))

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=min(batch_size, len(X_train)),
        shuffle=True,
        drop_last=False,
    )
    batches = infinite_batches(loader)

    for _ in range(iterations):
        batch_X, batch_y = next(batches)
        for _lr, model, optimizer, scheduler in heads:
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

    return [(lr, model) for lr, model, _optimizer, _scheduler in heads]


def train_linear_probe(X_train, y_train, lr, iterations=2000, batch_size=64, pos_weight=None,
                       head=LinearClassifier):
    """Train a single head (linear by default) at a fixed learning rate."""
    (_lr, model), = train_heads(
        X_train, y_train, [lr],
        iterations=iterations,
        batch_size=batch_size,
        pos_weight=pos_weight,
        head=head,
    )
    return model


def predict_scores(model, X):
    """Predicted positive-class probabilities for X, as a 1-D numpy array."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).cpu().numpy().ravel()


def safe_auc(y_true, scores):
    """ROC-AUC, or NaN when the evaluation set has a single class."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, scores)


def train_linear_probes(X_train, y_train, X_val, y_val, lr_grid=LR_GRID,
                        iterations=2000, batch_size=64, pos_weight=None, head=LinearClassifier):
    """Sweep the learning rate and keep the head with the best validation AUC."""
    heads = train_heads(
        X_train, y_train, lr_grid,
        iterations=iterations,
        batch_size=batch_size,
        pos_weight=pos_weight,
        head=head,
    )

    best_auc = -np.inf
    best_lr, best_model = None, None

    y_val_np = y_val.cpu().numpy().ravel()
    for lr, model in heads:
        # Sigmoid is monotonic, so ranking by it matches ranking by the logits.
        auc = safe_auc(y_val_np, predict_scores(model, X_val))
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_lr, best_model = lr, model

    if best_model is None:
        # Single-class validation fold: AUC is undefined for every head, so
        # there is nothing to rank on. Fall back to the first head.
        best_lr, best_model = heads[0]
        best_auc = np.nan

    return best_model, best_lr, best_auc


def subsample_pair(len_a, len_b, n_per_side, rng):
    """Row indices of the controls each cohort contributes to a paired comparison.

    Each side is capped at ``n_per_side``, drawn without replacement from
    ``rng``. Shared by ``cohort_separability`` and ``baseline_preds.py``, so the
    distribution-shift measures are computed on exactly the scans the cohort
    probe saw for the same seed.
    """
    idx_a = rng.choice(len_a, size=min(n_per_side, len_a), replace=False)
    idx_b = rng.choice(len_b, size=min(n_per_side, len_b), replace=False)
    return idx_a, idx_b


def cohort_separability(X_a, X_b, n_per_side=150, seed=0, iterations=2000,
                        lr=1e-3, batch_size=64, standardize=True):
    """How separable two cohorts are, as a linear probe's held-out performance.

    ``X_a`` and ``X_b`` are feature matrices from two cohorts. Callers pass
    cognitively-normal scans only, so that disease-related signal -- including
    the cohorts' differing AD prevalence -- cannot be picked up as a cohort cue.

    Each side is capped at ``n_per_side``; a cohort with fewer scans than that
    contributes all of them, and the resulting class imbalance in the training
    half is undone by oversampling the smaller side with repetition, as in
    ``classify_cohort.py``. The probe is the same linear head at a fixed
    learning rate, and its scaler is fit on its own training half, so nothing
    leaks from the held-out half or from the disease probe's scaler.

    Returns ``(auc, balanced_accuracy, n_a, n_b)`` -- with NaN metrics when
    either side is too small to split.
    """
    rng = np.random.default_rng(seed)

    n_a = min(n_per_side, len(X_a))
    n_b = min(n_per_side, len(X_b))
    if n_a < MIN_PER_SIDE or n_b < MIN_PER_SIDE:
        return np.nan, np.nan, n_a, n_b

    idx_a, idx_b = subsample_pair(len(X_a), len(X_b), n_per_side, rng)
    X_a, X_b = X_a[idx_a], X_b[idx_b]

    X = np.concatenate([X_a, X_b]).astype(np.float32)
    y = np.concatenate([np.ones(n_a), np.zeros(n_b)]).astype(np.float32)

    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.5,
        stratify=y,
        random_state=seed,
    )

    X_train_np, X_test_np = X[train_idx], X[test_idx]
    y_train_np, y_test_np = y[train_idx], y[test_idx]

    # Fit on the training half only, so nothing leaks from the held-out half.
    if standardize:
        scaler = StandardScaler().fit(X_train_np)
        X_train_np = scaler.transform(X_train_np).astype(np.float32)
        X_test_np = scaler.transform(X_test_np).astype(np.float32)

    # Oversample the smaller cohort with repetition to match the larger one.
    pos = np.flatnonzero(y_train_np == 1)
    neg = np.flatnonzero(y_train_np == 0)
    if len(pos) != len(neg) and min(len(pos), len(neg)) > 0:
        minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
        minority = rng.choice(minority, size=len(majority), replace=True)
        keep = np.concatenate([majority, minority])
        X_train_np, y_train_np = X_train_np[keep], y_train_np[keep]

    perm = rng.permutation(len(X_train_np))
    X_train_np, y_train_np = X_train_np[perm], y_train_np[perm]

    # The head's initialisation and its batch order come from the global torch
    # RNG, so without this the result would depend on how many probes happened
    # to run before it. Seed it from ``seed`` and put the global stream back
    # afterwards, so the caller's own probes are unaffected by this function.
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        model = train_linear_probe(
            torch.from_numpy(X_train_np).to(DEVICE),
            torch.from_numpy(y_train_np).to(DEVICE).unsqueeze(1),
            lr=lr,
            iterations=iterations,
            batch_size=batch_size,
        )
        scores = predict_scores(model, torch.from_numpy(X_test_np).to(DEVICE))
    finally:
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    preds = (scores > 0.5).astype(np.float32)

    return (
        safe_auc(y_test_np, scores),
        balanced_accuracy_score(y_test_np, preds),
        n_a,
        n_b,
    )
