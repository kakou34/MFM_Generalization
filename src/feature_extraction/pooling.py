"""
Pooling methods used to collapse a raw feature grid (``[N_tokens, C]``) saved
by the feature-extraction scripts into a single per-sample feature vector
(``[C]``).

All models except 3DINO now save their raw grid instead of pooling in
``forward()`` (see ``grid_io.py``); ``pool_grids.py`` applies one of the
functions below to reproduce (``mean``) or replace (``max``/``gem``/
``attention``) the pooling step that used to be hard-coded per model.

None of the 4 current extractors emit a CLS token, so all of them pass
``has_cls=False`` and pool over every row of the grid. In particular BrainIAC
does *not* have one: its MONAI ``ViT`` is built with the default
``classification=False`` (see ``brainiac.py``), so the backbone prepends no
CLS token and its ``[216, 768]`` grid is 6^3 patch tokens only.

The ``has_cls=True`` path is kept for a future backbone that does emit one:
row 0 is then excluded from the spatial aggregation done by
``mean_pool``/``max_pool``/``gem_pool``, and used as the query for
``attention_pool``. Do not set it for the current models -- it would drop a
genuine patch token and make ``attention_pool`` query with an arbitrary patch
instead of the mean.
"""
import torch


def _patch_tokens(grid: torch.Tensor, has_cls: bool) -> torch.Tensor:
    """Return the tokens to spatially pool over (drops the CLS row if present)."""
    return grid[1:] if has_cls else grid


def mean_pool(grid: torch.Tensor, has_cls: bool = False) -> torch.Tensor:
    """Average pooling over tokens. Reproduces the previous per-model baseline."""
    tokens = _patch_tokens(grid, has_cls)
    return tokens.mean(dim=0)


def max_pool(grid: torch.Tensor, has_cls: bool = False) -> torch.Tensor:
    """Element-wise max pooling over tokens."""
    tokens = _patch_tokens(grid, has_cls)
    return tokens.amax(dim=0)


def gem_pool(grid: torch.Tensor, has_cls: bool = False, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Generalized-mean (GeM) pooling: ``(mean(clamp(x, eps) ** p)) ** (1/p)``.

    ``p`` is fixed (not learned), interpolating between mean pooling (p=1)
    and max pooling (p -> inf).
    """
    tokens = _patch_tokens(grid, has_cls)
    pooled = tokens.clamp(min=eps).pow(p).mean(dim=0).pow(1.0 / p)
    return pooled


def attention_pool(grid: torch.Tensor, has_cls: bool = False) -> torch.Tensor:
    """Deterministic, parameter-free soft-attention pooling.

    When a CLS token is available (``has_cls=True``), it is used as the query
    attending over the remaining patch tokens (keys = values = patch tokens).
    Otherwise the mean of all tokens is used as the query, attending over the
    full token set ("mean-as-query" self-attention pooling). No trained
    weights are involved, consistent with the rest of the pipeline being a
    frozen feature extractor.
    """
    if has_cls:
        query = grid[0]
        keys = values = grid[1:]
    else:
        query = grid.mean(dim=0)
        keys = values = grid

    dim = query.shape[-1]
    scores = (keys @ query) / (dim ** 0.5)  # [N]
    weights = torch.softmax(scores, dim=0)  # [N]
    pooled = (weights.unsqueeze(-1) * values).sum(dim=0)
    return pooled


POOL_FNS = {
    "mean": mean_pool,
    "max": max_pool,
    "gem": gem_pool,
    "attention": attention_pool,
}
