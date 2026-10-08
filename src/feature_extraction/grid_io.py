"""
Shared I/O helpers for the raw-feature-grid extraction / pooling workflow.

The 4 non-DINO feature-extraction scripts (brainiac.py, sammed.py, medsam2.py,
medicalsam2.py) save one raw feature grid per sample as a ``.npy`` file
(instead of directly pooling to a single vector), plus one metadata CSV with
the ``id``, ``label``, ``cohort`` columns. ``pool_grids.py`` then reads both
back to produce the final ``Feature1..FeatureN`` CSV.

Grids are stored as ``[N_tokens, C]`` float16 arrays (channel-last, all
spatial/slice dimensions flattened into the token axis) so that a single
pooling script can operate on any of the 4 models uniformly.
"""
import csv
from pathlib import Path

import numpy as np


def save_grid(sample_id, grid: np.ndarray, save_dir: Path, dtype=np.float16) -> Path:
    """Save a single sample's raw feature grid as ``<save_dir>/<sample_id>.npy``.

    Args:
        sample_id: unique sample identifier (used as the filename stem).
        grid: array of shape ``[N_tokens, C]``.
        save_dir: destination directory (created if missing).
        dtype: on-disk dtype (float16 by default to save space).

    Returns:
        The path the grid was written to.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{sample_id}.npy"
    np.save(out_path, grid.astype(dtype))
    return out_path


def load_grid(sample_id, grid_dir: Path) -> np.ndarray:
    """Load a previously saved grid and cast it back to float32 for pooling math."""
    grid_dir = Path(grid_dir)
    grid_path = grid_dir / f"{sample_id}.npy"
    return np.load(grid_path).astype(np.float32)


def write_metadata_csv(records: list, path: Path) -> None:
    """Write the ``id,label,cohort`` metadata CSV alongside the saved grids."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "label", "cohort"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record[k] for k in fieldnames})
