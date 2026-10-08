"""Shared paths and data loading for the analysis_v2 scripts.

REPO / OUT can be overridden with the env variables FM_REPO / FM_OUT.
One row of results_final/<condition>/all.csv = one (iteration, FM, train, test); every analysis first averages the
20 iterations of a (FM, train, test) cell, so the models see one row per cell (6 FMs x 30 ordered pairs = 180).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("FM_REPO", Path(__file__).resolve().parents[2]))
DATA = REPO / "data_v2"
RESULTS = DATA / "results_final"
FEATURES = DATA / "features_original"
META_CSV = DATA / "metadata" / "meta_info_used_samples.csv"
OUT = Path(os.environ.get("FM_OUT", REPO / "local" / "output_v2"))

FM_LABELS = {"3dino": "3DINO", "brainiac": "BrainIAC", "sammed": "SAM-Med3D", "medicalsam2": "Medical-SAM2",
             "curia2_cls": "Curia-2", "nnfoundation_vit": "nnFoundation (ViT)"}
FMS = list(FM_LABELS)
POOLED_FMS = [m for m in FMS if m != "3dino"]  # 3DINO has no pooling step
COHORTS = ["ADNI", "AIBL", "BrainLAT", "NigerianBrain", "OASIS3", "OASIS4"]
NB = "NigerianBrain"
COND_LABELS = {"pooling_mean": "Original (mean pooling)", "pooling_max": "Max pooling", "pooling_gem": "GeM pooling",
               "combat_sex": "ComBat (sex)", "combat_sex_age": "ComBat (sex + age)"}

PAIR_COLS = ["auc_cohort", "auc_disease", "auc_disease_internal", "jsd_sex", "wasserstein_age", "jsd_field_strength",
             "jsd_manufacturer"]


def outdir(name: str) -> Path:
    p = OUT / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_cells(condition: str = "pooling_mean", models: list[str] | None = None) -> pd.DataFrame:
    """Mean over the 20 iterations of every (model, train, test) cell; columns are suffixed ``_mean``."""
    df = pd.read_csv(RESULTS / condition / "all.csv")
    if models is not None:
        df = df[df["model"].isin(models)]
    g = df.groupby(["model", "train", "test"], sort=False)
    return g[PAIR_COLS].mean().add_suffix("_mean").reset_index()


def drop_cohort(d: pd.DataFrame, cohort: str = NB) -> pd.DataFrame:
    """Remove a cohort both as source and as target."""
    return d[(d["train"] != cohort) & (d["test"] != cohort)].copy()


def save_table(df: pd.DataFrame, folder: Path, name: str) -> None:
    df.to_csv(folder / f"{name}.csv", index=False)
    try:
        (folder / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n")
    except ImportError:
        pass
