"""
Pool raw feature grids (saved by brainiac.py / sammed.py / medsam2.py /
medicalsam2.py) into a final per-sample feature vector CSV.

This is the second stage of the two-stage feature-extraction workflow: it
runs entirely on CPU (no model, no GPU needed) and can be re-run cheaply for
every pooling method without repeating the expensive FM forward pass. Output
schema matches the original single-stage scripts exactly:
``id,label,cohort,Feature1..FeatureN``, so it's a drop-in for
``src/adaptation/classify_cohort.py`` / ``classify_disease.py``.

Several pooling methods can be requested in one run. Each grid is then read
from disk exactly once and every method is applied to it before moving on,
which matters for the SAM2 models whose grids are ~99 MB per sample (~300 GB
per cohort): re-running the script once per method would re-read all of it
each time.

Usage:
    # one method, explicit output path
    python src/feature_extraction/pool_grids.py \\
        --grid_dir /path/to/grids \\
        --metadata_csv /path/to/metadata.csv \\
        --save_path features_brainiac_mean.csv \\
        --method mean

    # all methods in a single pass, written to <save_dir>/<method>.csv
    python src/feature_extraction/pool_grids.py \\
        --grid_dir /path/to/grids \\
        --metadata_csv /path/to/metadata.csv \\
        --save_dir csvs/features/brainiac \\
        --method mean max gem attention --gem_p 3.0
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from grid_io import load_grid
from pooling import POOL_FNS


def pool_grids(grid_dir, metadata_csv, save_paths, methods, has_cls_token, gem_p):
    """Pool every grid listed in ``metadata_csv`` with each of ``methods``.

    Args:
        save_paths: mapping ``method -> output CSV path``.
        methods: pooling method names (keys of ``POOL_FNS``).
    """
    df_meta = pd.read_csv(metadata_csv)

    pool_kwargs = {}
    for method in methods:
        kwargs = {"has_cls": has_cls_token}
        if method == "gem":
            kwargs["p"] = gem_p
        pool_kwargs[method] = kwargs

    n = len(df_meta)
    all_features = {method: [] for method in methods}
    for i, row in enumerate(df_meta.itertuples(index=False)):
        if i % 100 == 0:
            print(f"Pooling ({', '.join(methods)}): {i}/{n}", flush=True)

        # Read each grid once and reuse it for every requested method.
        grid = torch.from_numpy(load_grid(row.id, grid_dir))

        for method in methods:
            feats = POOL_FNS[method](grid, **pool_kwargs[method]).numpy()

            record = {
                "id": row.id,
                "label": row.label,
                "cohort": row.cohort,
            }
            record.update({f"Feature{j + 1}": feats[j] for j in range(feats.shape[0])})
            all_features[method].append(record)

    for method in methods:
        save_path = Path(save_paths[method])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_features[method]).to_csv(save_path, index=False)
        print(f"Pooled features ({method}) saved to: {save_path}")


def main(args):
    methods = list(dict.fromkeys(args.method))  # de-duplicate, keep order

    if args.save_dir is not None:
        save_paths = {m: Path(args.save_dir) / f"{m}.csv" for m in methods}
    else:
        save_paths = {methods[0]: args.save_path}

    print("Pooling grids from:", args.grid_dir)
    pool_grids(
        grid_dir=args.grid_dir,
        metadata_csv=args.metadata_csv,
        save_paths=save_paths,
        methods=methods,
        has_cls_token=args.has_cls_token,
        gem_p=args.gem_p,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pool raw feature grids into a final feature-vector CSV"
    )

    parser.add_argument(
        "--grid_dir",
        type=str,
        required=True,
        help="Directory of per-sample .npy raw feature grids",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="Metadata CSV (id,label,cohort) written by the grid-extraction script",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the pooled features CSV (single --method only)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save one <method>.csv per requested pooling method",
    )
    parser.add_argument(
        "--method",
        type=str,
        nargs="+",
        required=True,
        choices=sorted(POOL_FNS.keys()),
        help="Pooling method(s) to apply; several are applied in a single pass "
             "over the grids (requires --save_dir)",
    )
    parser.add_argument(
        "--has_cls_token",
        action="store_true",
        help="Set for grids that include a CLS token at row 0. None of the 4 "
             "current extractors produce one (BrainIAC's MONAI ViT is built "
             "with classification=False), so leave this unset for them.",
    )
    parser.add_argument(
        "--gem_p",
        type=float,
        default=3.0,
        help="Power p for GeM pooling (only used when 'gem' is among --method)",
    )

    args = parser.parse_args()

    if (args.save_path is None) == (args.save_dir is None):
        parser.error("provide exactly one of --save_path or --save_dir")
    if args.save_path is not None and len(set(args.method)) > 1:
        parser.error("--save_path takes a single --method; use --save_dir for several")

    main(args)
