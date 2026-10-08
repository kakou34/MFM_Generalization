# Cross-Cohort Generalizability of Foundation Models for Alzheimer's Disease Classification


# Abstract
Robustness and generalizability are critical for translating AI systems into clinical practice, yet limited and heterogeneous real-world data hinder their achievement. Medical foundation models (FMs) have emerged as a promising solution for enhanced generalizability. Adapting FMs for downstream tasks requires less annotated data than training models from scratch. However, recent findings indicate that FM-extracted features may encode cohort-specific information, potentially biasing predictions toward the characteristics of the cohort used for adapting FMs for the downstream task. We hypothesize that feature-level differences between the adaptation and target cohorts have an impact on FM performance on the downstream task of Alzheimer’s disease (AD) classification. We used T1-weighted MRI scans from six dementia cohorts and five FMs. For each pair of cohorts, we quantified cohort separability by training linear classifiers to distinguish cohorts based on FM-derived features. Separately, we assessed AD classification generalizability by training an AD-versus-control classifier on one cohort and evaluating it on the remaining cohorts as external test sets. A linear mixed-effects model was then used to examine whether larger cohort differences between the adaptation and target cohorts were associated with lower AD classification performance. Larger feature-level differences between the adaptation and test cohorts were significantly associated with lower AD classification performance on external cohorts (β=−0.16, p<0.001). This association remained significant after adjusting for cohort size, scanner manufacturer, and magnetic field strength distributions. Qualitatively, FM-derived features clustered primarily by cohort rather than by diagnostic class, indicating greater sensitivity to cohort-related variation than to disease-related differences. Larger feature-level cohort differences have a negative impact on FM generalizability in AD classification. Assessing these differences can help identify when FMs require additional adaptation before deployment in new clinical settings. 

## Cohorts and Foundation Models

Six T1-weighted MRI dementia cohorts are used: **ADNI**, **OASIS-3**, **OASIS-4**, **AIBL**, **BrainLAT**, and **Nigerian Brain**.

Six imaging foundation models are evaluated as frozen feature extractors:

| Foundation model | Script | Notes |
|---|---|---|
| [3DINO](https://github.com/AICONSlab/3DINO) | `src/feature_extraction/3dino.py` | Included as a git submodule (`3DINO/`). Single-stage: the model's own `forward()` already returns a pooled (CLS-token) vector, so this script writes the final feature CSV directly. |
| BrainIAC | `src/feature_extraction/brainiac.py` | ViT-B/16 backbone, SimCLR-pretrained checkpoint. Two-stage (see below): saves the raw token grid (216 patch tokens; the ViT is built with `classification=False`, so there is no CLS token). |
| SAM-Med3D | `src/feature_extraction/sammed.py` | Via the `medim` package. Two-stage: saves the raw `8x8x8x384` encoder grid. |
| Medical-SAM2 | `src/feature_extraction/medicalsam2.py` | 2D slice-wise encoder, image predictor variant. Two-stage: same raw-grid format as MedSAM2. |
| [nnFoundationViT](https://huggingface.co/MIC-DKFZ/nnFoundationViT) | `src/feature_extraction/nnfoundation_vit.py` | Primus / EVA ViT (40 layers, dim 1056, 8³ patches), MAE-pretrained with [nnssl](https://github.com/MIC-DKFZ/nnssl). Self-contained: the encoder is re-implemented in plain PyTorch (no nnssl, timm or dynamic-network-architectures needed) and the checkpoint is downloaded from the Hugging Face Hub on first use. Two-stage: saves the raw `24x24x24x1056` token grid (no CLS token). |
| [Curia-2](https://huggingface.co/raidium/curia-2) | `src/feature_extraction/curia2.py` | 2D radiology FM: DINOv2 ViT-B/16 (1 channel, 512² slices). Self-contained: the encoder is re-implemented in plain PyTorch (no transformers needed) and the checkpoint is downloaded from the (gated) Hugging Face Hub repo on first use. Two-stage: saves the raw per-slice `32x32x768` patch-token grid of every axial brain slice, and optionally the per-slice CLS tokens as a second grid. |

All FMs except 3DINO are "two-stage": the extraction script only runs the frozen backbone and saves the **raw, unpooled feature grid** to disk; a separate, CPU-only `pool_grids.py` step then collapses each grid into the final `Feature1..FeatureN` vector using a chosen pooling method (see [Pooling methods](#2-pooling) below). This lets you try several pooling mechanisms on the same extracted grid without re-running the (expensive) FM forward pass. 3DINO is the exception: `DinoVisionTransformer.forward()` already does CLS-token pooling internally, so there is no raw grid exposed to save — `3dino.py` remains a single-stage script.

## Repository structure

```
.
├── 3DINO/                              # 3DINO submodule (see .gitmodules)
├── requirements.txt                    # Python dependencies
├── src/
│   ├── feature_extraction/             # Frozen FM feature extraction
│   │   ├── 3dino.py                    # Single-stage: image -> pooled feature CSV
│   │   ├── brainiac.py                 # Two-stage: image -> raw feature grid (.npy)
│   │   ├── sammed.py                   #   "
│   │   ├── medicalsam2.py              #   "
│   │   ├── nnfoundation_vit.py         #   "
│   │   ├── curia2.py                   #   "
│   │   ├── grid_io.py                  # Shared raw-grid / metadata-CSV I/O helpers
│   │   ├── pooling.py                  # Pooling methods (mean/max/gem/attention)
│   │   └── pool_grids.py               # Raw feature grid -> pooled feature CSV
│   ├── adaptation/                     # Classification based on extracted features
│   │   ├── linear_probe.py             # Shared linear probe (DINOv2-style head) + paired cohort-separability probe
│   │   ├── classify_cohort.py          # Pairwise cohort (site) classification
│   │   └── classify_disease.py         # Train-one-cohort / test-all-cohorts AD classification
│   └── analysis/                       # Post-hoc analysis and figures
│       ├── compute_img_diff.py         # Jensen-Shannon divergence of cohort imaging parameters
│       ├── correlate_lmm.rmd           # Linear mixed-effects model (cohort vs. disease BACC)
│       ├── distance.ipynb              # Optimal-transport distance between cohort feature distributions
│       ├── correlate.ipynb             # Scatter plot of cohort separability vs. AD classification performance
│       └── tsne.ipynb                  # t-SNE visualization of FM feature spaces
├── slurm/                              # SLURM job scripts (one array task per model)
│   ├── extract_grids_brainiac.sh
│   ├── extract_grids_sammed.sh
│   ├── extract_grids_medicalsam2.sh
│   ├── extract_grids_nnfoundation_vit.sh
│   ├── extract_grids_curia2.sh
│   ├── pool_grids.sh                   # Raw grids -> pooled feature CSVs
│   └── run_adaptation.sh               # Cohort + disease classification for all 5 models
└── results/                            # Classification results written by run_adaptation.sh
```

## Installation

```bash
git clone --recurse-submodules <this-repo-url>
cd cross-cohort-fm-finetuning
pip install -r requirements.txt
```

If the submodule was not fetched automatically:

```bash
git submodule update --init --recursive
```

Each foundation model also requires its own pretrained checkpoint (and, for MedSAM2/Medical-SAM2, config files), which are not distributed in this repository and must be obtained separately from the respective model's release. The exceptions are SAM-Med3D, nnFoundationViT and Curia-2, whose extraction scripts download their checkpoints from the Hugging Face Hub themselves (Curia-2's repo is gated: accept its license on the model page and run `hf auth login` first).

## Data format

All scripts expect a CSV describing the dataset, with (at minimum) the following columns:

| Column | Description |
|---|---|
| `id` | Unique scan/sample identifier |
| `img` | Path to the T1-weighted MRI volume, relative to `--root_dir` |
| `label` | Binary diagnosis label (`0` = cognitively normal, `1` = AD) |
| `cohort` | Cohort/site name (e.g. `ADNI`, `OASIS-3`, ...) |
| `subject_id` *(optional)* | Subject identifier, used for group-aware splitting when a subject has multiple scans |

## Usage

### 1. Feature extraction

**3DINO** (single-stage) loads the frozen FM, runs it over the volumes listed in the CSV, and writes one row per sample with columns `id`, `label`, `cohort`, and `Feature1..FeatureN` directly to `--save_path`:

```bash
python src/feature_extraction/3dino.py \
    --csv_path data.csv \
    --root_dir /path/to/images \
    --save_path features_3dino.csv \
    --checkpoint_path /path/to/3dino_checkpoint.pth
```

**BrainIAC, SAM-Med3D, Medical-SAM2, nnFoundationViT, Curia-2** (two-stage) instead save one raw, unpooled feature grid per sample (`.npy`, under `--save_dir`) plus a metadata CSV (`id`, `label`, `cohort`, under `--metadata_csv`):

```bash
python src/feature_extraction/brainiac.py \
    --csv_path data.csv \
    --root_dir /path/to/images \
    --save_dir grids_brainiac/ \
    --metadata_csv grids_brainiac_meta.csv \
    --checkpoint_path /path/to/brainiac_checkpoint.pth

python src/feature_extraction/sammed.py \
    --csv_path data.csv \
    --root_dir /path/to/images \
    --save_dir grids_sammed/ \
    --metadata_csv grids_sammed_meta.csv

python src/feature_extraction/medicalsam2.py \
    --csv_path data.csv \
    --root_dir /path/to/images \
    --save_dir grids_medicalsam2/ \
    --metadata_csv grids_medicalsam2_meta.csv \
    --checkpoint_path /path/to/medicalsam2_checkpoint.pth \
    --config_path sam2_hiera_t.yaml

python src/feature_extraction/nnfoundation_vit.py \
    --csv_path data.csv \
    --root_dir /path/to/images \
    --save_dir grids_nnfoundation_vit/ \
    --metadata_csv grids_nnfoundation_vit_meta.csv \
    --checkpoint_dir /path/to/checkpoints/nnFoundationViT

python src/feature_extraction/curia2.py \
    --csv_path data.csv \
    --root_dir /path/to/images \
    --save_dir grids_curia2/ \
    --cls_save_dir grids_curia2_cls/ \
    --metadata_csv grids_curia2_meta.csv \
    --checkpoint_dir /path/to/checkpoints/curia2
```

**nnFoundationViT** downloads `checkpoint_final.pth` (~2.7 GB) from [huggingface.co/MIC-DKFZ/nnFoundationViT](https://huggingface.co/MIC-DKFZ/nnFoundationViT) into `--checkpoint_dir` on first use and reuses it afterwards. Its preprocessing follows the model's own pretraining plan rather than the resize-to-a-cube used for BrainIAC/SAM-Med3D: crop to the nonzero bounding box, z-score over the cropped volume, **no resampling**, then center pad (with 0) / crop to 192³, the only input size the encoder accepts. It therefore expects ~1 mm scans in which the brain (nearly) fits in 192 voxels per axis; longer axes are center-cropped (in the cohorts here, the anterior-posterior extent is ~178-195 voxels). Volumes are read with SimpleITK, so the voxel axis order matches what nnssl fed the model during pretraining. The forward pass runs under fp16 autocast by default, the precision nnssl pretrained the model with; `--precision bf16|fp32` are available.

**Curia-2** downloads `config.json` and `model.safetensors` (~345 MB) from [huggingface.co/raidium/curia-2](https://huggingface.co/raidium/curia-2) into `--checkpoint_dir` on first use. It is a 2D model, so each volume is fed as a stack of slices (`--plane axial` by default; `coronal`/`sagittal` also work), reoriented with nibabel to the slice orientation Curia-2 was trained with (PL for axial, IL for coronal, IP for sagittal). Preprocessing follows the model's own `CuriaImageProcessor`: bicubic resize of each slice to 512², then one z-score over the whole stack. There are three deliberate departures from it. Only the slab of slices containing brain is kept. Each slice is zero-padded to a square instead of being stretched. Intensities stay float rather than being cast to int16, which would truncate them. The forward pass runs in fp32 by default, as the released model does; `--precision bf16|fp16` are available. With `--cls_save_dir`, each slice's CLS token is saved too, as a second `[n_slices, 768]` grid that is pooled over slices like any other grid.

Grids are stored as float16 `[N_tokens, C]` arrays (channel-last, all spatial/slice dimensions flattened into the token axis). **Storage caveat:** BrainIAC/SAM-Med3D grids are small (well under 1MB/sample), but Medical-SAM2 grids are per-slice spatial maps (`32x32x256` per slice, all slices kept) — for a ~150-200 slice scan that's roughly 75-100MB/sample even at float16, i.e. ~230-310GB for the 3095-scan `combined_data_2` dataset. Check available scratch quota (`myquota`) before running these two at full scale. nnFoundationViT sits in between: `[13824, 1056]` per scan, ~29MB/sample or ~90GB for `combined_data_2`. Curia-2's patch grids are the largest: `[n_slices * 1024, 768]` for the ~155-160 axial brain slices, ~250MB/sample or ~0.75TB for `combined_data_2` (its CLS grids are under 0.3MB/sample).

On Snellius, `slurm/` holds one ready-to-run job script per model so the six extractions can be launched in parallel:

```bash
sbatch slurm/extract_grids_brainiac.sh
sbatch slurm/extract_grids_sammed.sh
sbatch slurm/extract_grids_medicalsam2.sh
sbatch slurm/extract_grids_nnfoundation_vit.sh
sbatch slurm/extract_grids_curia2.sh
```

Each writes to `fms_grids/<model>/<dataset>/` (one `<id>.npy` per scan plus `metadata.csv`) and loops over the datasets listed in its `DATASETS=(...)` array — currently just `combined_data_2`; add further `csvs/input/<name>.csv` stems there as needed. Logs go to `slurm_output/{output,error}/`. Note that `/scratch-shared` files are removed after ~14 days, so run the pooling step below before then; `/scratch-local` is deliberately *not* used because it resolves to a different per-node directory and is wiped at job end, which would hide the grids from a later pooling job.

### 2. Pooling

`pool_grids.py` is the second stage for BrainIAC/SAM-Med3D/Medical-SAM2/nnFoundationViT/Curia-2: a CPU-only script that pools the raw grids saved above into the final `id,label,cohort,Feature1..FeatureN` CSV, so it can be re-run cheaply for every pooling method without repeating the FM forward pass. Available methods (`src/feature_extraction/pooling.py`):

| Method | Description |
|---|---|
| `mean` | Average over all tokens — reproduces the pooling every script used before this change. |
| `max` | Element-wise max over all tokens. |
| `gem` | Generalized-mean pooling, `(mean(clamp(x,eps)**p))**(1/p)` with fixed `--gem_p` (default 3.0). |
| `attention` | Deterministic, parameter-free soft attention: mean-vector-as-query (CLS-token-as-query only if `--has_cls_token` is set, which none of the current models use). |

Leave `--has_cls_token` unset for all 6 models: none of their grids has a CLS token in row 0 (BrainIAC included — its MONAI `ViT` uses the default `classification=False`; nnFoundationViT has neither CLS nor register tokens; Curia-2's patch grid excludes its CLS tokens, and its separate CLS grid holds one per slice, all pooled alike). The flag exists for a future backbone that does, where it would drop row 0 from the spatial pooling and use it as the `attention` query.

Several methods can be requested in one run with `--save_dir`, which writes one `<method>.csv` per method and reads each grid only once — worth using for MedSAM2/Medical-SAM2, whose grids are ~99 MB per scan (~300 GB per model):

```bash
python src/feature_extraction/pool_grids.py \
    --grid_dir grids_brainiac/ \
    --metadata_csv grids_brainiac_meta.csv \
    --save_dir csvs/features/brainiac \
    --method mean max gem attention --gem_p 3.0

python src/feature_extraction/pool_grids.py \
    --grid_dir grids_sammed/ \
    --metadata_csv grids_sammed_meta.csv \
    --save_path features_sammed_gem.csv \
    --method gem --gem_p 3.0
```

The output CSV has the exact same schema as the old single-stage scripts, so it's a drop-in for `src/adaptation/classify_cohort.py` / `classify_disease.py` below.

### 3. Adaptation / linear probing

`src/adaptation/full_pipeline.py` runs the whole adaptation step end to end. It implements Algorithm 1 of `src/adaptation/pseudo_code.tex`. It takes one pooled feature CSV per FM and repeats the procedure below for `--n_iterations` iterations (default 20). Each iteration first draws every random choice: the sampled controls, all splits, the oversampling and the probe seeds. It then runs every FM on those same draws, so within an iteration the models differ only in their features. All splits are group-aware when the CSV has a `subject_id` column.

1. **Cohort classification.** For each unordered cohort pair, up to `--n_max` cognitively-normal scans are sampled per cohort (default 150) and split 50/50, stratified by cohort. The smaller cohort of the training half is oversampled by repetition. A linear probe trained at a fixed learning rate (`--cohort_lr`, default `1e-3`) then separates the two cohorts and is scored on the test half. The same sampled controls also give the feature-distribution shift measures: `wasserstein_dist`, `wasserstein_rel`, `mmd2` and `frechet_dist`. With `--metadata_csv`, the pipeline also computes metadata divergences (scanner manufacturer, field strength, sex, age). `jsd_label` is always computed and compares the AD/CN mix of the two full cohorts.
2. **Disease classification.** Each cohort is split 70/30, stratified by diagnosis (`--val_size`, default 0.3). One AD-vs-CN linear probe is trained on the 70% for each learning rate in `LR_GRID` (`1e-5` to `1e-2`), and the probe with the best AUC on the 30% is kept. It is then evaluated on that 30% (internal validation) and on every other full cohort (external test). Since the 30% fold also selects the learning rate, the internal scores are optimistically biased; the external scores are not.
3. **Output.** Both result matrices are flattened into one row per `(iteration, model, train cohort, test cohort)`.

Both probes use the head from `src/adaptation/linear_probe.py`. It follows the DINOv2 linear-evaluation protocol: a single linear layer on the frozen features, trained for a fixed iteration budget with cosine annealing, but optimized with Adam instead of SGD. By default, features are standardized with a scaler fit on the training split only. Pass `--no_cohort_standardize` or `--no_disease_standardize` to switch this off for one probe.

```bash
cd src/adaptation
python full_pipeline.py \
    --features 3dino=csvs/features/3dino/features.csv \
               brainiac=csvs/features/brainiac/mean.csv \
               sammed=csvs/features/sammed/mean.csv \
    --output_csv results/full_pipeline.csv \
    --metadata_csv csvs/metadata/meta_info_used_samples.csv \
    --n_iterations 20 --n_max 150
```

`--features` takes `name=path` pairs, or bare paths, in which case the parent folder name is used as the model name. All feature CSVs must contain the same scans with the same `cohort` and `label`. If an `id` column is present, the CSVs are aligned on it; otherwise they must already be in the same row order. Other options include `--iterations` (optimizer steps per probe, default 2000), `--cohort_iterations` (a separate budget for the cohort probe), `--batch_size` (default 64), `--n_ot_permutations` (permutations for the null mean subtracted in `wasserstein_rel`, default 20) and `--shift_standardize` (z-score each pair before computing the shift measures).

The output CSV has one row per `(i, model, train, test)` with these columns:

| Columns | Description |
|---|---|
| `auc_cohort`, `bacc_cohort`, `acc_cohort` | Cohort separability of the `train`/`test` pair (symmetric) |
| `auc_disease`, `bacc_disease`, `acc_disease` | AD classification trained on `train`, tested on the full `test` cohort |
| `auc_disease_internal`, `bacc_disease_internal`, `acc_disease_internal` | AD classification on the 30% held-out fold of `train` |
| `disease_lr` | Learning rate selected for the disease probe |
| `wasserstein_dist`, `wasserstein_rel`, `mmd2`, `frechet_dist`, `jsd_label` | Feature and label shift between the two cohorts |
| `jsd_manufacturer`, `jsd_field_strength`, `jsd_sex`, `jsd_age`, `wasserstein_age` | Metadata divergences (only with `--metadata_csv`) |

Accuracies threshold the predicted probability at 0.5. A metric is NaN when its evaluation set contains a single class, or when a cohort has too few scans to split.

`src/adaptation/baseline_preds.py` computes a broader set of shift measures, such as sliced Wasserstein, energy distance, KL estimates, CORAL and MMD permutation p-values. It is CPU-only and trains no probe.

### 4. Analysis

TBD