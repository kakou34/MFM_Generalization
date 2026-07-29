# Cross-Cohort Generalizability of Foundation Models for Alzheimer's Disease Classification


# Abstract
Robustness and generalizability are critical for translating AI systems into clinical practice, yet limited and heterogeneous real-world data hinder their achievement. Medical foundation models (FMs) have emerged as a promising solution for enhanced generalizability. Adapting FMs for downstream tasks requires less annotated data than training models from scratch. However, recent findings indicate that FM-extracted features may encode cohort-specific information, potentially biasing predictions toward the characteristics of the cohort used for adapting FMs for the downstream task. We hypothesize that feature-level differences between the adaptation and target cohorts have an impact on FM performance on the downstream task of Alzheimer’s disease (AD) classification. We used T1-weighted MRI scans from six dementia cohorts and five FMs. For each pair of cohorts, we quantified cohort separability by training linear classifiers to distinguish cohorts based on FM-derived features. Separately, we assessed AD classification generalizability by training an AD-versus-control classifier on one cohort and evaluating it on the remaining cohorts as external test sets. A linear mixed-effects model was then used to examine whether larger cohort differences between the adaptation and target cohorts were associated with lower AD classification performance. Larger feature-level differences between the adaptation and test cohorts were significantly associated with lower AD classification performance on external cohorts (β=−0.16, p<0.001). This association remained significant after adjusting for cohort size, scanner manufacturer, and magnetic field strength distributions. Qualitatively, FM-derived features clustered primarily by cohort rather than by diagnostic class, indicating greater sensitivity to cohort-related variation than to disease-related differences. Larger feature-level cohort differences have a negative impact on FM generalizability in AD classification. Assessing these differences can help identify when FMs require additional adaptation before deployment in new clinical settings. 

## Cohorts and Foundation Models

Six T1-weighted MRI dementia cohorts are used: **ADNI**, **OASIS-3**, **OASIS-4**, **AIBL**, **BrainLAT**, and **Nigerian Brain**.

Five imaging foundation models are evaluated as frozen feature extractors:

| Foundation model | Script | Notes |
|---|---|---|
| [3DINO](https://github.com/AICONSlab/3DINO) | `src/feature_extraction/3dino.py` | Included as a git submodule (`3DINO/`) |
| BrainIAC | `src/feature_extraction/brainiac.py` | ViT-B/16 backbone, SimCLR-pretrained checkpoint |
| SAM-Med3D | `src/feature_extraction/sammed.py` | Via the `medim` package |
| MedSAM2 | `src/feature_extraction/medsam2.py` | 2D slice-wise encoder, video predictor variant |
| Medical-SAM2 | `src/feature_extraction/medicalsam2.py` | 2D slice-wise encoder, image predictor variant |

## Repository structure

```
.
├── 3DINO/                              # 3DINO submodule (see .gitmodules)
├── requirements.txt                    # Python dependencies
├── src/
│   ├── feature_extraction/             # Frozen FM feature extraction
│   │   ├── 3dino.py
│   │   ├── brainiac.py
│   │   ├── sammed.py
│   │   ├── medsam2.py
│   │   └── medicalsam2.py
│   ├── adaptation/                     # Classification based on extracted features
│   │   ├── classify_cohort.py          # Pairwise cohort (site) classification
│   │   └── classify_disease.py         # Train-one-cohort / test-all-cohorts AD classification
│   └── analysis/                       # Post-hoc analysis and figures
│       ├── compute_img_diff.py         # Jensen-Shannon divergence of cohort imaging parameters
│       ├── correlate_lmm.rmd           # Linear mixed-effects model (cohort vs. disease BACC)
│       ├── distance.ipynb              # Optimal-transport distance between cohort feature distributions
│       ├── correlate.ipynb             # Scatter plot of cohort separability vs. AD classification performance
│       └── tsne.ipynb                  # t-SNE visualization of FM feature spaces
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

Each foundation model also requires its own pretrained checkpoint (and, for MedSAM2/Medical-SAM2, config files), which are not distributed in this repository and must be obtained separately from the respective model's release.

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

Each script in `src/feature_extraction/` loads a frozen FM, runs it over the volumes listed in the CSV, and writes one row per sample with columns `id`, `label`, `cohort`, and `Feature1..FeatureN` to `--save_path`.

```bash
python src/feature_extraction/3dino.py \
    --csv_path data.csv \
    --root_dir /path/to/images \
    --save_path features_3dino.csv \
    --checkpoint_path /path/to/3dino_checkpoint.pth

python src/feature_extraction/brainiac.py \
    --csv_path data.csv \
    --root_dir /path/to/images \
    --save_path features_brainiac.csv \
    --checkpoint_path /path/to/brainiac_checkpoint.pth

python src/feature_extraction/sammed.py \
    --csv_path data.csv \
    --root_dir /path/to/images \
    --save_path features_sammed.csv

python src/feature_extraction/medsam2.py \
    --csv_path data.csv \
    --root_dir /path/to/images \
    --save_path features_medsam2.csv \
    --checkpoint_path /path/to/medsam2_checkpoint.pth \
    --config_path sam2.1_hiera_t512.yaml

python src/feature_extraction/medicalsam2.py \
    --csv_path data.csv \
    --root_dir /path/to/images \
    --save_path features_medicalsam2.csv \
    --checkpoint_path /path/to/medicalsam2_checkpoint.pth \
    --config_path sam2_hiera_t.yaml
```

### 2. Adaptation / linear probing

**Cohort separability** — for every pair of cohorts, trains a linear classifier to distinguish the two cohorts from FM features (label `0`/cognitively-normal samples only), repeated over multiple random splits:

```bash
python src/adaptation/classify_cohort.py \
    --features_csv features_3dino.csv \
    --output_csv results_cohort_classification.csv \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --num_splits 20
```

**AD classification generalizability** — trains a classifier on each cohort in turn (with hyperparameter search over learning rate/epochs) and evaluates it on every cohort, including the held-out ones as external test sets. Also reports the optimal-transport (Wasserstein) distance between the train/test cognitively-normal feature distributions for each cohort pair:

```bash
python src/adaptation/classify_disease.py \
    --features_csv features_3dino.csv \
    --output_csv results_disease_classification.csv \
    --num_splits 20
```

### 3. Analysis

- `src/analysis/compute_img_diff.py` computes pairwise Jensen-Shannon divergence between cohorts over scanner manufacturer & magnetic field strength & label.
- `src/analysis/distance.ipynb` computes optimal-transport distances between per-cohort feature distributions for each FM.
- `src/analysis/correlate.ipynb` combines cohort-classification and disease-classification results across FMs and correlates cohort separability with AD classification performance by plotting them on a scatter plot.
- `src/analysis/correlate_lmm.rmd` fits the linear mixed-effects model (disease BACC ~ cohort BACC with random effects for model/train/test cohort) reported in the paper, and exports a publication-ready table.
- `src/analysis/tsne.ipynb` produces t-SNE visualizations of the FM feature spaces, colored by cohort and diagnosis.
