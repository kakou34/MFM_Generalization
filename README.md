# Assessing Cross-Cohort Generalization of Foundation Models for MRI-based Alzheimer's Disease Classification

Six frozen imaging foundation models (FMs) are used as feature extractors on six T1-weighted MRI cohorts: **ADNI**, **OASIS-3**, **OASIS-4**, **AIBL**, **BrainLAT** and **Nigerian Brain**. Linear probes then measure how easily each FM tells the cohorts apart, and how well its AD classifier transfers from one cohort to another.

**Pipeline:** feature extraction → pooling → (optional) ComBat harmonization → linear probing → analysis

## Installation

```bash
git clone --recurse-submodules <this-repo-url>   # 3DINO is a git submodule
cd cross-cohort-fm-finetuning
pip install -r requirements.txt
```

Checkpoints are not included. SAM-Med3D, nnFoundationViT and Curia-2 download theirs from the Hugging Face Hub on first use. Curia-2 is gated, so accept its license on the model page and run `hf auth login` first. For 3DINO, BrainIAC and Medical-SAM2, download the checkpoint from the model's own release.

## Input CSV

| Column | Description |
|---|---|
| `id` | Scan identifier |
| `img` | Path to the T1w volume, relative to `--root_dir` |
| `label` | `0` = cognitively normal, `1` = AD |
| `cohort` | Cohort name (e.g. `ADNI`) |
| `subject_id` *(optional)* | Keeps all scans of a subject in the same split |

## Usage

All commands are run from the repository root.

### 1. Feature extraction

| FM | Script (`src/feature_extraction/`) | Checkpoint arguments |
|---|---|---|
| [3DINO](https://github.com/AICONSlab/3DINO) | `3dino.py` | `--checkpoint_path` |
| BrainIAC | `brainiac.py` | `--checkpoint_path` |
| SAM-Med3D | `sammed.py` | none (auto-download) |
| Medical-SAM2 | `medicalsam2.py` | `--checkpoint_path`, `--config_path` |
| [nnFoundationViT](https://huggingface.co/MIC-DKFZ/nnFoundationViT) | `nnfoundation_vit.py` | `--checkpoint_dir` (auto-download) |
| [Curia-2](https://huggingface.co/raidium/curia-2) | `curia2.py` | `--checkpoint_dir` (auto-download) |

3DINO writes the final feature CSV directly. The other FMs save one raw feature grid per scan (`.npy`) and a metadata CSV, which are pooled in step 2. All of these scripts take the same input and output arguments:

```bash
# 3DINO
python src/feature_extraction/3dino.py --csv_path data.csv --root_dir /path/to/images \
    --save_path features_3dino.csv --checkpoint_path /path/to/3dino.pth

# All other FMs (add the checkpoint arguments from the table above)
python src/feature_extraction/brainiac.py --csv_path data.csv --root_dir /path/to/images \
    --save_dir grids_brainiac/ --metadata_csv grids_brainiac_meta.csv \
    --checkpoint_path /path/to/brainiac.pth
```

For Curia-2, also pass `--cls_save_dir` to save the per-slice CLS tokens, which the paper uses as Curia-2's features.

> **Disk space:** grids are about 100 MB per scan for Medical-SAM2 and 250 MB per scan for Curia-2, which adds up to hundreds of GB per dataset. Check your quota before running these at full scale.

### 2. Pooling

```bash
python src/feature_extraction/pool_grids.py --grid_dir grids_brainiac/ \
    --metadata_csv grids_brainiac_meta.csv \
    --save_dir csvs/features/brainiac --method mean max gem
```

This writes one `<method>.csv` per pooling method (`id, label, cohort, Feature1..N`). For a single method, use `--save_path file.csv` instead. Pooling runs on the CPU, so you can try another method without re-running the FM.

### 3. ComBat harmonization (optional)

```bash
python src/mitigation/combat.py --features_csv csvs/features/brainiac/mean.csv \
    --output_csv csvs/features_combat/brainiac.csv --covariates sex age
```

ComBat removes per-cohort shifts from the features while keeping the effects of sex (and optionally age). The diagnosis label is never used. The output has the same format as the input, so it goes straight into step 4.

### 4. Linear probing

```bash
python src/adaptation/full_pipeline.py \
    --features 3dino=csvs/features/3dino/features.csv brainiac=csvs/features/brainiac/mean.csv \
    --output_csv results/full_pipeline.csv \
    --metadata_csv csvs/metadata/meta_info_used_samples.csv \
    --n_iterations 20
```

Each iteration runs every FM on the same random splits and runs two probes:

- **Cohort classification:** for each pair of cohorts, a linear probe on their cognitively normal scans tries to tell the two apart (`auc_cohort`). This step also measures how far apart the cohorts' features are (Wasserstein, MMD, Fréchet). With `--metadata_csv`, it also compares the cohorts' sex, age, scanner and field strength.
- **Disease classification:** an AD-vs-CN probe is trained on 70% of one cohort. It is tested on the remaining 30% (`auc_disease_internal`) and on every other cohort (`auc_disease`).

The output has one row per (iteration, FM, train cohort, test cohort). The full algorithm is in `src/adaptation/pseudo_code.tex`.

### 5. Analysis

`src/analysis/` reproduces the paper's statistics and figures from the step 4 results stored in `data_v2/results_final/<condition>/all.csv`:

```bash
cd src/analysis
python run_all.py   # needs R (lme4, lmerTest, performance) + rpy2; --skip-features skips the slow t-SNE figures
```

Outputs go to `local/output_v2/`. See `src/analysis/README.md` for what each script does.
