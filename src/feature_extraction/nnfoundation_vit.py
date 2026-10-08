"""
Extract raw nnFoundationViT feature grids from brain MRI data (pool them into
feature vectors with pool_grids.py).

nnFoundationViT (arXiv:2609.26924, https://huggingface.co/MIC-DKFZ/nnFoundationViT)
is the Primus / EVA ViT of the nnFoundation pair, MAE-pretrained with nnssl
(https://github.com/MIC-DKFZ/nnssl) on 192^3 crops: 40 blocks, embedding dim
1056, 16 heads, 8^3 patches. Neither the nnssl repo nor its timm /
dynamic-network-architectures dependencies are needed here: the checkpoint is
downloaded from the Hugging Face Hub on first use, and the encoder is
re-implemented below in plain PyTorch with the original module names, so the
released weights load with strict=True. The MAE decoder is not built.

Preprocessing follows the model's pretraining plan ("noresample"): crop to the
nonzero bounding box, z-score over the whole cropped volume, no resampling.
The volume is then center padded (with 0) / cropped to 192^3, the only input
size the encoder accepts (its absolute position embedding and RoPE are fixed to
a 24^3 token grid). In the 1 mm skull-stripped scans checked (one per cohort)
the brain's bounding box spans ~145-160 voxels left-right and
inferior-superior and 178-195 anterior-posterior, so at most a few
anterior-posterior voxels get cropped.

Output: one [24^3 = 13824, 1056] grid per scan (final-LayerNorm patch tokens,
no CLS or register token -- pool with has_cls=False), ~29 MB at float16.
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from grid_io import save_grid, write_metadata_csv

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Checkpoint download
# -----------------------------------------------------------------------------
HF_REPO_ID = "MIC-DKFZ/nnFoundationViT"
CHECKPOINT_NAME = "checkpoint_final.pth"


def get_checkpoint(checkpoint_dir) -> Path:
    """Return ``<checkpoint_dir>/checkpoint_final.pth``, downloading it from the
    Hugging Face Hub first if it is not there yet (~2.7 GB)."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / CHECKPOINT_NAME
    if not checkpoint_path.is_file():
        print(f"Downloading {HF_REPO_ID}/{CHECKPOINT_NAME} to {checkpoint_dir}...")
        hf_hub_download(HF_REPO_ID, CHECKPOINT_NAME, local_dir=checkpoint_dir)
    return checkpoint_path

# -----------------------------------------------------------------------------
# nnFoundationViT encoder
# -----------------------------------------------------------------------------
# Plain-PyTorch port of the encoder half of nnssl's EvaMAE, i.e.
# dynamic-network-architectures' PatchEmbed + Eva, whose blocks are timm's
# EvaBlock (separate q/k/v projections without k bias, LayerScale, SwiGLU MLP
# with inner LayerNorm, LayerNorm on the attention output) using timm's
# RotaryEmbeddingCat. All LayerNorms keep torch's default eps=1e-5, as in the
# original (dynamic-network-architectures passes torch.nn.LayerNorm to timm).


def build_rope(grid_shape, head_dim, temperature=10000.0):
    """sin/cos tables of timm's RotaryEmbeddingCat(in_pixels=False) on a 3D grid.

    Each of the 3 axes gets ``head_dim // 6`` frequency bands, each applied to an
    interleaved channel pair, which together cover the full head dimension.
    Returns two ``[prod(grid_shape), head_dim]`` tensors.
    """
    num_bands = head_dim // 6
    bands = 1.0 / temperature ** (torch.arange(num_bands, dtype=torch.float32) / num_bands)
    # "/ s * s" is timm's ref_feat_shape rescaling: an identity up to float
    # rounding at the pretraining grid size, kept so the tables match exactly.
    coords = [torch.arange(s, dtype=torch.float32) / s * s for s in grid_shape]
    grid = torch.stack(torch.meshgrid(coords, indexing="ij"), dim=-1)  # (*grid_shape, 3)
    angles = (grid.unsqueeze(-1) * bands).reshape(-1, 3 * num_bands)
    return angles.sin().repeat_interleave(2, -1), angles.cos().repeat_interleave(2, -1)


def apply_rope(x, sin, cos):
    """timm's apply_rot_embed_cat: rotates channel pairs (x0, x1) -> (-x1, x0)."""
    x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.shape)
    return x * cos + x_rot * sin


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope):
        B, N, C = x.shape
        q, k, v = (
            proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # (B, heads, N, head_dim)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        q = apply_rope(q, *rope).type_as(v)
        k = apply_rope(k, *rope).type_as(v)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(self.norm(x))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1_g = nn.Linear(dim, hidden_dim)
        self.fc1_x = nn.Linear(dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.norm(F.silu(self.fc1_g(x)) * self.fc1_x(x)))


class EvaBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.gamma_1 = nn.Parameter(torch.ones(dim))  # LayerScale
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, mlp_hidden_dim)
        self.gamma_2 = nn.Parameter(torch.ones(dim))

    def forward(self, x, rope):
        x = x + self.gamma_1 * self.attn(self.norm1(x), rope)
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x


class Eva(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, grid_shape):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, math.prod(grid_shape), embed_dim))
        # Not part of the checkpoint: rebuilt on construction, as in timm.
        rope_sin, rope_cos = build_rope(grid_shape, embed_dim // num_heads)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        mlp_hidden_dim = int(embed_dim * 4 * 2 / 3)
        self.blocks = nn.ModuleList(
            EvaBlock(embed_dim, num_heads, mlp_hidden_dim) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.pos_embed
        rope = (self.rope_sin, self.rope_cos)
        for block in self.blocks:
            x = block(x, rope)
        return self.norm(x)


class NnFoundationViTEncoder(nn.Module):
    """
    Input : (B, 1, 192, 192, 192)
    Output: (B, 24^3, 1056) -- raw token grid, channel-last. No pooling here --
             pooling is applied downstream by pool_grids.py on the saved grid.
    """
    def __init__(self, embed_dim, depth, num_heads, patch_size, input_shape):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.down_projection = PatchEmbed(patch_size, embed_dim)
        grid_shape = tuple(s // p for s, p in zip(input_shape, patch_size))
        self.eva = Eva(embed_dim, depth, num_heads, grid_shape)

    def forward(self, x):
        x = self.down_projection(x)       # (B, 1056, 24, 24, 24)
        x = x.flatten(2).transpose(1, 2)  # (B, 24^3, 1056), same token order as nnssl
        return self.eva(x)


def load_encoder(checkpoint_path):
    """Build the encoder from the checkpoint's own architecture plan and load its weights."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    arch = ckpt["nnssl_adaptation_plan"]["architecture_plans"]["arch_kwargs"]
    model = NnFoundationViTEncoder(
        embed_dim=arch["embed_dim"],
        depth=arch["encoder_eva_depth"],
        num_heads=arch["encoder_eva_numheads"],
        patch_size=tuple(arch["patch_embed_size"]),
        input_shape=tuple(arch["input_shape"]),
    )
    # Drop the MAE decoder (decoder.*, up_projection.*, mask_token); strict
    # loading then guarantees every encoder weight was matched.
    encoder_weights = {
        k: v for k, v in ckpt["network_weights"].items()
        if k.startswith(("down_projection.", "eva."))
    }
    model.load_state_dict(encoder_weights, strict=True)
    return model

# -----------------------------------------------------------------------------
# Preprocessing (nnFoundationViT pretraining plan: "noresample")
# -----------------------------------------------------------------------------
def load_volume(path):
    """(D, H, W) float32 array in the voxel order nnssl's SimpleITKIO reads
    (SimpleITK's z, y, x), without any reorientation."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def preprocess_volume(volume, input_shape):
    # Crop to the bounding box of the nonzero voxels, as nnssl's crop_to_nonzero
    # (whose hole filling cannot grow the bounding box, so it is skipped here).
    nonzero = np.argwhere(volume != 0)
    start, stop = nonzero.min(axis=0), nonzero.max(axis=0) + 1
    volume = volume[tuple(slice(a, b) for a, b in zip(start, stop))]

    # ZScoreNormalization over the whole cropped volume (use_mask_for_norm=False).
    volume = (volume - volume.mean()) / max(volume.std(), 1e-8)

    # Center crop axes longer than the input, center pad shorter ones with 0 --
    # what nnssl's pretraining data loader pads normalised patches with.
    out = np.zeros(input_shape, dtype=np.float32)
    src, dst = [], []
    for s, t in zip(volume.shape, input_shape):
        n = min(s, t)
        src.append(slice((s - n) // 2, (s - n) // 2 + n))
        dst.append(slice((t - n) // 2, (t - n) // 2 + n))
    out[tuple(dst)] = volume[tuple(src)]
    return out

# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------
class NnFoundationDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, root_dir, input_shape):
        self.df = pd.read_csv(csv_path)
        self.root_dir = Path(root_dir)
        self.input_shape = input_shape

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        volume = preprocess_volume(load_volume(self.root_dir / row["img"]), self.input_shape)

        return {
            "id": str(row["id"]),
            "label": row["label"],
            "cohort": row["cohort"],
            "img": torch.from_numpy(volume)[None],  # (1, D, H, W)
        }

# -----------------------------------------------------------------------------
# Raw feature grid extraction
# -----------------------------------------------------------------------------
AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def extract_grids(model, loader, save_dir, metadata_csv, precision):
    model.eval()
    model.to(device)
    autocast = torch.autocast(device.type, dtype=AMP_DTYPES.get(precision), enabled=precision != "fp32")

    all_metadata = []
    n = len(loader.dataset)

    with torch.no_grad(), autocast:
        for step, batch in enumerate(loader):
            if step % 25 == 0:
                print(f"Extracting: {len(all_metadata)}/{n}", flush=True)

            imgs = batch["img"].to(device)  # (B, 1, 192, 192, 192)
            ids = batch["id"]
            labels = batch["label"]
            cohorts = batch["cohort"]

            grids = model(imgs).float().cpu().numpy()  # (B, 13824, 1056)
            if not np.isfinite(grids).all():
                raise RuntimeError(f"Non-finite features for {ids} (fp16 overflow?); "
                                   "rerun with --precision bf16 or fp32")

            for i in range(grids.shape[0]):
                save_grid(ids[i], grids[i], save_dir)
                all_metadata.append({
                    "id": ids[i],
                    "label": labels[i].item(),
                    "cohort": cohorts[i],
                })

    write_metadata_csv(all_metadata, metadata_csv)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    checkpoint_path = get_checkpoint(args.checkpoint_dir)

    print("Loading nnFoundationViT encoder...")
    model = load_encoder(checkpoint_path)

    print("Preparing dataset...")
    dataset = NnFoundationDataset(args.csv_path, args.root_dir, model.input_shape)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print("Number of samples:", len(dataset))

    print("Extracting raw feature grids...")
    extract_grids(model, loader, args.save_dir, args.metadata_csv, args.precision)
    print("Raw feature grids saved to:", args.save_dir)
    print("Metadata saved to:", args.metadata_csv)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract raw nnFoundationViT feature grids from brain MRI data "
                     "(pool them into feature vectors with pool_grids.py)"
    )
    parser.add_argument("--csv_path", type=str, required=True,
                         help="Path to the input CSV file")
    parser.add_argument("--root_dir", type=str, required=True,
                         help="Root directory containing the data")
    parser.add_argument("--save_dir", type=str, required=True,
                         help="Directory to save the per-sample raw feature grid .npy files")
    parser.add_argument("--metadata_csv", type=str, required=True,
                         help="Path to save the id/label/cohort metadata CSV")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                         help=f"Directory holding {CHECKPOINT_NAME}; downloaded from "
                              f"huggingface.co/{HF_REPO_ID} into it if missing")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
                         help="Autocast precision of the forward pass. fp16 is what nnssl "
                              "pretrained the model with; fp32 is slower")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
