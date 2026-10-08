"""
Extract raw Curia-2 feature grids from brain MRI data (pool them into feature
vectors with pool_grids.py).

Curia-2 (arXiv:2604.01987, https://huggingface.co/raidium/curia-2) is a 2D
radiology foundation model: a DINOv2 ViT-B/16 (12 layers, dim 768, SwiGLU MLP,
LayerScale, 1 input channel) taking 512x512 slices. It is distributed as a
transformers Dinov2Model, but neither transformers nor safetensors is needed
here: config.json and model.safetensors are downloaded from the Hugging Face Hub
on first use (the repo is gated: accept its license on the model page and log
in with `hf auth login` first), the safetensors file is read with a minimal
reader, and the encoder is re-implemented below in plain PyTorch with
transformers' module names, so the released weights load with strict=True.

Preprocessing follows the model's own CuriaImageProcessor on a volume: every
slice is resized to 512x512 (bicubic, antialiased), then the whole stack of
slices is z-scored at once. Before that, each volume is reoriented (nibabel) so
that its slices along --plane are in the orientation Curia-2 expects (rows,
columns): PL for axial, IL for coronal, IP for sagittal. Three deliberate
departures from the processor:
  * Only the slab of slices that contain brain (nonzero voxels) is kept, so
    empty slices add no tokens.
  * Each slice is center padded with 0 (the skull-stripped background) to a
    square before the resize, instead of being stretched: the 233x197 axial
    MNI slices would be stretched by 2.2 and 2.6.
  * Intensities stay float32. The processor casts numpy input to int16 first,
    which truncates the fractional part of these float MRI intensities.
Its clip_below_air step (clamp at -1000 HU) is a no-op for MRI and is skipped.

Output: one [n_slices * 32^2, 768] grid per scan (final-LayerNorm patch tokens
of every kept slice, slice-major, no CLS token -- pool with has_cls=False),
~1.6 MB per slice at float16, i.e. ~250 MB per scan for the ~155-160 axial
brain slices of the 1 mm MNI scans. With --cls_save_dir, each scan's per-slice
CLS tokens are saved there too, as a second [n_slices, 768] grid (also pooled
with has_cls=False).
"""
import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
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
HF_REPO_ID = "raidium/curia-2"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"


def get_checkpoint(checkpoint_dir) -> Path:
    """Return ``checkpoint_dir`` once it holds ``config.json`` and
    ``model.safetensors``, downloading any missing one from the (gated) Hugging
    Face Hub repo first (~345 MB)."""
    checkpoint_dir = Path(checkpoint_dir)
    for name in (CONFIG_NAME, WEIGHTS_NAME):
        if not (checkpoint_dir / name).is_file():
            print(f"Downloading {HF_REPO_ID}/{name} to {checkpoint_dir}...")
            hf_hub_download(HF_REPO_ID, name, local_dir=checkpoint_dir)
    return checkpoint_dir


SAFETENSORS_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


def load_safetensors(path):
    """State dict of a .safetensors file: an 8-byte little-endian header length,
    a JSON header mapping each tensor name to its dtype, shape and byte range,
    then the raw tensor bytes."""
    data = bytearray(Path(path).read_bytes())  # writable, as torch.frombuffer wants
    header_len = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + header_len])
    header.pop("__metadata__", None)
    state_dict = {}
    for name, info in header.items():
        dtype = SAFETENSORS_DTYPES[info["dtype"]]
        start, end = info["data_offsets"]
        tensor = torch.frombuffer(data, dtype=dtype, count=(end - start) // dtype.itemsize,
                                  offset=8 + header_len + start)
        state_dict[name] = tensor.reshape(info["shape"])
    return state_dict

# -----------------------------------------------------------------------------
# Curia-2 encoder
# -----------------------------------------------------------------------------
# Plain-PyTorch port of transformers' Dinov2Model (v4.57, the version Curia-2
# was saved with) in eval mode, with use_swiglu_ffn=True as in Curia-2's
# config. The input size is fixed to the pretraining one (512x512), so the
# position embeddings are used as they are (transformers only interpolates
# them for other sizes). The mask token, only used in masked-image pretraining,
# is not built.


class PatchEmbeddings(nn.Module):
    def __init__(self, num_channels, hidden_size, patch_size):
        super().__init__()
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.projection(x).flatten(2).transpose(1, 2)  # (B, num_patches, C), row-major patches


class Embeddings(nn.Module):
    def __init__(self, num_channels, hidden_size, patch_size, image_size):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.patch_embeddings = PatchEmbeddings(num_channels, hidden_size, patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return x + self.position_embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_bias):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.key = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.value = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = (
            proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # (B, heads, N, head_dim)
            for proj in (self.query, self.key, self.value)
        )
        x = F.scaled_dot_product_attention(q, k, v)  # scale 1/sqrt(head_dim), as transformers
        return x.transpose(1, 2).reshape(B, N, C)


class SelfOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.dense(x)


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_bias):
        super().__init__()
        self.attention = SelfAttention(hidden_size, num_heads, qkv_bias)
        self.output = SelfOutput(hidden_size)

    def forward(self, x):
        return self.output(self.attention(x))


class LayerScale(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return x * self.lambda1


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, mlp_ratio):
        super().__init__()
        hidden_features = (int(int(hidden_size * mlp_ratio) * 2 / 3) + 7) // 8 * 8
        self.weights_in = nn.Linear(hidden_size, 2 * hidden_features)
        self.weights_out = nn.Linear(hidden_features, hidden_size)

    def forward(self, x):
        x1, x2 = self.weights_in(x).chunk(2, dim=-1)
        return self.weights_out(F.silu(x1) * x2)


class Layer(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, qkv_bias, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.attention = Attention(hidden_size, num_heads, qkv_bias)
        self.layer_scale1 = LayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = SwiGLUFFN(hidden_size, mlp_ratio)
        self.layer_scale2 = LayerScale(hidden_size)

    def forward(self, x):
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, depth, **layer_kwargs):
        super().__init__()
        self.layer = nn.ModuleList(Layer(**layer_kwargs) for _ in range(depth))

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class Curia2Encoder(nn.Module):
    """
    Input : (B, 1, 512, 512) -- B slices
    Output: (B, 1 + 32^2, 768) -- final-LayerNorm CLS token (row 0) followed by
             the row-major patch tokens, i.e. transformers' last_hidden_state.
             No pooling here -- pooling is applied downstream by pool_grids.py
             on the saved grid.
    """
    def __init__(self, num_channels, hidden_size, depth, num_heads, mlp_ratio,
                 patch_size, image_size, qkv_bias, layer_norm_eps):
        super().__init__()
        self.image_size = image_size
        self.embeddings = Embeddings(num_channels, hidden_size, patch_size, image_size)
        self.encoder = Encoder(depth, hidden_size=hidden_size, num_heads=num_heads,
                               mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, eps=layer_norm_eps)
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, x):
        return self.layernorm(self.encoder(self.embeddings(x)))


def load_encoder(checkpoint_dir):
    """Build the encoder from the checkpoint's own config.json and load its weights."""
    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / CONFIG_NAME).read_text())
    if config["model_type"] != "dinov2" or not config["use_swiglu_ffn"]:
        raise ValueError(f"Expected a SwiGLU Dinov2Model config, got {config}")
    model = Curia2Encoder(
        num_channels=config["num_channels"],
        hidden_size=config["hidden_size"],
        depth=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        mlp_ratio=config["mlp_ratio"],
        patch_size=config["patch_size"],
        image_size=config["image_size"],
        qkv_bias=config["qkv_bias"],
        layer_norm_eps=config["layer_norm_eps"],
    )
    state_dict = load_safetensors(checkpoint_dir / WEIGHTS_NAME)
    # Drop the masked-image-modeling mask token; strict loading then guarantees
    # every encoder weight was matched.
    state_dict.pop("embeddings.mask_token", None)
    model.load_state_dict(state_dict, strict=True)
    return model

# -----------------------------------------------------------------------------
# Preprocessing (Curia-2's CuriaImageProcessor, see the module docstring)
# -----------------------------------------------------------------------------
# Array axes (rows, columns, slices) of the reoriented volume, as the nibabel
# axis codes each one increases toward: rows and columns give the 2D slice
# orientation Curia-2 was trained with.
PLANE_AXCODES = {
    "axial": ("P", "L", "S"),
    "coronal": ("I", "L", "A"),
    "sagittal": ("I", "P", "R"),
}


def load_volume(path, plane):
    """(n_slices, H, W) float32 stack of the volume's slices along ``plane``,
    each in Curia-2's orientation, whatever the orientation on disk."""
    img = nib.load(path)
    ornt = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(img.affine),
        nib.orientations.axcodes2ornt(PLANE_AXCODES[plane]),
    )
    volume = img.as_reoriented(ornt).get_fdata(dtype=np.float32)
    return np.ascontiguousarray(volume.transpose(2, 0, 1))


def preprocess_volume(volume, image_size):
    """(n_slices, H, W) array -> (n_brain_slices, 1, image_size, image_size) tensor."""
    # Keep the slab of slices that contain brain.
    brain_slices = np.flatnonzero((volume != 0).any(axis=(1, 2)))
    volume = volume[brain_slices[0]:brain_slices[-1] + 1]

    # Center pad each slice with 0 (the skull-stripped background) to a square.
    _, h, w = volume.shape
    size = max(h, w)
    top, left = (size - h) // 2, (size - w) // 2
    volume = np.pad(volume, ((0, 0), (top, size - h - top), (left, size - w - left)))

    x = torch.from_numpy(volume)[:, None]  # (n_slices, 1, size, size)
    x = F.interpolate(x, size=(image_size, image_size), mode="bicubic",
                      align_corners=False, antialias=True)
    return (x - x.mean()) / x.std()  # one z-score over all slices

# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------
class Curia2Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, root_dir, plane, image_size):
        self.df = pd.read_csv(csv_path)
        self.root_dir = Path(root_dir)
        self.plane = plane
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        volume = load_volume(self.root_dir / row["img"], self.plane)

        return {
            "id": str(row["id"]),
            "label": int(row["label"]),
            "cohort": row["cohort"],
            "img": preprocess_volume(volume, self.image_size),  # (n_slices, 1, 512, 512)
        }

# -----------------------------------------------------------------------------
# Raw feature grid extraction
# -----------------------------------------------------------------------------
AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def extract_grids(model, loader, save_dir, metadata_csv, cls_save_dir, batch_size, precision):
    model.eval()
    model.to(device)
    autocast = torch.autocast(device.type, dtype=AMP_DTYPES.get(precision), enabled=precision != "fp32")

    all_metadata = []
    n = len(loader.dataset)

    with torch.no_grad(), autocast:
        for step, sample in enumerate(loader):  # one scan per step
            if step % 25 == 0:
                print(f"Extracting: {step}/{n}", flush=True)

            slices = sample["img"]  # (n_slices, 1, 512, 512)
            tokens = torch.cat([
                model(slices[i:i + batch_size].to(device)).half()
                for i in range(0, len(slices), batch_size)
            ])  # (n_slices, 1 + 1024, 768)
            if not torch.isfinite(tokens).all():
                raise RuntimeError(f"Non-finite features for {sample['id']} at --precision {precision}")
            tokens = tokens.cpu().numpy()

            save_grid(sample["id"], tokens[:, 1:].reshape(-1, tokens.shape[-1]), save_dir)  # (n_slices * 1024, 768)
            if cls_save_dir is not None:
                save_grid(sample["id"], tokens[:, 0], cls_save_dir)  # (n_slices, 768)
            all_metadata.append({
                "id": sample["id"],
                "label": sample["label"],
                "cohort": sample["cohort"],
            })

    write_metadata_csv(all_metadata, metadata_csv)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    checkpoint_dir = get_checkpoint(args.checkpoint_dir)

    print("Loading Curia-2 encoder...")
    model = load_encoder(checkpoint_dir)

    print("Preparing dataset...")
    dataset = Curia2Dataset(args.csv_path, args.root_dir, args.plane, model.image_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,  # one scan (its stack of slices) at a time
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print("Number of samples:", len(dataset))

    print("Extracting raw feature grids...")
    extract_grids(model, loader, args.save_dir, args.metadata_csv, args.cls_save_dir,
                  args.batch_size, args.precision)
    print("Raw feature grids saved to:", args.save_dir)
    if args.cls_save_dir is not None:
        print("Per-slice CLS token grids saved to:", args.cls_save_dir)
    print("Metadata saved to:", args.metadata_csv)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract raw Curia-2 feature grids from brain MRI data "
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
                         help=f"Directory holding {CONFIG_NAME} and {WEIGHTS_NAME}; downloaded "
                              f"from huggingface.co/{HF_REPO_ID} (gated) into it if missing")
    parser.add_argument("--cls_save_dir", type=str, default=None,
                         help="Optional directory to also save each sample's per-slice CLS "
                              "tokens, as a second [n_slices, 768] grid")
    parser.add_argument("--plane", type=str, default="axial", choices=sorted(PLANE_AXCODES),
                         help="Slicing plane of the 2D slices fed to the model")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16", "fp16"],
                         help="Autocast precision of the forward pass. fp32 is what the "
                              "released model runs in")
    parser.add_argument("--batch_size", type=int, default=64,
                         help="Number of slices per forward pass")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
