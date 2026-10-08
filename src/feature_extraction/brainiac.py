import os
import random
import numpy as np
from pathlib import Path
import argparse
import torch
import pandas as pd
import torch.nn as nn
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    ResizeWithPadOrCropd,
    NormalizeIntensityd,
)

from monai.networks.nets import ViT

from grid_io import save_grid, write_metadata_csv

# fix random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

### Dataset preparation functions 
def get_full_img_path(row, root_dir: Path) -> pd.Series:
    img_path = row["img"]
    row["img"] = root_dir / img_path
    return row


def prepare_data(csv_path, root_path):
    df_data = pd.read_csv(csv_path)

    data_root_dir = Path(root_path)
    df_data = df_data.apply(get_full_img_path, axis=1, args=(data_root_dir,))
    files = df_data[["id", "img", "label", "cohort"]].to_dict(orient="records")
    
    transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),

            # Resize while preserving aspect ratio
            Resized(
                keys=["img"],
                spatial_size=96,
                mode="trilinear",
                align_corners=False,
                size_mode="longest",
            ),

            # Pad with zeros (or crop if slightly larger)
            ResizeWithPadOrCropd(
                keys=["img"],
                spatial_size=(96, 96, 96),
                mode="constant",  # zero padding
            ),

            NormalizeIntensityd(keys=["img"], nonzero=True),
        ]
    )

    ds = Dataset(data=files, transform=transforms)
    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return loader

### Function to extract the raw feature grids (patch tokens, no CLS token)
def extract_grids(model, loader, save_dir, metadata_csv):
    model.eval()
    model.cuda()  # Move model to GPU if available

    all_metadata = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].cuda()
            ids = batch["id"]
            labels = batch["label"]
            cohorts = batch["cohort"]

            grids = model(imgs)  # shape: [batch_size, num_tokens, hidden_dim]
            grids = grids.cpu().numpy()

            for i in range(grids.shape[0]):
                save_grid(ids[i], grids[i], save_dir)
                all_metadata.append({
                    "id": ids[i],
                    "label": labels[i].item(),
                    "cohort": cohorts[i],
                })

    write_metadata_csv(all_metadata, metadata_csv)


### Model definition
class ViTBackboneNet(nn.Module):
    def __init__(self, simclr_ckpt_path):
        super(ViTBackboneNet, self).__init__()
        
        # Create ViT backbone with same architecture as SimCLR
        self.backbone = ViT(
            in_channels=1,  # For single channel input
            img_size=(96,96,96),  # Adjust this to your input dimensions
            patch_size=(16, 16, 16),
            hidden_size=768,  # Standard for ViT-B
            mlp_dim=3072,
            num_layers=12,
            num_heads=12, 
            save_attn=True,
        )
        
        # Load pretrained weights from SimCLR checkpoint
        ckpt = torch.load(simclr_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        
        # Extract only backbone weights from SimCLR checkpoint
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                # Remove "backbone." prefix
                new_key = key[9:]  # len("backbone.") = 9
                backbone_state_dict[new_key] = value
        
        # Load the backbone weights
        self.backbone.load_state_dict(backbone_state_dict, strict=True)
        print("Backbone weights loaded!!")


    def forward(self, x):
        # Get raw token grid from ViT backbone (no pooling here anymore --
        # pooling is applied downstream by pool_grids.py on the saved grid).
        # features[0] shape: [batch_size, num_tokens, hidden_dim]. The ViT below
        # is built with MONAI's default classification=False, so no CLS token is
        # prepended -- every row is a patch token (6^3 = 216 for 96^3 input at
        # patch 16). Pool these with has_cls=False.
        features = self.backbone(x)
        return features[0]  # Shape: [batch_size, num_tokens, 768]


def main(args):
    print("Preparing data...")
    loader = prepare_data(args.csv_path, args.root_dir)
    print("Data prepared. Number of samples:", len(loader.dataset))

    print("Preparing model...")
    model = ViTBackboneNet(args.checkpoint_path)

    print("Model prepared. Extracting raw feature grids...")
    extract_grids(model, loader, args.save_dir, args.metadata_csv)
    print("Raw feature grids saved to:", args.save_dir)
    print("Metadata saved to:", args.metadata_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract raw BrainIAC feature grids from brain MRI data "
                     "(pool them into feature vectors with pool_grids.py)"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the input CSV file"
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing the data"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the per-sample raw feature grid .npy files"
    )

    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="Path to save the id/label/cohort metadata CSV"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the BrainIAC checkpoint file"
    )

    args = parser.parse_args()
    main(args)