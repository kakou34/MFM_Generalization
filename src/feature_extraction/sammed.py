import os
import random
import numpy as np
from pathlib import Path
import copy
import argparse
import torch
import medim
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
                spatial_size=128,
                mode="trilinear",
                align_corners=False,
                size_mode="longest",
            ),

            # Pad with zeros (or crop if slightly larger)
            ResizeWithPadOrCropd(
                keys=["img"],
                spatial_size=(128, 128, 128),
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


### Model preparation 
class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        ckpt_path = (
           "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
        )
        # or you can use the local path like: ckpt_path = "./ckpt/sam_med3d_turbo.pth"
        model = medim.create_model(
            "SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path
        )

        self.enc = copy.deepcopy(model.image_encoder)
        del model

    def forward(self, x):
        # No pooling here anymore -- pooling is applied downstream by
        # pool_grids.py on the saved grid.
        x = self.enc(x)  # Output shape: [batch_size, 384, 8, 8, 8]
        # Flatten to a channel-last token grid: [batch_size, 8*8*8, 384]
        b, c = x.shape[0], x.shape[1]
        x = x.reshape(b, c, -1).permute(0, 2, 1)  # [batch_size, 512, 384]
        return x


### Function to extract the raw feature grids
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

            grids = model(imgs)  # shape: [batch_size, 512, 384]
            grids = grids.cpu().numpy()

            for i in range(grids.shape[0]):
                save_grid(ids[i], grids[i], save_dir)
                all_metadata.append({
                    "id": ids[i],
                    "label": labels[i].item(),
                    "cohort": cohorts[i],
                })

    write_metadata_csv(all_metadata, metadata_csv)


def main(args):
    print("Preparing data...")
    loader = prepare_data(args.csv_path, args.root_dir)
    print("Data prepared. Number of samples:", len(loader.dataset))

    print("Preparing model...")
    model = SAM()

    print("Model prepared. Extracting raw feature grids...")
    extract_grids(model, loader, args.save_dir, args.metadata_csv)
    print("Raw feature grids saved to:", args.save_dir)
    print("Metadata saved to:", args.metadata_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract raw SAM-Med3D feature grids from brain MRI data "
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

    args = parser.parse_args()
    main(args)
