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
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, x):
        x = self.enc(x)  # Output shape: [batch_size, 384, 8, 8, 8]
        x = self.global_avg_pool(x)  # Output shape: [batch_size, 384, 1, 1, 1]
        x = x.squeeze(-1).squeeze(-1).squeeze(-1) # Output shape: [batch_size, 384]
        return x
    

### Function to extract the features
def extract_features(model, loader, save_path):
    model.eval()
    model.cuda()  # Move model to GPU if available

    all_features = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].cuda()
            ids = batch["id"]
            labels = batch["label"]
            cohorts = batch["cohort"]

            feats = model(imgs)  # shape: [batch_size, 384]
            feats = feats.cpu().numpy()

            for i in range(feats.shape[0]):
                record = {
                    "id": ids[i],
                    "label": labels[i].item(),
                    "cohort": cohorts[i],
                }
                record.update({f"Feature{j+1}": feats[i, j] for j in range(384)})
                all_features.append(record)

    df_feats = pd.DataFrame(all_features)
    df_feats.to_csv(save_path, index=False)
    
    
def main(args):
    print("Preparing data...")
    loader = prepare_data(args.csv_path, args.root_dir)
    print("Data prepared. Number of samples:", len(loader.dataset))

    print("Preparing model...")
    model = SAM()

    print("Model prepared. Extracting features...")
    extract_features(model, loader, args.save_path)
    print("Features extracted and saved to:", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract SAM features from brain MRI data"
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
        "--save_path",
        type=str,
        required=True,
        help="Path to save the extracted features CSV"
    )

    args = parser.parse_args()
    main(args)
    