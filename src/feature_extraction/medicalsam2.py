import os
import random
import numpy as np
from pathlib import Path
import argparse
import torch
import pandas as pd
import torch.nn as nn
import SimpleITK as sitk
from PIL import Image
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from sam2.build_sam import build_sam2_video_predictor

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Set hydra config paths
# -----------------------------------------------------------------------------
# Hydra needs the *directory* holding the SAM2 yaml configs; --config_path is
# the config filename within it. Defaults to the copy shipped in this repo.
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "local" / "MedSAM2" / "config"
GlobalHydra.instance().clear()

# -----------------------------------------------------------------------------
# Preprocessing functions (MedSAM2 style)
# -----------------------------------------------------------------------------
def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
        
    return resized_array
    
    
def preprocess_volume(volume: np.ndarray, target_size=512):
    """
    volume: np.ndarray (D,H,W) float
    returns: torch.Tensor (D, 3, H, W), float32
    """
    nii_image_data_pre = (volume - np.min(volume))/(np.max(volume)-np.min(volume))*255.0
    nii_image_data_pre = np.uint8(nii_image_data_pre)
    img_resized = resize_grayscale_to_rgb_and_resize(nii_image_data_pre, target_size)
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).float().cuda()
    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std
    return img_resized


def load_nifti(path):
    """
    Returns np.ndarray of shape (D,H,W)
    """
    nii_image = sitk.ReadImage(path)
    nii_image_data = sitk.GetArrayFromImage(nii_image)
    
    return nii_image_data

    

# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------
class MedSAM2Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, root_dir):
        self.df = pd.read_csv(csv_path)
        self.root_dir = Path(root_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["img"]
        volume = load_nifti(img_path)
        volume = preprocess_volume(volume, target_size=512)

        return {
            "id": row["id"],
            "label": row["label"],
            "cohort": row["cohort"],
            "img": volume,  # (D,3,H,W)
        }

# -----------------------------------------------------------------------------
# MedSAM2 backbone wrapper
# -----------------------------------------------------------------------------
class MedSAM2BackboneNet(nn.Module):
    """
    Input : (B, D, 3, H, W)
    Output: (B, D*32*32, 256) -- raw per-slice spatial feature grid, channel-last,
             slices and spatial positions flattened into a single token axis.
             No pooling here anymore -- pooling (over slices and/or space) is
             applied downstream by pool_grids.py on the saved grid.
    """
    def __init__(self, config_path, checkpoint_path, config_dir=DEFAULT_CONFIG_DIR):
        super().__init__()
        with initialize_config_dir(config_dir=str(config_dir), job_name="sam2"):
            predictor = build_sam2_video_predictor(config_path, checkpoint_path)
            self.image_encoder = predictor.image_encoder
            self.image_encoder.eval()
        
        # Freeze encoder
        for p in self.image_encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x: (B, D, 3, H, W)
        B, D, C, H, W = x.shape

        # Flatten slices
        x = x.view(B * D, C, H, W)  # (B*D,3,H,W)

        # Move to correct device/dtype
        device = next(self.image_encoder.parameters()).device
        dtype = next(self.image_encoder.parameters()).dtype
        x = x.to(device=device, dtype=dtype)

        with torch.no_grad():
            out = self.image_encoder(x)

        feats = out["vision_features"]              # (B*D, 256, 32, 32)
        feats = feats.permute(0, 2, 3, 1)            # (B*D, 32, 32, 256) channel-last
        feats = feats.reshape(B, D * 32 * 32, 256)   # (B, D*1024, 256) raw token grid

        return feats.float()

# -----------------------------------------------------------------------------
# Raw feature grid extraction
# -----------------------------------------------------------------------------
def extract_grids(model, loader, save_dir, metadata_csv):
    model.eval()
    model.cuda()

    all_metadata = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].cuda()  # (B,D,3,H,W)
            ids = batch["id"]
            labels = batch["label"]
            cohorts = batch["cohort"]

            grids = model(imgs)          # (B, D*1024, 256)
            grids = grids.cpu().numpy()

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
    print("Preparing dataset...")
    dataset = MedSAM2Dataset(args.csv_path, args.root_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    print("Loading MedSAM2 model...")
    model = MedSAM2BackboneNet(args.config_path, args.checkpoint_path, args.config_dir)

    print("Extracting raw feature grids...")
    extract_grids(model, loader, args.save_dir, args.metadata_csv)
    print("Raw feature grids saved to:", args.save_dir)
    print("Metadata saved to:", args.metadata_csv)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract raw Medical-SAM2 feature grids "
                     "(pool them into feature vectors with pool_grids.py)"
    )
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True,
                         help="Directory to save the per-sample raw feature grid .npy files")
    parser.add_argument("--metadata_csv", type=str, required=True,
                         help="Path to save the id/label/cohort metadata CSV")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="sam2_hiera_t.yaml",
                         help="SAM2 config filename inside --config_dir")
    parser.add_argument("--config_dir", type=str, default=str(DEFAULT_CONFIG_DIR),
                         help="Directory containing the SAM2 yaml configs")
    args = parser.parse_args()
    main(args)
