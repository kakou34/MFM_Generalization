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
CONFIG_DIR = Path("/path/to/MedSAM2/config")
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
    Output: (B, 256)
    """
    def __init__(self, config_path, checkpoint_path):
        super().__init__()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), job_name="sam2"):
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

        feats = out["vision_features"]       # (B*D, 256, 32, 32)
        feats = feats.mean(dim=(-2, -1))     # spatial GAP → (B*D,256)
        feats = feats.view(B, D, -1)         # (B,D,256)
        feats = feats.mean(dim=1)            # slice pooling → (B,256)

        return feats.float()

# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------
def extract_features(model, loader, save_path):
    model.eval()
    model.cuda()

    all_features = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].cuda()  # (B,D,3,H,W)
            ids = batch["id"]
            labels = batch["label"]
            cohorts = batch["cohort"]

            feats = model(imgs)          # (B,256)
            feats = feats.cpu().numpy()

            for i in range(feats.shape[0]):
                record = {
                    "id": ids[i],
                    "label": labels[i].item(),
                    "cohort": cohorts[i],
                }
                record.update({f"Feature{j+1}": feats[i, j] for j in range(feats.shape[1])})
                all_features.append(record)

    df_feats = pd.DataFrame(all_features)
    df_feats.to_csv(save_path, index=False)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    print("Preparing dataset...")
    dataset = MedSAM2Dataset(args.csv_path, args.root_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    print("Loading MedSAM2 model...")
    model = MedSAM2BackboneNet(args.config_path, args.checkpoint_path)

    print("Extracting features...")
    extract_features(model, loader, args.save_path)
    print("Saved features to:", args.save_path)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSAM2 feature extraction")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="sam2_hiera_t.yaml")
    args = parser.parse_args()
    main(args)
