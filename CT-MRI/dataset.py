import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset


class FusionDataset(Dataset):
    def __init__(self, irm_dir, ct_dir, img_size=256, patch_size=32, augment=False):
        self.irm_paths = sorted(glob(os.path.join(irm_dir, "*")))
        self.ct_paths = sorted(glob(os.path.join(ct_dir, "*")))
        assert len(self.irm_paths) == len(self.ct_paths), "IRM et CT doivent avoir le même nombre d'images"

        self.img_size = img_size
        self.patch_size = patch_size

    def __len__(self):
        patches_per_img = (self.img_size // self.patch_size) ** 2
        return len(self.irm_paths) * patches_per_img

    def __getitem__(self, idx):
        patches_per_img = (self.img_size // self.patch_size) ** 2
        img_idx = idx // patches_per_img
        patch_idx = idx % patches_per_img

        irm_img = Image.open(self.irm_paths[img_idx]).convert("L")
        ct_img = Image.open(self.ct_paths[img_idx]).convert("L")

        irm_tensor, ct_tensor = self.preproc(irm_img, ct_img)

        # ----------------- Extraction du patch -----------------
        patches_per_row = self.img_size // self.patch_size
        row = patch_idx // patches_per_row
        col = patch_idx % patches_per_row

        irm_patch = irm_tensor[:, row*self.patch_size:(row+1)*self.patch_size,
                               col*self.patch_size:(col+1)*self.patch_size]
        ct_patch = ct_tensor[:, row*self.patch_size:(row+1)*self.patch_size,
                             col*self.patch_size:(col+1)*self.patch_size]

        filename = os.path.basename(self.irm_paths[img_idx]) + f"_r{row}_c{col}"
        return irm_patch, ct_patch, filename
