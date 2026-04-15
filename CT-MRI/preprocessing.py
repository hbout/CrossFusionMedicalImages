import os
from glob import glob
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
import random
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch

# ---------------------- Normalisation ----------------------
class Normalize01:
    """Normalisation min-max [0,1]"""
    def __call__(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

class NormalizeZScore:
    """Normalisation z-score suivie d'un min-max"""
    def __call__(self, x):
        x = (x - x.mean()) / (x.std() + 1e-8)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return x

# ---------------------- Prétraitement ----------------------
class FusionPreprocessor:
    def __init__(self, img_size=256, patch_size=32, augment=False):
        self.img_size = img_size
        self.patch_size = patch_size
        self.augment = augment
        self.resize = T.Resize((img_size, img_size))
        self.to_tensor = T.ToTensor()
        self.normalize_ct = Normalize01()
        self.normalize_irm = NormalizeZScore()

    def apply_clahe(self, img):
        """Appliquer CLAHE sur PIL.Image en niveau de gris"""
        if not isinstance(img, Image.Image):
            raise ValueError("img doit être un PIL.Image")
        
        img_np = np.array(img)  # doit être uint8 0-255
        if img_np.dtype != np.uint8:
            img_np = (img_np / img_np.max() * 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_np = clahe.apply(img_np)
        return Image.fromarray(img_np)

    def random_transform(self, img):
        if self.augment:
            if random.random() > 0.5:
                img = T.functional.hflip(img)
            if random.random() > 0.5:
                img = T.functional.vflip(img)
            angle = random.uniform(-10, 10)
            img = T.functional.rotate(img, angle)
        return img

    def preprocess_ct(self, img):
        img = self.resize(img)
        img = self.to_tensor(img)
        img = self.normalize_ct(img)
        return img

    def preprocess_irm(self, img):
        img = self.resize(img)
        img = self.apply_clahe(img)          # CLAHE avant conversion tensor
        img = self.to_tensor(img)            # Convertir en tensor float [0,1]
        img = self.normalize_irm(img)        # Z-score puis min-max
        return img

    def __call__(self, irm_img, ct_img):
        # Transformation aléatoire identique pour cohérence
        if self.augment:
            seed = random.randint(0, 99999)
            random.seed(seed)
            irm_img = self.random_transform(irm_img)
            random.seed(seed)
            ct_img = self.random_transform(ct_img)

        # Prétraitement
        ct_tensor = self.preprocess_ct(ct_img)
        irm_tensor = self.preprocess_irm(irm_img)

        # Ajustement dimensions multiples de patch_size
        H, W = ct_tensor.shape[1:]
        H_new = (H // self.patch_size) * self.patch_size
        W_new = (W // self.patch_size) * self.patch_size
        ct_tensor = ct_tensor[:, :H_new, :W_new]
        irm_tensor = irm_tensor[:, :H_new, :W_new]

        return irm_tensor, ct_tensor

