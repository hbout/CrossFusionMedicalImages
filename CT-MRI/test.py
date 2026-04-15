import os
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
# dataset_full.py
import os
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import sys
from model.fusion_model import FusionModel


class FusionDatasetFull(Dataset):
    def __init__(self, irm_dir, ct_dir, img_size=256, augment=False):
        self.irm_paths = sorted(glob(os.path.join(irm_dir, "*")))
        self.ct_paths  = sorted(glob(os.path.join(ct_dir, "*")))
        assert len(self.irm_paths) == len(self.ct_paths), "IRM et CT doivent avoir le même nombre d'images"

        # Transforms
        base = [T.Resize((img_size, img_size)), T.ToTensor()]
        if augment:
            aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(10)]
            self.transform = T.Compose(aug + base)
        else:
            self.transform = T.Compose(base)

    def __len__(self):
        return len(self.irm_paths)

    def __getitem__(self, idx):
        irm = Image.open(self.irm_paths[idx]).convert("L")
        ct  = Image.open(self.ct_paths[idx]).convert("L")

        irm = self.transform(irm)
        ct  = self.transform(ct)

        filename = os.path.splitext(os.path.basename(self.irm_paths[idx]))[0]
        return irm, ct, filename


def run_inference(model_ckpt, irm_dir, ct_dir, save_dir="./results", img_size=256, batch=1, device=None):
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device", device)

    dataset = FusionDatasetFull(irm_dir, ct_dir, img_size=img_size, augment=False)  # Utilisez FusionDatasetFull
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)

    model = FusionModel().to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()
    print("model checkpoint loaded", model_ckpt)

    with torch.no_grad():
        for irm, ct, filenames in tqdm(loader, desc="inference"):
            irm, ct = irm.to(device), ct.to(device)
            fused, _, _ = model(irm, ct)
            fused = torch.clamp(fused, 0.0, 1.0).cpu()

            for i, fname in enumerate(filenames):
                out_path = os.path.join(save_dir, f"{fname}.png")
                torchvision.utils.save_image(fused[i], out_path)

    print("inference terminee, resultats sauvegardes dans", save_dir)

if __name__ == "__main__":
    ckpt = "./checkpoints/model_epoch_015.pth"
    irm_dir = "./data/test/irm"
    ct_dir = "./data/test/ct"
    save_dir = "test"

    run_inference(ckpt, irm_dir, ct_dir, save_dir=save_dir, img_size=256, batch=1)