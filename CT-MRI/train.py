import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric
from PIL import Image
import sys
from .dataset import FusionDataset
from torch.utils.data import Dataset, DataLoader
from .model.fusion_model import FusionModel

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric
from torch.utils.data import Dataset, DataLoader

# ================== GAUSSIAN WINDOW FOR SSIM ==================
def create_gaussian_window(window_size=11, sigma=1.5, channels=1, device='cpu'):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    window = window_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return window

def ssim_map(x, y, window):
    ws = window.shape[-1]
    padding = ws // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=padding, groups=x.shape[1])
    mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=x.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=x.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=x.shape[1]) - mu_xy

    C1, C2 = (0.01)**2, (0.03)**2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2) + 1e-12)
    return ssim_map

def ssim_loss(x, y, window_size=11, sigma=1.5):
    C = x.shape[1]
    window = create_gaussian_window(window_size, sigma, C, x.device)
    s_map = ssim_map(x, y, window)
    return 1.0 - s_map.mean()

# ================== MULTI-SCALE GRADIENT LOSS ==================
sobel_x = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).view(1, 1, 3, 3)
sobel_y = torch.tensor([[-1., -2., -1.],
                        [0., 0., 0.],
                        [1., 2., 1.]]).view(1, 1, 3, 3)

def gradient_map(img):
    device = img.device
    C = img.shape[1]
    kx, ky = sobel_x.to(device).repeat(C, 1, 1, 1), sobel_y.to(device).repeat(C, 1, 1, 1)
    gx, gy = F.conv2d(img, kx, padding=1, groups=C), F.conv2d(img, ky, padding=1, groups=C)
    return torch.sqrt(gx**2 + gy**2 + 1e-8)

def multiscale_edge_loss(fused, ref, scales=(1, 2, 4)):
    loss = 0.0
    for s in scales:
        pf, pr = (fused, ref) if s == 1 else (F.avg_pool2d(fused, s, s), F.avg_pool2d(ref, s, s))
        gf, gr = gradient_map(pf), gradient_map(pr)
        loss += F.l1_loss(gf, gr)
    return loss / len(scales)

# ================== INTENSITY LOSS ==================
def local_variance_map(x, kernel=7):
    pad = kernel // 2
    mean = F.avg_pool2d(x, kernel, 1, pad)
    mean_sq = F.avg_pool2d(x * x, kernel, 1, pad)
    var = (mean_sq - mean**2).clamp(min=0.0)
    return var

def intensity_loss_adaptive(fused, ref):
    var = local_variance_map(ref)
    weight = 1.0 + torch.tanh(var * 10.0)
    return (weight * torch.abs(fused - ref)).mean()

# ================== FEATURE LOSS ==================
def feature_consistency_loss(fused_feat, irm_feat, ct_feat):
    if fused_feat is None:
        return 0.0
    if isinstance(fused_feat, (list, tuple)):
        loss = 0.0
        for ff, fi, fc in zip(fused_feat, irm_feat, ct_feat):
            loss += F.mse_loss(ff, fi) + F.mse_loss(ff, fc)
        return loss / (2 * len(fused_feat))
    return 0.5 * (F.mse_loss(fused_feat, irm_feat) + F.mse_loss(fused_feat, ct_feat))

# ================== FINAL LOSS ==================
def fusion_loss_ct_irm(fused, irm, ct, irm_feat=None, ct_feat=None, fused_feat=None, weights=None):
    if weights is None:
        weights = {'ssim': 0.35, 'intensity': 0.25, 'edge': 0.25, 'feat': 0.15}

    # SSIM Loss
    ssim_total = 0.7 * ssim_loss(fused, irm) + 0.3 * ssim_loss(fused, ct)

    # Intensity Loss
    inten_total = 0.4* intensity_loss_adaptive(fused, irm) + 0.6* intensity_loss_adaptive(fused, ct)

    # Edge Loss
    edge_total = 0.4 * multiscale_edge_loss(fused, irm) + 0.6 * multiscale_edge_loss(fused, ct)

    # Feature Loss
    feat_loss = 0.0
    if fused_feat is not None and irm_feat is not None and ct_feat is not None:
        feat_loss = feature_consistency_loss(fused_feat, irm_feat, ct_feat)

    # Total Loss
    total = (weights['ssim'] * ssim_total +
             weights['intensity'] * inten_total +
             weights['edge'] * edge_total +
             weights['feat'] * feat_loss)
    return total

# ================== METRICS ==================
def compute_metrics_batch(fused, irm, ct):
    ssim_vals, psnr_vals, cc_vals = [], [], []
    for b in range(fused.size(0)):
        f, i, c = fused[b].squeeze().cpu().numpy(), irm[b].squeeze().cpu().numpy(), ct[b].squeeze().cpu().numpy()
        f, i, c = np.clip(f, 0, 1), np.clip(i, 0, 1), np.clip(c, 0, 1)
        try:
            ssim_avg = (ssim_metric(f, i, data_range=1.0) + ssim_metric(f, c, data_range=1.0)) / 2
            psnr_avg = (psnr_metric(f, i, data_range=1.0) + psnr_metric(f, c, data_range=1.0)) / 2
            cc_avg = (np.corrcoef(f.flatten(), i.flatten())[0, 1] + np.corrcoef(f.flatten(), c.flatten())[0, 1]) / 2
        except:
            ssim_avg = psnr_avg = cc_avg = 0
        ssim_vals.append(ssim_avg)
        psnr_vals.append(psnr_avg)
        cc_vals.append(cc_avg)
    return np.mean(ssim_vals), np.mean(psnr_vals), np.mean(cc_vals)

# ================== TRAIN LOOP ==================
def train_ct_irm(train_loader, val_loader, model, device, epochs=30, lr=1e-3, save_dir="./checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for irm, ct, _ in tqdm(train_loader, desc=f"Train E{epoch}"):
            irm, ct = irm.to(device), ct.to(device)
            fused, irm_feat, ct_feat = model(irm, ct)
            loss = fusion_loss_ct_irm(fused, irm, ct, irm_feat, ct_feat, fused)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = val_ssim = val_psnr = val_cc = 0
        with torch.no_grad():
            for irm, ct, _ in val_loader:
                irm, ct = irm.to(device), ct.to(device)
                fused, irm_feat, ct_feat = model(irm, ct)
                loss = fusion_loss_ct_irm(fused, irm, ct, irm_feat, ct_feat, fused)
                val_loss += loss.item()
                s, p, c = compute_metrics_batch(fused, irm, ct)
                val_ssim += s
                val_psnr += p
                val_cc += c

        val_loss /= len(val_loader)

        epoch_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(epoch_dir, f"model_epoch_{epoch:03d}.pth"))

        with open(os.path.join(epoch_dir, "metrics.txt"), "w") as f:
            f.write(f"Epoch: {epoch}\nTrain Loss: {avg_loss:.6f}\nVal Loss: {val_loss:.6f}\n")
          
# ================== MAIN ==================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_dataset = FusionDataset("data/data/train/irm", "data/data/train/ct")
    val_dataset = FusionDataset("data/data/val/irm", "data/data/val/ct")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = FusionModel().to(device)
    train_ct_irm(train_loader, val_loader, model, device, epochs=15, lr=1e-4, save_dir="./checkpoints")
