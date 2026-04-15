import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp  # VIF (Visual Information Fidelity)

# --------------------------
# Dossiers des images
# --------------------------
folder_irm = './data/test/irm'
folder_ct = './data/test/ct'
folder_fused = 'test_results'

# --------------------------
# Vérification des fichiers
# --------------------------
irm_files = sorted(os.listdir(folder_irm))
ct_files = sorted(os.listdir(folder_ct))
fused_files = sorted(os.listdir(folder_fused))

print("IRM files:", irm_files)
print("CT files:", ct_files)
print("Fused files:", fused_files)

# --------------------------
# Fonction pour CC
# --------------------------
def correlation_coefficient(img1, img2):
    img1 = img1.flatten()
    img2 = img2.flatten()
    return np.corrcoef(img1, img2)[0, 1]

# --------------------------
# Liste pour stocker les résultats
# --------------------------
results = []

# --------------------------
# Parcours des fichiers fusionnés
# --------------------------
for f in fused_files:
    path_irm = os.path.join(folder_irm, f)
    path_ct = os.path.join(folder_ct, f)
    path_fused = os.path.join(folder_fused, f)

    if not (os.path.exists(path_irm) and os.path.exists(path_ct)):
        print(f"⚠️ Fichier manquant pour {f}, on saute.")
        continue

    # Lecture images
    img_irm = cv2.imread(path_irm, cv2.IMREAD_GRAYSCALE)
    img_ct = cv2.imread(path_ct, cv2.IMREAD_GRAYSCALE)
    img_fused = cv2.imread(path_fused, cv2.IMREAD_GRAYSCALE)

    if img_irm is None or img_ct is None or img_fused is None:
        print(f"⚠️ Erreur de lecture pour {f}, on saute.")
        continue

    # Conversion float pour VIF
    img_irm_f = img_irm.astype(np.float32)
    img_ct_f = img_ct.astype(np.float32)
    img_fused_f = img_fused.astype(np.float32)

    # Calcul métriques par rapport à IRM
    metrics_irm = {
        'SSIM': ssim(img_irm, img_fused, data_range=img_fused.max() - img_fused.min()),
        'PSNR': psnr(img_irm, img_fused, data_range=img_fused.max() - img_fused.min()),
        'CC': correlation_coefficient(img_irm, img_fused),
        'VIF': vifp(img_irm_f, img_fused_f)
    }

    # Calcul métriques par rapport à CT
    metrics_ct = {
        'SSIM': ssim(img_ct, img_fused, data_range=img_fused.max() - img_fused.min()),
        'PSNR': psnr(img_ct, img_fused, data_range=img_fused.max() - img_fused.min()),
        'CC': correlation_coefficient(img_ct, img_fused),
        'VIF': vifp(img_ct_f, img_fused_f)
    }

    # Moyenne IRM/CT
    avg_metrics = {
        'Image': f,
        'SSIM': (metrics_irm['SSIM'] + metrics_ct['SSIM']) / 2,
        'PSNR': (metrics_irm['PSNR'] + metrics_ct['PSNR']) / 2,
        'CC': (metrics_irm['CC'] + metrics_ct['CC']) / 2,
        'VIF': (metrics_irm['VIF'] + metrics_ct['VIF']) / 2
    }

    results.append(avg_metrics)

    # --------------------------
    # Affichage des détails pour chaque image
    # --------------------------
    print(f"\n Image : {f}")
    print("   ➤ IRM → Fused : SSIM={:.4f}, PSNR={:.4f}, CC={:.4f}, VIF={:.4f}".format(
        metrics_irm['SSIM'], metrics_irm['PSNR'], metrics_irm['CC'], metrics_irm['VIF']))
    print("   ➤ CT  → Fused : SSIM={:.4f}, PSNR={:.4f}, CC={:.4f}, VIF={:.4f}".format(
        metrics_ct['SSIM'], metrics_ct['PSNR'], metrics_ct['CC'], metrics_ct['VIF']))
    print("   ➤ Moyenne IRM/CT : SSIM={:.4f}, PSNR={:.4f}, CC={:.4f}, VIF={:.4f}".format(
        avg_metrics['SSIM'], avg_metrics['PSNR'], avg_metrics['CC'], avg_metrics['VIF']))

# --------------------------
# Création DataFrame et sauvegarde CSV
# --------------------------
if len(results) > 0:
    df = pd.DataFrame(results)
    df.to_csv('fusion_metrics_avg.csv', index=False)
    print("\n Les métriques (SSIM, PSNR, CC, VIF) ont été sauvegardées dans fusion_metrics_avg.csv")

    # Moyennes globales
    mean_ssim = df['SSIM'].mean()
    mean_psnr = df['PSNR'].mean()
    mean_cc = df['CC'].mean()
    mean_vif = df['VIF'].mean()

    print("\n Moyennes globales sur toutes les images :")
    print(f"   ➤ SSIM moyen : {mean_ssim:.4f}")
    print(f"   ➤ PSNR moyen : {mean_psnr:.4f}")
    print(f"   ➤ CC moyen   : {mean_cc:.4f}")
    print(f"   ➤ VIF moyen  : {mean_vif:.4f}")
else:
    print("Aucun fichier traité. Vérifie les dossiers et les noms de fichiers.")
