# -*- coding: utf-8 -*-
import os
import json
import math
import random
from glob import glob
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

# ========================= 1. 配置与超参数 =========================
from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = str(PACKAGE_ROOT / 'data' / 'full' / 'train')
VERSION = 'mresunet'
A_DIR = os.path.join(DATA_ROOT, 'A_float')
B_DIR = os.path.join(DATA_ROOT, 'B_float')
COND_DIR = os.path.join(DATA_ROOT, 'cond_vec')
RUN_ROOT = str(PACKAGE_ROOT / 'outputs' / 'training')
DATA_SAVE = RUN_ROOT
CHECKPOINT_DIR = os.path.join(RUN_ROOT, 'checkpoints')
SAMPLE_DIR = os.path.join(RUN_ROOT, 'samples')
LOG_DIR = os.path.join(RUN_ROOT, 'logs')
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
HPARAM_JSON = os.path.join(RUN_ROOT, 'hparams.json')
data_ratio = 1.0

# 打印频率
PRINT_FREQ = 200

# 网络参数
INPUT_NC = 3
OUTPUT_NC = 1

BASE_CH = 16
N_DOWN = 2
N_BLOCKS = 3

COND_HIDDEN_DIM = 64
FUSION_DIM = 64

USE_DROPOUT = True
DROPOUT_RATE = 0.1

# 训练参数
NUM_EPOCHS = 3000
BATCH_SIZE = 8
LR_G = 0.0002
LR_G_MIN = 0.00005
LR_G_DECAY_START_EPOCH = 200
LR_G_DECAY_EPOCHS = 3000

WEIGHT_DECAY = 1e-4
SAVE_INTERVAL = 100
SAVE_EVERY_EPOCH = False

# Early Stopping
EARLY_STOP = True
PATIENCE_EPOCHS = 500
MIN_DELTA = 1e-3

# 损失与权重
LAMBDA_L1 = 1.0
LUX_MAX = 1003.12
MASK_ERODE_KERNEL = 7

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==================== 2. 工具函数 ====================

def get_linear_lr_plateau(epoch, base_lr, min_lr, start_decay_epoch, decay_epochs):
    if epoch <= start_decay_epoch:
        return base_lr
    t = epoch - start_decay_epoch
    if t >= decay_epochs:
        return min_lr
    progress = t / float(decay_epochs)
    return base_lr + (min_lr - base_lr) * progress

def compute_cond_stats(file_list):
    arrs = []
    for fn in file_list:
        arr = np.load(os.path.join(COND_DIR, fn)).astype(np.float32).ravel()
        arrs.append(arr)
    X = np.stack(arrs, axis=0)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std

def collect_hparams(cond_dim):
    total_files = data_ratio * len(sorted(glob(os.path.join(A_DIR, "*.npy"))))
    n_train = int(0.9 * total_files)
    n_val = total_files - n_train

    hparams = {
        "run_id": RUN_ID,
        "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {
            "DATA_ROOT": DATA_ROOT,
            "A_DIR": A_DIR,
            "B_DIR": B_DIR,
            "COND_DIR": COND_DIR,
            "RUN_ROOT": RUN_ROOT,
            "CHECKPOINT_DIR": CHECKPOINT_DIR,
            "SAMPLE_DIR": SAMPLE_DIR,
            "LOG_DIR": LOG_DIR,
        },
        "dataset": {
            "total_files": total_files,
            "train_files": n_train,
            "val_files": n_val,
            "train_ratio": 0.9,
            "val_ratio": 0.1,
            "cond_dim": cond_dim,
        },
        "network": {
            "INPUT_NC": INPUT_NC,
            "OUTPUT_NC": OUTPUT_NC,
            "BASE_CH": BASE_CH,
            "N_DOWN": N_DOWN,
            "N_BLOCKS": N_BLOCKS,
            "COND_HIDDEN_DIM": COND_HIDDEN_DIM,
            "FUSION_DIM": FUSION_DIM,
            "USE_DROPOUT": USE_DROPOUT,
            "DROPOUT_RATE": DROPOUT_RATE,
        },
        "train": {
            "NUM_EPOCHS": NUM_EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LR_G": LR_G,
            "LR_G_MIN": LR_G_MIN,
            "LR_G_DECAY_START_EPOCH": LR_G_DECAY_START_EPOCH,
            "LR_G_DECAY_EPOCHS": LR_G_DECAY_EPOCHS,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "SAVE_INTERVAL": SAVE_INTERVAL,
            "SAVE_EVERY_EPOCH": SAVE_EVERY_EPOCH,
            "PRINT_FREQ": PRINT_FREQ,
        },
        "early_stop": {
            "EARLY_STOP": EARLY_STOP,
            "PATIENCE_EPOCHS": PATIENCE_EPOCHS,
            "MIN_DELTA": MIN_DELTA,
        },
        "loss": {
            "LAMBDA_L1": LAMBDA_L1,
            "LUX_MAX": LUX_MAX,
            "mask_erode_kernel": MASK_ERODE_KERNEL,
            "tan_lux_median": LUX_MAX / 2,
            "tan_lux_max": LUX_MAX,
            "tan_sharpness": 0.95,
        },
        "env": {
            "SEED": SEED,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }
    }
    return hparams

def save_hparams(cond_dim):
    hparams = collect_hparams(cond_dim)
    with open(HPARAM_JSON, "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)
    print(f"[HyperParams Saved] {HPARAM_JSON}")


def save_checkpoint(path, epoch, model, optimizer, best_val, cond_mean, cond_std):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
            "cond_mean": torch.as_tensor(cond_mean),
            "cond_std": torch.as_tensor(cond_std),
        },
        path
    )

# ==================== 3. 数据集 ====================

class LuxDualInputDataset(Dataset):
    def __init__(self, a_dir, b_dir, cond_dir, files, cond_mean, cond_std):
        self.a_dir = a_dir
        self.b_dir = b_dir
        self.cond_dir = cond_dir
        self.files = files
        self.cond_mean = cond_mean
        self.cond_std = cond_std

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]

        A_raw = np.load(os.path.join(self.a_dir, fn)).astype(np.float32)
        A = A_raw[:INPUT_NC, :, :]

        B_raw = np.load(os.path.join(self.b_dir, fn)).astype(np.float32)
        if B_raw.ndim == 2:
            B_raw = B_raw[None, ...]

        mask = (B_raw > 0.0).astype(np.float32)

        cond = np.load(os.path.join(self.cond_dir, fn)).astype(np.float32).ravel()
        cond = (cond - self.cond_mean) / self.cond_std

        return (
            torch.from_numpy(np.clip(A, 0, 1)),
            torch.from_numpy(B_raw),
            torch.from_numpy(mask),
            torch.from_numpy(cond.astype(np.float32)),
        )

def build_loader():
    files = sorted([os.path.basename(p) for p in glob(os.path.join(A_DIR, "*.npy"))])
    files = files[:int(data_ratio * len(files))]
    n_train = int(0.9 * len(files))

    cond_mean, cond_std = compute_cond_stats(files[:n_train])

    train_loader = DataLoader(
        LuxDualInputDataset(A_DIR, B_DIR, COND_DIR, files[:n_train], cond_mean, cond_std),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        LuxDualInputDataset(A_DIR, B_DIR, COND_DIR, files[n_train:], cond_mean, cond_std),
        batch_size=BATCH_SIZE, shuffle=False
    )
    return train_loader, val_loader, cond_mean, cond_std

# ==================== 4. 网络定义 ====================

from .model import DualHeadUNet

# ==================== 5. Tan 权重 ====================

def compute_tan_weights(B, lux_median, lux_max, sharpness=0.95):
    dist = (B - lux_median) / (lux_max - lux_median + 1e-6)
    dist = torch.clamp(dist, -1.0, 1.0)
    phi = dist * (sharpness * np.pi / 2)
    w = torch.abs(torch.tan(phi)) + 1.0
    return w

def gray_to_rgb(x):
    return x.repeat(1, 3, 1, 1)

# ==================== 6. 训练主逻辑 ====================

def train():
    train_loader, val_loader, cond_mean, cond_std = build_loader()

    sample_cond = np.load(sorted(glob(os.path.join(COND_DIR, "*.npy")))[0]).astype(np.float32).ravel()
    cond_dim = sample_cond.shape[0]
    save_hparams(cond_dim)

    G = DualHeadUNet(
        INPUT_NC, OUTPUT_NC,
        cond_dim=cond_dim,
        base_ch=BASE_CH,
        n_down=N_DOWN,
        n_blocks=N_BLOCKS
    ).to(device)

    optG = torch.optim.Adam(
        G.parameters(),
        lr=LR_G,
        betas=(0.5, 0.999),
        weight_decay=WEIGHT_DECAY
    )

    writer = SummaryWriter(LOG_DIR)

    step = 0
    best_val = float("inf")
    best_ep = 0
    bad_epochs = 0
    eps = 1e-6

    for ep in range(1, NUM_EPOCHS + 1):
        tr_sum_loss_G_l1 = 0.0
        tr_sum_loss_design = 0.0
        tr_sum_loss_outdoor = 0.0
        tr_sum_loss_boundary = 0.0
        tr_sum_mae_design = 0.0
        tr_n_batches = 0
        last_img_grid = None

        current_lr_g = get_linear_lr_plateau(
            epoch=ep,
            base_lr=LR_G,
            min_lr=LR_G_MIN,
            start_decay_epoch=LR_G_DECAY_START_EPOCH,
            decay_epochs=LR_G_DECAY_EPOCHS,
        )
        for param_group in optG.param_groups:
            param_group["lr"] = current_lr_g

        print(f"\n[Epoch {ep}/{NUM_EPOCHS}] lr_G = {current_lr_g:.6e}")
        writer.add_scalar("LR/G", current_lr_g, ep)

        G.train()
        for i, (A, B, mask_valid, cond) in enumerate(train_loader):
            step += 1
            A, B, mask_valid, cond = A.to(device), B.to(device), mask_valid.to(device), cond.to(device)

            if mask_valid.ndim == 3:
                mask_valid = mask_valid.unsqueeze(1)

            with torch.no_grad():
                eroded_mask = 1.0 - F.max_pool2d(
                    1.0 - mask_valid,
                    kernel_size=MASK_ERODE_KERNEL,
                    stride=1,
                    padding=MASK_ERODE_KERNEL // 2
                )
                eroded_mask = (eroded_mask > 0.5).float()

                boundary_mask = mask_valid - eroded_mask
                outdoor_mask = 1.0 - mask_valid

            optG.zero_grad()
            fake = G(A, cond)

            abs_diff = torch.abs(fake - B)
            w_tan = compute_tan_weights(B, lux_median=LUX_MAX/2, lux_max=LUX_MAX, sharpness=0.90)

            design_sum = (abs_diff * w_tan * eroded_mask).sum(dim=(1, 2, 3))
            design_pixels = eroded_mask.sum(dim=(1, 2, 3)) + eps
            loss_design = (design_sum / design_pixels).mean()

            outdoor_sum = (abs_diff * outdoor_mask).sum(dim=(1, 2, 3))
            outdoor_pixels = outdoor_mask.sum(dim=(1, 2, 3)) + eps
            loss_outdoor = (outdoor_sum / outdoor_pixels).mean()

            boundary_sum = (abs_diff * boundary_mask).sum(dim=(1, 2, 3))
            boundary_pixels = boundary_mask.sum(dim=(1, 2, 3)) + eps
            loss_boundary = (boundary_sum / boundary_pixels).mean()

            loss_G_l1 = (
                1 * loss_design +
                1 * loss_outdoor +
                3 * loss_boundary
            ) * LAMBDA_L1

            loss_G_l1.backward()
            optG.step()

            with torch.no_grad():
                batch_mae_design = (abs_diff * eroded_mask).sum() / (eroded_mask.sum() + eps)

            tr_sum_loss_G_l1 += loss_G_l1.item()
            tr_sum_loss_design += loss_design.item()
            tr_sum_loss_outdoor += loss_outdoor.item()
            tr_sum_loss_boundary += loss_boundary.item()
            tr_sum_mae_design += batch_mae_design.item()
            tr_n_batches += 1

            if step % PRINT_FREQ == 0 or step == 1 or i == len(train_loader) - 1:
                print(f"  [Step {step}] G_l1_weighted={loss_G_l1.item():.3f} Design_MAE={batch_mae_design.item():.2f}")

                A_vis = A[:4]
                B_vis = gray_to_rgb((B[:4] / LUX_MAX).clamp(0.0, 1.0))
                F_vis = gray_to_rgb((fake[:4] / LUX_MAX).clamp(0.0, 1.0))

                D_raw = (fake[:4] - B[:4]) / (2 * LUX_MAX)
                D_vis = (D_raw * eroded_mask[:4]) + 0.5
                D_vis = gray_to_rgb(D_vis.clamp(0.0, 1.0))

                img_grid = torch.cat([A_vis, B_vis, F_vis, D_vis], dim=0)
                vutils.save_image(img_grid, os.path.join(SAMPLE_DIR, f"ep{ep}_step{step}.png"), nrow=4)
                last_img_grid = img_grid

        if tr_n_batches > 0:
            avg_tr_loss_G_l1 = tr_sum_loss_G_l1 / tr_n_batches
            avg_tr_loss_design = tr_sum_loss_design / tr_n_batches
            avg_tr_loss_outdoor = tr_sum_loss_outdoor / tr_n_batches
            avg_tr_loss_boundary = tr_sum_loss_boundary / tr_n_batches
            avg_tr_mae_design = tr_sum_mae_design / tr_n_batches
        else:
            avg_tr_loss_G_l1 = 0.0
            avg_tr_loss_design = 0.0
            avg_tr_loss_outdoor = 0.0
            avg_tr_loss_boundary = 0.0
            avg_tr_mae_design = 0.0

        writer.add_scalar("Loss/G_l1_weighted", avg_tr_loss_G_l1, ep)
        writer.add_scalar("Loss/design", avg_tr_loss_design, ep)
        writer.add_scalar("Loss/outdoor", avg_tr_loss_outdoor, ep)
        writer.add_scalar("Loss/boundary", avg_tr_loss_boundary, ep)
        writer.add_scalar("Val/train_Design_MAE_Lux", avg_tr_mae_design, ep)

        if last_img_grid is not None:
            writer.add_images("Train/Vis_A_B_Fake_Diff", last_img_grid, ep)

        # ------------------------ 验证 ------------------------
        G.eval()
        with torch.no_grad():
            val_mae_design = 0.0
            val_loss_total = 0.0
            n_batches = 0

            for A_v, B_v, mask_v, cond_v in val_loader:
                A_v, B_v, mask_v, cond_v = A_v.to(device), B_v.to(device), mask_v.to(device), cond_v.to(device)
                if mask_v.ndim == 3:
                    mask_v = mask_v.unsqueeze(1)

                v_eroded = 1.0 - F.max_pool2d(
                    1.0 - mask_v,
                    kernel_size=MASK_ERODE_KERNEL,
                    stride=1,
                    padding=MASK_ERODE_KERNEL // 2
                )
                v_eroded = (v_eroded > 0.5).float()

                v_boundary = mask_v - v_eroded
                v_outdoor = 1.0 - mask_v

                pred_v = G(A_v, cond_v).clamp(min=0.0)
                abs_diff_v = torch.abs(pred_v - B_v)

                w_tan_v = compute_tan_weights(B_v, lux_median=LUX_MAX/2, lux_max=LUX_MAX, sharpness=0.95)

                design_sum_v = (abs_diff_v * w_tan_v * v_eroded).sum(dim=(1, 2, 3))
                design_pixels_v = v_eroded.sum(dim=(1, 2, 3)) + eps
                loss_design_v = (design_sum_v / design_pixels_v).mean()

                outdoor_sum_v = (abs_diff_v * v_outdoor).sum(dim=(1, 2, 3))
                outdoor_pixels_v = v_outdoor.sum(dim=(1, 2, 3)) + eps
                loss_outdoor_v = (outdoor_sum_v / outdoor_pixels_v).mean()

                boundary_sum_v = (abs_diff_v * v_boundary).sum(dim=(1, 2, 3))
                boundary_pixels_v = v_boundary.sum(dim=(1, 2, 3)) + eps
                loss_boundary_v = (boundary_sum_v / boundary_pixels_v).mean()

                loss_G_l1_v = (
                    1 * loss_design_v +
                    1 * loss_outdoor_v +
                    3 * loss_boundary_v
                ) * LAMBDA_L1

                val_loss_total += loss_G_l1_v.item()

                mae = (abs_diff_v * v_eroded).sum() / (v_eroded.sum() + eps)
                val_mae_design += mae.item()
                n_batches += 1

            avg_val_mae = val_mae_design / max(1, n_batches)
            avg_val_loss = val_loss_total / max(1, n_batches)

            print(f"==> Ep {ep} Val MAE: {avg_val_mae:.2f} Lux | Val Weighted Loss: {avg_val_loss:.4f}")
            writer.add_scalar("Val/Val_Design_MAE_Lux", avg_val_mae, ep)
            writer.add_scalar("Val/Val_Loss_Weighted", avg_val_loss, ep)

        improved = (best_val - avg_val_mae) > MIN_DELTA
        if improved:
            best_val = avg_val_mae
            best_ep = ep
            bad_epochs = 0
            save_checkpoint(
                os.path.join(CHECKPOINT_DIR, "G_best.pth"),
                ep,
                G,
                optG,
                best_val,
                cond_mean,
                cond_std,
            )
            print(f"  [Best Save] Ep {ep}: val_mae={best_val:.4f}")
        else:
            bad_epochs += 1
            print(f"  [No Improve] bad_epochs={bad_epochs}/{PATIENCE_EPOCHS} | Best: {best_val:.2f} @ Ep {best_ep}")

        if SAVE_EVERY_EPOCH or ep % SAVE_INTERVAL == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"G_ep{ep:04d}.pth")
            save_checkpoint(
                save_path,
                ep,
                G,
                optG,
                best_val,
                cond_mean,
                cond_std,
            )
            print(f"  [Epoch Save] Model saved to {save_path}")

        writer.flush()

        if EARLY_STOP and bad_epochs >= PATIENCE_EPOCHS:
            print(f"--- Early Stop triggered at Ep {ep}. Best MAE={best_val:.4f} ---")
            break

    writer.close()


