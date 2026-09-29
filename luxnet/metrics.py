"""Historical test metrics, including SSIM clipping at 603.86 lux."""
import numpy as np
import torch
import torch.nn.functional as F
LUX_MAX = 603.86
WINDOW_SIZE = 11
SIGMA = 1.5
ERODE_PX = 3
EPS = 1e-12
def gaussian_kernel_2d(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / (g.sum() + EPS)
    return (g[:, None] * g[None, :])[None, None, :, :]


@torch.no_grad()
def compute_batch_metrics(
    pred_np: np.ndarray,
    gt_np: np.ndarray,
    device: torch.device,
    kernel: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred = torch.from_numpy(pred_np).to(device=device, dtype=torch.float32).unsqueeze(1)
    gt = torch.from_numpy(gt_np).to(device=device, dtype=torch.float32).unsqueeze(1)

    valid = (gt > 1e-6).float()
    invalid = 1.0 - valid
    erode_kernel = 2 * ERODE_PX + 1
    core = 1.0 - F.max_pool2d(
        invalid,
        kernel_size=erode_kernel,
        stride=1,
        padding=ERODE_PX,
    )
    core = (core > 0.5).float()
    core_pixels = core.sum(dim=(1, 2, 3)).clamp_min(1.0)

    diff = pred - gt
    mae = (diff.abs() * core).sum(dim=(1, 2, 3)) / core_pixels
    mse = ((diff * diff) * core).sum(dim=(1, 2, 3)) / core_pixels

    x = torch.clamp(gt / LUX_MAX, 0.0, 1.0)
    y = torch.clamp(pred / LUX_MAX, 0.0, 1.0)

    pad = kernel.shape[-1] // 2
    mu_x = F.conv2d(x, kernel, padding=pad)
    mu_y = F.conv2d(y, kernel, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = torch.clamp(F.conv2d(x * x, kernel, padding=pad) - mu_x2, min=0.0)
    sigma_y2 = torch.clamp(F.conv2d(y * y, kernel, padding=pad) - mu_y2, min=0.0)
    sigma_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + EPS
    )
    ssim = (ssim_map * core).sum(dim=(1, 2, 3)) / core_pixels

    return (
        core_pixels.cpu().numpy().astype(np.int64),
        mae.cpu().numpy(),
        mse.cpu().numpy(),
        ssim.cpu().numpy(),
    )
