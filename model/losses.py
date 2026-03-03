from __future__ import annotations

import torch
import torch.nn.functional as F


def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.float32:
        mask = mask.float()
    return mask


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = _safe_mask(valid_mask)
    if mask.shape[1] == 1 and pred.shape[1] > 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    diff = torch.abs(pred - target) * mask
    denom = mask.sum().clamp_min(eps)
    return diff.sum() / denom


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = _safe_mask(valid_mask)
    if mask.shape[1] == 1 and pred.shape[1] > 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(eps)
    return diff2.sum() / denom


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    mse = masked_mse(pred, target, valid_mask, eps=eps).clamp_min(eps)
    return 10.0 * torch.log10(torch.tensor(max_val * max_val, device=pred.device) / mse)


def masked_ssim(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Masked global SSIM approximation over valid pixels.
    """
    mask = _safe_mask(valid_mask)
    if mask.shape[1] == 1 and pred.shape[1] > 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    reduce_dims = (2, 3)
    wsum = mask.sum(dim=reduce_dims, keepdim=True).clamp_min(eps)
    mu_x = (pred * mask).sum(dim=reduce_dims, keepdim=True) / wsum
    mu_y = (target * mask).sum(dim=reduce_dims, keepdim=True) / wsum

    xm = (pred - mu_x) * mask
    ym = (target - mu_y) * mask
    sigma_x = (xm * xm).sum(dim=reduce_dims, keepdim=True) / wsum
    sigma_y = (ym * ym).sum(dim=reduce_dims, keepdim=True) / wsum
    sigma_xy = (xm * ym).sum(dim=reduce_dims, keepdim=True) / wsum

    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim = num / den.clamp_min(eps)
    return ssim.mean()


def _sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x: [B, C, H, W]
    Returns Sobel gx, gy with depthwise grouped conv.
    """
    channels = x.shape[1]
    device = x.device
    dtype = x.dtype

    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    kx = kx.repeat(channels, 1, 1, 1)
    ky = ky.repeat(channels, 1, 1, 1)

    gx = F.conv2d(x, kx, padding=1, groups=channels)
    gy = F.conv2d(x, ky, padding=1, groups=channels)
    return gx, gy


def masked_edge_l1_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    L1 loss on Sobel gradient magnitude under valid mask.
    """
    mask = _safe_mask(valid_mask)
    if mask.shape[1] == 1 and pred.shape[1] > 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)

    pred_gx, pred_gy = _sobel_gradients(pred)
    tgt_gx, tgt_gy = _sobel_gradients(target)
    pred_mag = torch.sqrt(pred_gx * pred_gx + pred_gy * pred_gy + eps)
    tgt_mag = torch.sqrt(tgt_gx * tgt_gx + tgt_gy * tgt_gy + eps)

    diff = torch.abs(pred_mag - tgt_mag) * mask
    denom = mask.sum().clamp_min(eps)
    return diff.sum() / denom
