"""
Evaluation metrics for the layered neural codec.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Args:
        pred: Predicted tensor
        target: Target tensor
        max_val: Maximum value of the signal

    Returns:
        PSNR value
    """
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_val


def gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Create a Gaussian kernel.

    Args:
        size: Kernel size
        sigma: Standard deviation
        device: Device to create tensor on

    Returns:
        Gaussian kernel
    """
    coords = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, -1) * g.view(1, -1, 1)


def ssim(pred: torch.Tensor, target: torch.Tensor,
        window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Calculate Structural Similarity Index (SSIM).

    Args:
        pred: Predicted tensor of shape [B, C, H, W]
        target: Target tensor of shape [B, C, H, W]
        window_size: Window size for SSIM calculation
        sigma: Standard deviation for Gaussian kernel

    Returns:
        SSIM value
    """
    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Generate Gaussian kernel
    window = gaussian_kernel(window_size, sigma, pred.device)
    window = window.expand(pred.size(1), 1, window_size, window_size).contiguous()

    # Calculate mean, variance, and covariance
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=pred.size(1))
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=target.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=target.size(1)) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=pred.size(1)) - mu1_mu2

    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Return mean SSIM
    return ssim_map.mean([1, 2, 3])


def ms_ssim(pred: torch.Tensor, target: torch.Tensor,
           window_size: int = 11, sigma: float = 1.5,
           weights: Optional[List[float]] = None) -> torch.Tensor:
    """
    Calculate Multi-Scale Structural Similarity Index (MS-SSIM).

    Args:
        pred: Predicted tensor of shape [B, C, H, W]
        target: Target tensor of shape [B, C, H, W]
        window_size: Window size for SSIM calculation
        sigma: Standard deviation for Gaussian kernel
        weights: Weights for different scales

    Returns:
        MS-SSIM value
    """
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Generate Gaussian kernel
    window = gaussian_kernel(window_size, sigma, pred.device)
    window = window.expand(pred.size(1), 1, window_size, window_size).contiguous()

    # Calculate MS-SSIM
    levels = len(weights)
    values = []

    for i in range(levels):
        # Calculate SSIM for current level
        mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=pred.size(1))
        mu2 = F.conv2d(target, window, padding=window_size // 2, groups=target.size(1))

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=pred.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=target.size(1)) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=pred.size(1)) - mu1_mu2

        # Formula for SSIM
        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        ssim_map = ((2 * mu1_mu2 + C1) * cs_map) / (mu1_sq + mu2_sq + C1)

        # Take mean over spatial dimensions
        ssim_val = ssim_map.mean([2, 3])

        # For last level, use SSIM
        if i == levels - 1:
            values.append(ssim_val)
        else:
            # For other levels, use contrast and structure (CS)
            cs_val = cs_map.mean([2, 3])
            values.append(cs_val)

            # Downsample for next level
            if i < levels - 1:
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                target = F.avg_pool2d(target, kernel_size=2, stride=2)

    # Combine values from all levels
    msssim = torch.ones_like(values[0])
    for i, val in enumerate(values):
        msssim = msssim * (val ** weights[i])

    return msssim


def calculate_bpp(latent: torch.Tensor,
                 height: int,
                 width: int,
                 ppm_factor: float = 1.4) -> torch.Tensor:
    """
    Calculate bits per pixel (bpp) for latent representation.

    Args:
        latent: Latent tensor
        height: Original image height
        width: Original image width
        ppm_factor: Adjustment factor for practical model performance

    Returns:
        BPP value
    """
    # Calculate number of elements in latent
    num_elements = torch.tensor(latent.numel(), device=latent.device).float()

    # Calculate number of pixels in original image
    num_pixels = height * width

    # Calculate bpp (assuming each element requires 32 bits)
    # In practice, entropy coding would be used, so we apply a compression factor
    bpp = num_elements * 32 / num_pixels / ppm_factor

    return bpp


def calculate_metrics(pred: torch.Tensor,
                     target: torch.Tensor,
                     latents: List[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """
    Calculate various image quality metrics.

    Args:
        pred: Predicted tensor of shape [B, C, H, W] or [B, T, C, H, W]
        target: Target tensor of shape [B, C, H, W] or [B, T, C, H, W]
        latents: List of latent tensors for bpp calculation

    Returns:
        Dictionary of metrics
    """
    # Handle sequence dimension if present
    if pred.dim() == 5:  # [B, T, C, H, W]
        batch_size, seq_length = pred.shape[:2]
        C, H, W = pred.shape[2:]

        # Reshape to [B*T, C, H, W]
        pred_flat = pred.reshape(-1, C, H, W)
        target_flat = target.reshape(-1, C, H, W)

        # Calculate metrics
        psnr_val = psnr(pred_flat, target_flat)
        ssim_val = ssim(pred_flat, target_flat)
        msssim_val = ms_ssim(pred_flat, target_flat)

        # Reshape metrics back to [B, T]
        psnr_val = psnr_val.reshape(batch_size, seq_length)
        ssim_val = ssim_val.reshape(batch_size, seq_length)
        msssim_val = msssim_val.reshape(batch_size, seq_length)

        # Average over sequence dimension
        metrics = {
            "psnr": psnr_val.mean(dim=1),
            "ssim": ssim_val.mean(dim=1),
            "msssim": msssim_val.mean(dim=1)
        }
    else:  # [B, C, H, W]
        # Calculate metrics directly
        metrics = {
            "psnr": psnr(pred, target),
            "ssim": ssim(pred, target),
            "msssim": ms_ssim(pred, target)
        }

    # Calculate bpp if latents are provided
    if latents:
        _, _, H, W = target.shape[-4:] if target.dim() == 5 else target.shape[-3:]

        total_bpp = torch.zeros(pred.size(0), device=pred.device)

        for latent in latents:
            # If latent has sequence dimension, flatten it for bpp calculation
            if latent.dim() > 4:
                # Calculate bpp for each frame and average
                seq_length = latent.size(1)
                frame_bpps = []

                for t in range(seq_length):
                    frame_latent = latent[:, t]
                    frame_bpp = calculate_bpp(frame_latent, H, W)
                    frame_bpps.append(frame_bpp)

                latent_bpp = torch.stack(frame_bpps).mean(dim=0)
            else:
                latent_bpp = calculate_bpp(latent, H, W)

            total_bpp += latent_bpp

        metrics["bpp"] = total_bpp

    return metrics


def calculate_rd_curve(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Calculate rate-distortion curve metrics.

    Args:
        metrics: Dictionary of metrics for different bit rates

    Returns:
        Dictionary with BD-Rate and BD-PSNR metrics
    """
    # This function would calculate Bjøntegaard metrics for rate-distortion evaluation
    # For now, we'll just return placeholders
    return {
        "bd_rate": 0.0,
        "bd_psnr": 0.0
    }
