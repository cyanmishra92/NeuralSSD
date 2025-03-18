"""
Reconstruction loss functions for the layered neural codec.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union


class ReconstructionLoss(nn.Module):
    """Base class for reconstruction losses."""

    def __init__(self, reduction: str = "mean"):
        """
        Initialize the reconstruction loss.

        Args:
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the reconstruction loss.

        Args:
            x: Predicted tensor
            y: Target tensor

        Returns:
            Loss value
        """
        raise NotImplementedError("Subclasses must implement this method")


class MSELoss(ReconstructionLoss):
    """Mean squared error loss."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the MSE loss.

        Args:
            x: Predicted tensor
            y: Target tensor

        Returns:
            MSE loss
        """
        return F.mse_loss(x, y, reduction=self.reduction)


class L1Loss(ReconstructionLoss):
    """L1 loss."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the L1 loss.

        Args:
            x: Predicted tensor
            y: Target tensor

        Returns:
            L1 loss
        """
        return F.l1_loss(x, y, reduction=self.reduction)


class CharbonnierLoss(ReconstructionLoss):
    """Charbonnier loss (robust L1 loss)."""

    def __init__(self, epsilon: float = 1e-3, reduction: str = "mean"):
        """
        Initialize the Charbonnier loss.

        Args:
            epsilon: Small constant for numerical stability
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__(reduction=reduction)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Charbonnier loss.

        Args:
            x: Predicted tensor
            y: Target tensor

        Returns:
            Charbonnier loss
        """
        diff = x - y
        loss = torch.sqrt(diff * diff + self.epsilon**2)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss


class SSIMLoss(ReconstructionLoss):
    """Structural similarity index (SSIM) loss."""

    def __init__(self, window_size: int = 11, reduction: str = "mean"):
        """
        Initialize the SSIM loss.

        Args:
            window_size: Window size for SSIM calculation
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__(reduction=reduction)
        self.window_size = window_size
        self.register_buffer('window', self._create_window(window_size))

    def _create_window(self, window_size: int) -> torch.Tensor:
        """
        Create a Gaussian window for SSIM calculation.

        Args:
            window_size: Window size

        Returns:
            Gaussian window tensor
        """
        sigma = 1.5
        gauss = torch.exp(
            -torch.arange(window_size).float()**2 / (2 * sigma**2)
        )
        window = gauss / gauss.sum()
        window = window.unsqueeze(0).unsqueeze(0)
        return window

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate SSIM between two tensors.

        Args:
            x: First tensor
            y: Second tensor

        Returns:
            SSIM value
        """
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        # Expand window to match input channels
        window = self.window.expand(x.size(1), 1, self.window_size)

        # Convert to 2D
        b, c, h, w = x.size()

        # Calculate means
        mu_x = F.conv2d(x, window, padding=self.window_size//2, groups=c)
        mu_y = F.conv2d(y, window, padding=self.window_size//2, groups=c)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        # Calculate variances and covariance
        sigma_x_sq = F.conv2d(x**2, window, padding=self.window_size//2, groups=c) - mu_x_sq
        sigma_y_sq = F.conv2d(y**2, window, padding=self.window_size//2, groups=c) - mu_y_sq
        sigma_xy = F.conv2d(x*y, window, padding=self.window_size//2, groups=c) - mu_xy

        # Calculate SSIM
        ssim_map = ((2*mu_xy + c1) * (2*sigma_xy + c2)) / \
                  ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))

        return ssim_map

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the SSIM loss.

        Args:
            x: Predicted tensor
            y: Target tensor

        Returns:
            1 - SSIM (to make it a loss that decreases with similarity)
        """
        ssim_value = self._ssim(x, y)
        loss = 1 - ssim_value

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss


def get_reconstruction_loss(loss_type: str = "mse", **kwargs) -> ReconstructionLoss:
    """
    Get the appropriate reconstruction loss.

    Args:
        loss_type: Type of loss ("mse", "l1", "charbonnier", "ssim")
        **kwargs: Additional arguments for the loss

    Returns:
        Reconstruction loss module
    """
    if loss_type.lower() == "mse":
        return MSELoss(**kwargs)
    elif loss_type.lower() == "l1":
        return L1Loss(**kwargs)
    elif loss_type.lower() == "charbonnier":
        return CharbonnierLoss(**kwargs)
    elif loss_type.lower() == "ssim":
        return SSIMLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
