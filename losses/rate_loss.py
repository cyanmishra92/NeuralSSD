"""
Rate loss functions for the layered neural codec.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class EntropyBottleneck(nn.Module):
    """
    Entropy bottleneck for rate estimation, based on the approach from
    "End-to-end Optimized Image Compression" (Ballé et al., 2018)
    """

    def __init__(self,
                channels: int,
                init_scale: float = 10.0,
                filters: Tuple[int, ...] = (3, 3, 3),
                tail_mass: float = 1e-9):
        """
        Initialize the entropy bottleneck.

        Args:
            channels: Number of input channels
            init_scale: Initial scale factor
            filters: Number of filters in each convolutional layer
            tail_mass: Tail mass for quantization bounds
        """
        super().__init__()

        self.channels = channels
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Build the non-parametric CDF model
        self._build_cdf_model()

        # Initialize parameters
        self.quantiles = nn.Parameter(torch.zeros(channels, 1, 1))
        self.register_buffer("target", torch.zeros(1))

    def _build_cdf_model(self):
        """Build the convolutional transforms for density modeling."""
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))

        # Create layers for modeling CDF
        modules = []

        for i in range(len(self.filters) + 1):
            modules.append(nn.Conv2d(
                filters[i], filters[i + 1], kernel_size=3, padding=1))

            if i < len(self.filters):
                modules.append(nn.ReLU(inplace=True))
                modules.append(nn.Conv2d(
                    filters[i + 1], filters[i + 1], kernel_size=3, padding=1))
                modules.append(nn.ReLU(inplace=True))

        self.cdf_model = nn.Sequential(*modules)

    def _quantize(self, x: torch.Tensor, mode: str = "noise") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input tensor.

        Args:
            x: Input tensor
            mode: Quantization mode ('noise' for training, 'symbols' for evaluation)

        Returns:
            Quantized tensor and likelihood
        """
        if mode == "noise":
            # Add uniform noise during training
            noise = torch.rand_like(x) - 0.5
            x_noisy = x + noise

            # Model the density
            x_reshape = x_noisy.permute(1, 0, 2, 3)  # [C, B, H, W]
            x_reshape = x_reshape.unsqueeze(0)  # [1, C, B, H, W]

            # Apply CDF model
            logits = self.cdf_model(x_reshape)

            # Get likelihood
            likelihood = F.softplus(logits)
            likelihood = likelihood.squeeze(0).permute(1, 0, 2, 3)  # [B, C, H, W]

            return x_noisy, likelihood

        elif mode == "symbols":
            # Actual quantization
            x_rounded = torch.round(x)

            # Calculate likelihood based on CDF model
            x_reshape = x_rounded.permute(1, 0, 2, 3)  # [C, B, H, W]
            x_reshape = x_reshape.unsqueeze(0)  # [1, C, B, H, W]

            # Apply CDF model
            logits = self.cdf_model(x_reshape)

            # Get likelihood
            likelihood = F.softplus(logits)
            likelihood = likelihood.squeeze(0).permute(1, 0, 2, 3)  # [B, C, H, W]

            return x_rounded, likelihood

        else:
            raise ValueError(f"Unsupported quantization mode: {mode}")

    def forward(self, x: torch.Tensor, training: bool = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Quantized tensor and rate
        """
        if training is None:
            training = self.training

        # Choose quantization mode based on training state
        mode = "noise" if training else "symbols"

        # Apply quantization
        x_hat, likelihood = self._quantize(x, mode=mode)

        # Calculate rate
        rate = torch.log(likelihood).sum(dim=(1, 2, 3))

        return x_hat, rate


class RateLoss(nn.Module):
    """
    Rate loss to control bitrate.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Initialize the rate loss.

        Args:
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, rate: torch.Tensor) -> torch.Tensor:
        """
        Compute the rate loss.

        Args:
            rate: Rate tensor

        Returns:
            Rate loss
        """
        if self.reduction == "mean":
            return rate.mean()
        elif self.reduction == "sum":
            return rate.sum()
        else:  # "none"
            return rate


class LaplacianRateLoss(nn.Module):
    """
    Rate loss using Laplacian distribution modeling.
    """

    def __init__(self,
                scale: float = 1.0,
                reduction: str = "mean"):
        """
        Initialize the Laplacian rate loss.

        Args:
            scale: Scale parameter for Laplacian distribution
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.scale = scale
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the rate loss using Laplacian distribution.

        Args:
            x: Latent representation tensor

        Returns:
            Rate loss
        """
        # Add small constant for numerical stability
        eps = 1e-9

        # Calculate negative log-likelihood of Laplacian distribution
        rate = torch.abs(x) / self.scale + torch.log(2 * self.scale + eps)

        # Apply reduction
        if self.reduction == "mean":
            return rate.mean()
        elif self.reduction == "sum":
            return rate.sum()
        else:  # "none"
            return rate


class GaussianRateLoss(nn.Module):
    """
    Rate loss using Gaussian distribution modeling.
    """

    def __init__(self,
                std: float = 1.0,
                reduction: str = "mean"):
        """
        Initialize the Gaussian rate loss.

        Args:
            std: Standard deviation for Gaussian distribution
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.std = std
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the rate loss using Gaussian distribution.

        Args:
            x: Latent representation tensor

        Returns:
            Rate loss
        """
        # Constants
        log_2pi = np.log(2 * np.pi)

        # Calculate negative log-likelihood of Gaussian distribution
        rate = 0.5 * (x / self.std) ** 2 + 0.5 * log_2pi + torch.log(self.std)

        # Apply reduction
        if self.reduction == "mean":
            return rate.mean()
        elif self.reduction == "sum":
            return rate.sum()
        else:  # "none"
            return rate


def get_rate_loss(loss_type: str = "laplacian", **kwargs) -> nn.Module:
    """
    Get the appropriate rate loss.

    Args:
        loss_type: Type of rate loss ("laplacian", "gaussian", "entropy")
        **kwargs: Additional arguments for the loss

    Returns:
        Rate loss module
    """
    if loss_type.lower() == "laplacian":
        return LaplacianRateLoss(**kwargs)
    elif loss_type.lower() == "gaussian":
        return GaussianRateLoss(**kwargs)
    elif loss_type.lower() == "entropy":
        # For entropy bottleneck, we expect the rate to be passed directly
        return RateLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported rate loss type: {loss_type}")
