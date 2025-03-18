"""
Enhancement layer modules for the layered neural codec.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

from .base_layer import ConvBlock, SelfAttention


class ROIDetector(nn.Module):
    """Region of interest detector for adaptive bit allocation."""

    def __init__(self, input_channels: int = 3, threshold: float = 0.5):
        """
        Initialize the ROI detector.

        Args:
            input_channels: Number of input channels
            threshold: Threshold for ROI detection
        """
        super().__init__()

        self.threshold = threshold

        # Feature extraction
        self.features = nn.Sequential(
            ConvBlock(input_channels, 32, kernel_size=3, stride=2, activation="relu"),
            ConvBlock(32, 64, kernel_size=3, stride=2, activation="relu"),
            ConvBlock(64, 64, kernel_size=3, stride=1, activation="relu"),
        )

        # ROI prediction
        self.predict = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Upsampling convs
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(1, 16, kernel_size=3, stride=1, activation="relu"),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect regions of interest.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            ROI mask of shape [B, 1, H, W]
        """
        # Extract features
        features = self.features(x)

        # Predict ROI
        roi = self.predict(features)

        # Upsample to original resolution
        roi = self.upsample(roi)

        return roi


class EnhancementLayer(nn.Module):
    """Enhancement layer for refining reconstruction quality."""

    def __init__(self,
                input_channels: int = 3,
                latent_dim: int = 64,
                use_roi: bool = True,
                roi_weight_factor: float = 2.0):
        """
        Initialize the enhancement layer.

        Args:
            input_channels: Number of input channels (RGB + previous reconstruction)
            latent_dim: Dimension of latent representation
            use_roi: Whether to use region of interest weighting
            roi_weight_factor: Factor to weight ROI regions
        """
        super().__init__()

        self.use_roi = use_roi
        self.roi_weight_factor = roi_weight_factor

        # ROI detector
        if use_roi:
            self.roi_detector = ROIDetector(input_channels)

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(input_channels, 32, kernel_size=3, stride=2, activation="relu"),
            ConvBlock(32, 64, kernel_size=3, stride=2, activation="relu"),
            ConvBlock(64, latent_dim, kernel_size=3, stride=1, activation="relu"),
        )

        # Bottleneck attention
        self.attention = SelfAttention(latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            ConvBlock(latent_dim, 64, kernel_size=3, stride=1, activation="relu"),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor,
               prev_reconstruction: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process input through enhancement layer.

        Args:
            x: Input tensor (original image or residual) of shape [B, C, H, W]
            prev_reconstruction: Previous reconstruction of shape [B, C, H, W]

        Returns:
            Dictionary with enhancement outputs
        """
        # Concatenate with previous reconstruction if provided
        if prev_reconstruction is not None:
            input_tensor = torch.cat([x, prev_reconstruction], dim=1)
        else:
            input_tensor = x

        # Detect ROI if enabled
        roi_mask = None
        if self.use_roi and prev_reconstruction is not None:
            roi_mask = self.roi_detector(input_tensor)

        # Encode
        latent = self.encoder(input_tensor)

        # Apply attention
        latent = self.attention(latent)

        # Decode
        refinement = self.decoder(latent)

        # Apply ROI weighting if enabled
        if roi_mask is not None:
            # Weight the refinement based on ROI
            weighted_refinement = refinement * (1.0 + (self.roi_weight_factor - 1.0) * roi_mask)
        else:
            weighted_refinement = refinement

        # If previous reconstruction is provided, add refinement
        if prev_reconstruction is not None:
            output = prev_reconstruction + weighted_refinement
        else:
            output = weighted_refinement

        return {
            "latent": latent,
            "refinement": refinement,
            "weighted_refinement": weighted_refinement,
            "output": output,
            "roi_mask": roi_mask
        }


class EnhancementLayerStack(nn.Module):
    """Stack of enhancement layers for progressive refinement."""

    def __init__(self, config):
        """
        Initialize the enhancement layer stack.

        Args:
            config: Configuration object
        """
        super().__init__()

        self.num_layers = config.enhancement_layer.num_layers
        self.latent_dims = config.enhancement_layer.latent_dims
        self.roi_enabled = config.enhancement_layer.roi_enabled
        self.roi_weight_factor = config.enhancement_layer.roi_weight_factor

        # Create enhancement layers
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            # First layer takes 3 channels, subsequent layers take 6 (original + previous)
            input_channels = 3 if i == 0 else 6

            self.layers.append(EnhancementLayer(
                input_channels=input_channels,
                latent_dim=self.latent_dims[i],
                use_roi=self.roi_enabled,
                roi_weight_factor=self.roi_weight_factor
            ))

    def forward(self, x: torch.Tensor,
               base_reconstruction: torch.Tensor) -> Dict[str, Any]:
        """
        Apply progressive enhancement.

        Args:
            x: Original input tensor of shape [B, C, H, W]
            base_reconstruction: Base layer reconstruction of shape [B, C, H, W]

        Returns:
            Dictionary with enhancement outputs
        """
        current_reconstruction = base_reconstruction
        layer_outputs = []
        roi_masks = []
        latents = []

        # Apply enhancement layers sequentially
        for i, layer in enumerate(self.layers):
            if i == 0:
                # First layer takes original and base reconstruction
                layer_out = layer(x, base_reconstruction)
            else:
                # Subsequent layers take original and previous enhanced reconstruction
                layer_out = layer(x, current_reconstruction)

            # Update current reconstruction
            current_reconstruction = layer_out["output"]

            # Store outputs
            layer_outputs.append(layer_out)
            latents.append(layer_out["latent"])

            if layer_out["roi_mask"] is not None:
                roi_masks.append(layer_out["roi_mask"])

        return {
            "final_output": current_reconstruction,
            "layer_outputs": layer_outputs,
            "latents": latents,
            "roi_masks": roi_masks
        }
