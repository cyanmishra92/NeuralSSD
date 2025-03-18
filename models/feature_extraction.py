"""
Feature extraction modules for the layered neural codec.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union


class MobileNetFeatureExtractor(nn.Module):
    """Feature extractor based on MobileNetV2."""

    def __init__(self,
                feature_dim: int = 512,
                pretrained: bool = True,
                freeze_backbone: bool = False):
        """
        Initialize the feature extractor.

        Args:
            feature_dim: Dimension of output features
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone
        """
        super().__init__()

        # Load pretrained model
        self.backbone = models.mobilenet_v2(pretrained=pretrained)

        # Remove classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Feature adaptation layer
        self.feature_adaptation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, feature_dim),
            nn.ReLU(inplace=True)
        )

        # Spatial feature map projection
        self.spatial_projection = nn.Conv2d(1280, feature_dim, kernel_size=1)

        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from input frames.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Dictionary with 'features' and 'spatial_features'
        """
        # Get feature maps from backbone
        features = self.backbone(x)

        # Global features
        global_features = self.feature_adaptation(features)

        # Spatial features
        spatial_features = self.spatial_projection(features)

        return {
            'features': global_features,
            'spatial_features': spatial_features
        }


class TransformerFeatureExtractor(nn.Module):
    """Feature extractor based on Vision Transformer."""

    def __init__(self,
                feature_dim: int = 512,
                pretrained: bool = True,
                freeze_backbone: bool = False):
        """
        Initialize the feature extractor.

        Args:
            feature_dim: Dimension of output features
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone
        """
        super().__init__()

        # Load pretrained model
        self.backbone = models.vit_b_16(pretrained=pretrained)

        # Remove classifier head
        self.backbone.heads = nn.Identity()

        # Feature adaptation layer
        self.feature_adaptation = nn.Sequential(
            nn.Linear(768, feature_dim),
            nn.ReLU(inplace=True)
        )

        # Spatial projection for tokens
        self.spatial_projection = nn.Sequential(
            nn.Linear(768, feature_dim),
            nn.ReLU(inplace=True)
        )

        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from input frames.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Dictionary with 'features' and 'spatial_features'
        """
        # Get features from backbone (includes CLS token)
        features = self.backbone(x)

        # Reshape features to recover token structure
        batch_size = x.shape[0]
        token_dim = 768
        spatial_tokens = features[:, 1:].reshape(batch_size, -1, token_dim)

        # Global features from CLS token
        global_features = self.feature_adaptation(features[:, 0])

        # Spatial features from other tokens
        spatial_features = self.spatial_projection(spatial_tokens)

        # Convert back to spatial format for compatibility with other modules
        h = w = int((spatial_tokens.shape[1]) ** 0.5)
        spatial_features = spatial_features.transpose(1, 2).reshape(
            batch_size, -1, h, w)

        return {
            'features': global_features,
            'spatial_features': spatial_features
        }


def get_feature_extractor(config) -> nn.Module:
    """
    Get the feature extractor based on configuration.

    Args:
        config: Configuration object

    Returns:
        Feature extractor module
    """
    model_type = config.feature_extraction.model_type.lower()

    if model_type == "mobilenet":
        return MobileNetFeatureExtractor(
            feature_dim=config.feature_extraction.feature_dim,
            pretrained=config.feature_extraction.pretrained,
            freeze_backbone=config.feature_extraction.freeze_backbone
        )
    elif model_type == "transformer":
        return TransformerFeatureExtractor(
            feature_dim=config.feature_extraction.feature_dim,
            pretrained=config.feature_extraction.pretrained,
            freeze_backbone=config.feature_extraction.freeze_backbone
        )
    else:
        raise ValueError(f"Unsupported feature extractor type: {model_type}")
