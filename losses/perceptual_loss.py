"""
Perceptual loss functions for the layered neural codec.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    """

    def __init__(self,
                layers: List[str] = None,
                weights: List[float] = None,
                normalize: bool = True,
                resize: bool = False):
        """
        Initialize the VGG perceptual loss.

        Args:
            layers: List of VGG layer names to use for perceptual loss
            weights: Optional weights for each layer
            normalize: Whether to normalize input images
            resize: Whether to resize input images to match VGG input size
        """
        super().__init__()

        if layers is None:
            layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']

        self.normalize = normalize
        self.resize = resize

        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True)

        # Register feature extraction layers
        self.slices = nn.ModuleDict()
        self.layer_weights = {}

        # Layer mapping
        layer_map = {
            'conv1_1': '0', 'relu1_1': '1', 'conv1_2': '2', 'relu1_2': '3', 'pool1': '4',
            'conv2_1': '5', 'relu2_1': '6', 'conv2_2': '7', 'relu2_2': '8', 'pool2': '9',
            'conv3_1': '10', 'relu3_1': '11', 'conv3_2': '12', 'relu3_2': '13', 'conv3_3': '14', 'relu3_3': '15', 'pool3': '16',
            'conv4_1': '17', 'relu4_1': '18', 'conv4_2': '19', 'relu4_2': '20', 'conv4_3': '21', 'relu4_3': '22', 'pool4': '23',
            'conv5_1': '24', 'relu5_1': '25', 'conv5_2': '26', 'relu5_2': '27', 'conv5_3': '28', 'relu5_3': '29', 'pool5': '30'
        }

        # Create slices for each target layer
        features = vgg.features
        start_idx = 0
        for layer_name in layers:
            if layer_name not in layer_map:
                raise ValueError(f"Unknown layer name: {layer_name}")

            end_idx = int(layer_map[layer_name]) + 1
            self.slices[layer_name] = nn.Sequential(*list(features[start_idx:end_idx]))
            start_idx = end_idx

            # Set layer weights
            if weights is not None:
                self.layer_weights[layer_name] = weights[layers.index(layer_name)]
            else:
                self.layer_weights[layer_name] = 1.0

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor to match VGG preprocessing.

        Args:
            x: Input tensor with values in [0, 1]

        Returns:
            Normalized tensor
        """
        # ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        return (x - mean) / std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the perceptual loss.

        Args:
            x: Predicted tensor of shape [B, C, H, W] in range [0, 1]
            y: Target tensor of shape [B, C, H, W] in range [0, 1]

        Returns:
            Perceptual loss
        """
        if self.normalize:
            x = self._normalize(x)
            y = self._normalize(y)

        if self.resize and (x.shape[2] != 224 or x.shape[3] != 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)

        total_loss = 0.0

        # Compute feature-based losses for each layer
        for layer_name, layer in self.slices.items():
            x_features = layer(x)
            y_features = layer(y)

            # L2 loss on normalized features
            loss = F.mse_loss(x_features, y_features)

            # Apply layer weight
            total_loss += self.layer_weights[layer_name] * loss

        return total_loss


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) loss.
    Simplified implementation based on the LPIPS paper.
    """

    def __init__(self,
                net_type: str = 'alex',
                lpips_weights: bool = True):
        """
        Initialize the LPIPS loss.

        Args:
            net_type: Network type ('alex', 'vgg', 'squeeze')
            lpips_weights: Whether to use pretrained LPIPS weights
        """
        super().__init__()

        # For a complete implementation, we would need to include
        # the pretrained LPIPS weights. For now, we'll use a simplified
        # version based on feature distances.

        if net_type == 'alex':
            self.net = models.alexnet(pretrained=True).features
        elif net_type == 'vgg':
            self.net = models.vgg16(pretrained=True).features
        elif net_type == 'squeeze':
            self.net = models.squeezenet1_1(pretrained=True).features
        else:
            raise ValueError(f"Unsupported network type: {net_type}")

        # Freeze network parameters
        for param in self.net.parameters():
            param.requires_grad = False

        # Define layers to use
        self.layers = [0, 4, 9]  # Example layers

        # If using LPIPS weights, we would load them here
        self.use_lpips_weights = lpips_weights
        if self.use_lpips_weights:
            # In a real implementation, we would load the LPIPS weights here
            # For now, we'll just use equal weighting
            self.weights = [1.0 / len(self.layers)] * len(self.layers)
        else:
            self.weights = [1.0 / len(self.layers)] * len(self.layers)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor.

        Args:
            x: Input tensor with values in [0, 1]

        Returns:
            Normalized tensor
        """
        # ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        return (x - mean) / std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the LPIPS loss.

        Args:
            x: Predicted tensor of shape [B, C, H, W] in range [0, 1]
            y: Target tensor of shape [B, C, H, W] in range [0, 1]

        Returns:
            LPIPS loss
        """
        # Normalize inputs
        x = self._normalize(x)
        y = self._normalize(y)

        # Compute feature distances
        total_loss = 0.0

        for i, layer_idx in enumerate(self.layers):
            # Extract features
            for j in range(layer_idx + 1):
                if j == 0:
                    x_feat = self.net[j](x)
                    y_feat = self.net[j](y)
                else:
                    x_feat = self.net[j](x_feat)
                    y_feat = self.net[j](y_feat)

            # Normalize features
            x_feat = F.normalize(x_feat, p=2, dim=1)
            y_feat = F.normalize(y_feat, p=2, dim=1)

            # Compute distance
            loss = F.mse_loss(x_feat, y_feat)

            # Apply weight
            total_loss += self.weights[i] * loss

        return total_loss


def get_perceptual_loss(loss_type: str = "vgg", **kwargs) -> nn.Module:
    """
    Get the appropriate perceptual loss.

    Args:
        loss_type: Type of perceptual loss ("vgg", "lpips")
        **kwargs: Additional arguments for the loss

    Returns:
        Perceptual loss module
    """
    if loss_type.lower() == "vgg":
        return VGGPerceptualLoss(**kwargs)
    elif loss_type.lower() == "lpips":
        return LPIPSLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported perceptual loss type: {loss_type}")
