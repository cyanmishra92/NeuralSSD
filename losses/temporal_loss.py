"""
Temporal loss functions for the layered neural codec.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss to ensure smooth transitions between consecutive frames.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Initialize the temporal consistency loss.

        Args:
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.reduction = reduction

    def forward(self,
               prev_frame: torch.Tensor,
               curr_frame: torch.Tensor,
               flow: torch.Tensor) -> torch.Tensor:
        """
        Compute the temporal consistency loss.

        Args:
            prev_frame: Previous frame of shape [B, C, H, W]
            curr_frame: Current frame of shape [B, C, H, W]
            flow: Optical flow from previous to current frame of shape [B, 2, H, W]

        Returns:
            Temporal consistency loss
        """
        B, C, H, W = prev_frame.shape

        # Create sampling grid
        xx = torch.linspace(-1, 1, W, device=prev_frame.device)
        yy = torch.linspace(-1, 1, H, device=prev_frame.device)
        grid_y, grid_x = torch.meshgrid(yy, xx)
        grid = torch.stack((grid_x, grid_y), dim=0)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

        # Normalize flow to [-1, 1] range
        flow_normalized = torch.zeros_like(flow)
        flow_normalized[:, 0, :, :] = flow[:, 0, :, :] / (W / 2)
        flow_normalized[:, 1, :, :] = flow[:, 1, :, :] / (H / 2)

        # Add flow to sampling grid
        grid = grid + flow_normalized

        # Reshape grid for grid_sample
        grid = grid.permute(0, 2, 3, 1)  # [B, H, W, 2]

        # Warp previous frame using flow
        warped_prev_frame = F.grid_sample(
            prev_frame, grid, mode='bilinear', padding_mode='border', align_corners=True)

        # Compute mask for valid flow (areas where flow is well-defined)
        valid_mask = torch.ones_like(prev_frame)

        # Compute consistency loss (L1 distance between warped previous frame and current frame)
        consistency_loss = F.l1_loss(
            warped_prev_frame * valid_mask, curr_frame * valid_mask, reduction=self.reduction)

        return consistency_loss


class SequentialTemporalLoss(nn.Module):
    """
    Temporal loss for a sequence of frames.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Initialize the sequential temporal loss.

        Args:
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.reduction = reduction
        self.consistency_loss = TemporalConsistencyLoss(reduction="none")

    def forward(self,
               frames: torch.Tensor,
               flows: torch.Tensor,
               weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the temporal loss for a sequence of frames.

        Args:
            frames: Sequence of frames of shape [B, T, C, H, W]
            flows: Sequence of flows of shape [B, T-1, 2, H, W]
            weights: Optional weights for each frame pair of shape [B, T-1]

        Returns:
            Temporal consistency loss
        """
        B, T, C, H, W = frames.shape

        # Initialize loss
        total_loss = 0.0

        # Default weights (equal weighting)
        if weights is None:
            weights = torch.ones(B, T-1, device=frames.device)

        # Compute loss for each consecutive frame pair
        for t in range(T - 1):
            prev_frame = frames[:, t]
            curr_frame = frames[:, t + 1]
            flow = flows[:, t]

            # Compute consistency loss
            pair_loss = self.consistency_loss(prev_frame, curr_frame, flow)

            # Apply weight
            pair_loss = pair_loss * weights[:, t].view(B, 1, 1, 1)

            # Accumulate loss
            if self.reduction == "mean":
                total_loss += pair_loss.mean()
            elif self.reduction == "sum":
                total_loss += pair_loss.sum()
            else:  # "none"
                if t == 0:
                    total_loss = pair_loss
                else:
                    total_loss = torch.cat([total_loss, pair_loss], dim=1)

        # Normalize by number of frame pairs if using mean reduction
        if self.reduction == "mean":
            total_loss /= (T - 1)

        return total_loss


class FeatureTemporalLoss(nn.Module):
    """
    Temporal loss using features from a pretrained network.
    """

    def __init__(self,
                feature_extractor: nn.Module,
                layers: List[str] = None,
                weights: List[float] = None,
                reduction: str = "mean"):
        """
        Initialize the feature temporal loss.

        Args:
            feature_extractor: Feature extraction network
            layers: List of layer names to use for feature-based temporal loss
            weights: Optional weights for each layer
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.layers = layers
        self.reduction = reduction

        # Set layer weights
        self.layer_weights = {}
        if weights is not None:
            for i, layer in enumerate(layers):
                self.layer_weights[layer] = weights[i]
        else:
            for layer in layers:
                self.layer_weights[layer] = 1.0 / len(layers)

    def forward(self,
               prev_frame: torch.Tensor,
               curr_frame: torch.Tensor,
               flow: torch.Tensor) -> torch.Tensor:
        """
        Compute the feature-based temporal loss.

        Args:
            prev_frame: Previous frame of shape [B, C, H, W]
            curr_frame: Current frame of shape [B, C, H, W]
            flow: Optical flow from previous to current frame of shape [B, 2, H, W]

        Returns:
            Feature-based temporal loss
        """
        B, C, H, W = prev_frame.shape

        # Extract features
        prev_features = self.feature_extractor(prev_frame)
        curr_features = self.feature_extractor(curr_frame)

        # Initialize loss
        total_loss = 0.0

        # Compute loss for each feature layer
        for layer in self.layers:
            # Get features
            prev_feat = prev_features[layer]
            curr_feat = curr_features[layer]

            # Create sampling grid for feature map
            feat_h, feat_w = prev_feat.shape[2:]
            xx = torch.linspace(-1, 1, feat_w, device=prev_frame.device)
            yy = torch.linspace(-1, 1, feat_h, device=prev_frame.device)
            grid_y, grid_x = torch.meshgrid(yy, xx)
            grid = torch.stack((grid_x, grid_y), dim=0)
            grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, feat_h, feat_w]

            # Resample flow to match feature map size
            resampled_flow = F.interpolate(
                flow, size=(feat_h, feat_w), mode='bilinear', align_corners=False)

            # Normalize flow to [-1, 1] range
            flow_normalized = torch.zeros_like(resampled_flow)
            flow_normalized[:, 0, :, :] = resampled_flow[:, 0, :, :] / (feat_w / 2)
            flow_normalized[:, 1, :, :] = resampled_flow[:, 1, :, :] / (feat_h / 2)

            # Add flow to sampling grid
            grid = grid + flow_normalized

            # Reshape grid for grid_sample
            grid = grid.permute(0, 2, 3, 1)  # [B, feat_h, feat_w, 2]

            # Warp previous features using flow
            warped_prev_feat = F.grid_sample(
                prev_feat, grid, mode='bilinear', padding_mode='border', align_corners=True)

            # Compute consistency loss (L1 distance between warped previous features and current features)
            layer_loss = F.l1_loss(warped_prev_feat, curr_feat, reduction=self.reduction)

            # Apply layer weight
            total_loss += self.layer_weights[layer] * layer_loss

        return total_loss


def get_temporal_loss(loss_type: str = "consistency", **kwargs) -> nn.Module:
    """
    Get the appropriate temporal loss.

    Args:
        loss_type: Type of temporal loss ("consistency", "sequential", "feature")
        **kwargs: Additional arguments for the loss

    Returns:
        Temporal loss module
    """
    if loss_type.lower() == "consistency":
        return TemporalConsistencyLoss(**kwargs)
    elif loss_type.lower() == "sequential":
        return SequentialTemporalLoss(**kwargs)
    elif loss_type.lower() == "feature":
        if "feature_extractor" not in kwargs:
            raise ValueError("Feature extractor must be provided for feature-based temporal loss")
        return FeatureTemporalLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported temporal loss type: {loss_type}")
