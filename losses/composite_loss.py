"""
Composite loss function for the layered neural codec.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union

from .reconstruction_loss import get_reconstruction_loss
from .perceptual_loss import get_perceptual_loss
from .temporal_loss import get_temporal_loss
from .rate_loss import get_rate_loss


class CompositeLoss(nn.Module):
    """
    Composite loss function combining reconstruction, perceptual, temporal, and rate losses.
    """

    def __init__(self, config):
        """
        Initialize the composite loss.

        Args:
            config: Configuration object
        """
        super().__init__()

        # Lambda values from config
        self.lambda_perceptual = config.loss.lambda_perceptual
        self.lambda_temporal = config.loss.lambda_temporal
        self.lambda_rate = config.loss.lambda_rate

        # Initialize loss components
        self.reconstruction_loss = get_reconstruction_loss("mse")

        self.perceptual_loss = get_perceptual_loss(
            loss_type=config.loss.perceptual_type,
            layers=config.loss.perceptual_layers
        )

        self.temporal_loss = get_temporal_loss("consistency")

        self.rate_loss = get_rate_loss("laplacian")

    def forward(self,
               outputs: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute the composite loss.

        Args:
            outputs: Dictionary of model outputs
            targets: Dictionary of targets

        Returns:
            Dictionary with total loss and individual loss components
        """
        # Unpack tensors
        pred_frames = outputs["reconstructed_frames"]
        target_frames = targets["frames"]

        # Initialize loss components
        losses = {}

        # Reconstruction loss
        rec_loss = self.reconstruction_loss(pred_frames, target_frames)
        losses["rec"] = rec_loss

        # Perceptual loss (compute for individual frames, then average)
        perc_loss = 0.0
        batch_size, seq_length = pred_frames.shape[:2]

        for t in range(seq_length):
            frame_loss = self.perceptual_loss(
                pred_frames[:, t], target_frames[:, t])
            perc_loss += frame_loss

        perc_loss /= seq_length
        losses["perc"] = perc_loss

        # Temporal loss (if flow is available)
        temp_loss = 0.0

        if "flows" in outputs:
            flows = outputs["flows"]

            for t in range(seq_length - 1):
                frame_loss = self.temporal_loss(
                    pred_frames[:, t], pred_frames[:, t + 1], flows[:, t])
                temp_loss += frame_loss

            temp_loss /= (seq_length - 1)

        losses["temp"] = temp_loss

        # Rate loss (for base and enhancement layers)
        rate_loss = 0.0

        if "base_latent" in outputs:
            base_rate = self.rate_loss(outputs["base_latent"])
            rate_loss += base_rate
            losses["base_rate"] = base_rate

        if "enhancement_latents" in outputs:
            for i, latent in enumerate(outputs["enhancement_latents"]):
                layer_rate = self.rate_loss(latent)
                rate_loss += layer_rate
                losses[f"enhance_{i}_rate"] = layer_rate

        losses["rate"] = rate_loss

        # Compute total loss
        total_loss = rec_loss + \
                    self.lambda_perceptual * perc_loss + \
                    self.lambda_temporal * temp_loss + \
                    self.lambda_rate * rate_loss

        losses["total"] = total_loss

        return losses


class LayeredLoss(nn.Module):
    """
    Loss function with separate weights for each layer of the codec.
    """

    def __init__(self, config):
        """
        Initialize the layered loss.

        Args:
            config: Configuration object
        """
        super().__init__()

        # Lambda values from config
        self.lambda_perceptual = config.loss.lambda_perceptual
        self.lambda_temporal = config.loss.lambda_temporal
        self.lambda_rate = config.loss.lambda_rate

        # Initialize loss components
        self.reconstruction_loss = get_reconstruction_loss("mse")

        self.perceptual_loss = get_perceptual_loss(
            loss_type=config.loss.perceptual_type,
            layers=config.loss.perceptual_layers
        )

        self.temporal_loss = get_temporal_loss("consistency")

        self.rate_loss = get_rate_loss("laplacian")

        # Layer weights (giving more importance to final output)
        self.num_enhancement_layers = config.enhancement_layer.num_layers
        self.base_weight = 0.5
        self.enhancement_weights = [
            0.5 * (i + 1) / self.num_enhancement_layers
            for i in range(self.num_enhancement_layers)
        ]

    def forward(self,
               outputs: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute the layered loss.

        Args:
            outputs: Dictionary of model outputs
            targets: Dictionary of targets

        Returns:
            Dictionary with total loss and individual loss components
        """
        # Unpack tensors
        target_frames = targets["frames"]

        # Initialize loss components
        losses = {}

        # Base layer reconstruction loss
        base_rec_loss = 0.0
        if "base_output" in outputs:
            base_rec_loss = self.reconstruction_loss(
                outputs["base_output"], target_frames)
            losses["base_rec"] = base_rec_loss

        # Enhancement layer reconstruction losses
        enhance_rec_losses = []

        if "enhancement_outputs" in outputs:
            enhancement_outputs = outputs["enhancement_outputs"]

            for i, output in enumerate(enhancement_outputs):
                layer_loss = self.reconstruction_loss(output, target_frames)
                enhance_rec_losses.append(layer_loss)
                losses[f"enhance_{i}_rec"] = layer_loss

        # Final output reconstruction loss
        final_rec_loss = self.reconstruction_loss(
            outputs["reconstructed_frames"], target_frames)
        losses["final_rec"] = final_rec_loss

        # Perceptual loss on final output
        batch_size, seq_length = outputs["reconstructed_frames"].shape[:2]
        perc_loss = 0.0

        for t in range(seq_length):
            frame_loss = self.perceptual_loss(
                outputs["reconstructed_frames"][:, t], target_frames[:, t])
            perc_loss += frame_loss

        perc_loss /= seq_length
        losses["perc"] = perc_loss

        # Temporal loss on final output
        temp_loss = 0.0

        if "flows" in outputs:
            flows = outputs["flows"]

            for t in range(seq_length - 1):
                frame_loss = self.temporal_loss(
                    outputs["reconstructed_frames"][:, t],
                    outputs["reconstructed_frames"][:, t + 1],
                    flows[:, t])
                temp_loss += frame_loss

            temp_loss /= (seq_length - 1)

        losses["temp"] = temp_loss

        # Rate loss for all layers
        rate_loss = 0.0

        if "base_latent" in outputs:
            base_rate = self.rate_loss(outputs["base_latent"])
            rate_loss += base_rate
            losses["base_rate"] = base_rate

        if "enhancement_latents" in outputs:
            for i, latent in enumerate(outputs["enhancement_latents"]):
                layer_rate = self.rate_loss(latent)
                rate_loss += layer_rate
                losses[f"enhance_{i}_rate"] = layer_rate

        losses["rate"] = rate_loss

        # Compute weighted reconstruction loss
        weighted_rec_loss = self.base_weight * base_rec_loss

        for i, layer_loss in enumerate(enhance_rec_losses):
            weighted_rec_loss += self.enhancement_weights[i] * layer_loss

        # Add final reconstruction loss with highest weight
        weighted_rec_loss += 1.0 * final_rec_loss

        # Compute total loss
        total_loss = weighted_rec_loss + \
                    self.lambda_perceptual * perc_loss + \
                    self.lambda_temporal * temp_loss + \
                    self.lambda_rate * rate_loss

        losses["total"] = total_loss

        return losses


def get_loss_function(config, loss_type: str = "composite") -> nn.Module:
    """
    Get the appropriate loss function.

    Args:
        config: Configuration object
        loss_type: Type of loss function ("composite", "layered")

    Returns:
        Loss function module
    """
    if loss_type.lower() == "composite":
        return CompositeLoss(config)
    elif loss_type.lower() == "layered":
        return LayeredLoss(config)
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")
