"""
Main codec module that integrates all components of the layered neural codec.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any

from .feature_extraction import get_feature_extractor
from .motion_estimation import MotionEstimation
from .base_layer import BaseLayerCodec
from .enhancement_layer import EnhancementLayerStack


class LayeredNeuralCodec(nn.Module):
    """
    Layered neural codec for video compression.
    """

    def __init__(self, config):
        """
        Initialize the layered neural codec.

        Args:
            config: Configuration object
        """
        super().__init__()

        self.config = config

        # Feature extraction
        self.feature_extractor = get_feature_extractor(config)

        # Motion estimation
        self.motion_estimation = MotionEstimation(config)

        # Base layer codec
        self.base_layer = BaseLayerCodec(config)

        # Enhancement layers
        self.enhancement_layers = EnhancementLayerStack(config)

        # Anchor frame interval
        self.anchor_interval = config.training.anchor_frame_interval

    def encode_anchor_frame(self, frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode an anchor frame.

        Args:
            frame: Input frame of shape [B, C, H, W]

        Returns:
            Dictionary with encoder outputs
        """
        # Extract features
        features = self.feature_extractor(frame)

        # Encode with base layer
        base_outputs = self.base_layer(frame)

        return {
            "features": features,
            "base_outputs": base_outputs,
            "base_latent": base_outputs["latent"]
        }

    def encode_non_anchor_frame(self,
                               frame: torch.Tensor,
                               reference_frame: torch.Tensor,
                               reference_rec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode a non-anchor frame using motion compensation.

        Args:
            frame: Input frame of shape [B, C, H, W]
            reference_frame: Reference frame of shape [B, C, H, W]
            reference_rec: Reconstructed reference frame of shape [B, C, H, W]

        Returns:
            Dictionary with encoder outputs
        """
        # Extract features
        features = self.feature_extractor(frame)

        # Estimate motion between reference and current frame
        motion_outputs = self.motion_estimation(reference_rec, frame)

        # Get residual frame
        residual = motion_outputs["residual"]

        # Encode residual with base layer
        base_outputs = self.base_layer(residual)

        return {
            "features": features,
            "motion_outputs": motion_outputs,
            "base_outputs": base_outputs,
            "base_latent": base_outputs["latent"],
            "flow": motion_outputs["flow"]
        }

    def decode_anchor_frame(self, base_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode an anchor frame.

        Args:
            base_latent: Base layer latent of shape [B, C, H, W]

        Returns:
            Dictionary with decoder outputs
        """
        # Decode with base layer
        base_outputs = self.base_layer.decoder(base_latent)
        base_rec = base_outputs["output"]

        return {
            "base_outputs": base_outputs,
            "base_reconstruction": base_rec,
            "reconstruction": base_rec
        }

    def decode_non_anchor_frame(self,
                               base_latent: torch.Tensor,
                               reference_rec: torch.Tensor,
                               flow: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode a non-anchor frame using motion compensation.

        Args:
            base_latent: Base layer latent of shape [B, C, H, W]
            reference_rec: Reconstructed reference frame of shape [B, C, H, W]
            flow: Optical flow from reference to current frame of shape [B, 2, H, W]

        Returns:
            Dictionary with decoder outputs
        """
        # Decode residual with base layer
        base_outputs = self.base_layer.decoder(base_latent)
        residual_rec = base_outputs["output"]

        # Warp reference frame using flow
        warped_reference = self.motion_estimation.warp_frame(reference_rec, flow)

        # Add residual to warped reference
        rec = warped_reference + residual_rec

        return {
            "base_outputs": base_outputs,
            "base_reconstruction": residual_rec,
            "warped_reference": warped_reference,
            "reconstruction": rec
        }

    def apply_enhancement(self,
                         original_frame: torch.Tensor,
                         base_reconstruction: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply enhancement layers to improve reconstruction quality.

        Args:
            original_frame: Original frame of shape [B, C, H, W]
            base_reconstruction: Base layer reconstruction of shape [B, C, H, W]

        Returns:
            Dictionary with enhancement outputs
        """
        # Apply enhancement layers
        enhancement_outputs = self.enhancement_layers(
            original_frame, base_reconstruction)

        return enhancement_outputs

    def forward(self, frames: torch.Tensor) -> Dict[str, Any]:
        """
        Process a sequence of frames.

        Args:
            frames: Input frames of shape [B, T, C, H, W]

        Returns:
            Dictionary with codec outputs
        """
        batch_size, seq_length, C, H, W = frames.shape

        # Initialize outputs
        reconstructed_frames = []
        base_reconstructions = []
        enhancement_outputs_list = []
        base_latents = []
        enhancement_latents_list = []
        flows = []

        # Process each frame
        for t in range(seq_length):
            current_frame = frames[:, t]

            # Check if anchor frame
            is_anchor = (t % self.anchor_interval == 0)

            if is_anchor:
                # Process anchor frame
                encode_outputs = self.encode_anchor_frame(current_frame)
                decode_outputs = self.decode_anchor_frame(
                    encode_outputs["base_latent"])

            else:
                # Process non-anchor frame
                reference_frame = frames[:, t - 1]
                reference_rec = reconstructed_frames[-1]

                encode_outputs = self.encode_non_anchor_frame(
                    current_frame, reference_frame, reference_rec)

                decode_outputs = self.decode_non_anchor_frame(
                    encode_outputs["base_latent"], reference_rec, encode_outputs["flow"])

                # Store flow
                flows.append(encode_outputs["flow"])

            # Apply enhancement layers
            if self.training or self.config.enhancement_layer.num_layers > 0:
                enhancement_outputs = self.apply_enhancement(
                    current_frame, decode_outputs["reconstruction"])

                # Store outputs
                final_rec = enhancement_outputs["final_output"]
                enhancement_outputs_list.append(enhancement_outputs)

                if "latents" in enhancement_outputs:
                    enhancement_latents_list.append(enhancement_outputs["latents"])
            else:
                # No enhancement in evaluation mode if no enhancement layers
                final_rec = decode_outputs["reconstruction"]

            # Store outputs
            reconstructed_frames.append(final_rec)
            base_reconstructions.append(decode_outputs["base_reconstruction"])
            base_latents.append(encode_outputs["base_latent"])

        # Stack outputs along time dimension
        reconstructed_frames = torch.stack(reconstructed_frames, dim=1)
        base_reconstructions = torch.stack(base_reconstructions, dim=1)

        # Prepare flow tensor (with zeros for first frame)
        if flows:
            # Create zero flow for first frame
            zero_flow = torch.zeros_like(flows[0])
            flows = [zero_flow] + flows
            flows = torch.stack(flows, dim=1)
        else:
            flows = torch.zeros((batch_size, seq_length, 2, H, W), device=frames.device)

        # Prepare latent tensors
        base_latents = torch.stack(base_latents, dim=1)

        # Prepare outputs
        outputs = {
            "reconstructed_frames": reconstructed_frames,
            "base_reconstructions": base_reconstructions,
            "base_latent": base_latents,
            "flows": flows
        }

        # Add enhancement outputs if available
        if enhancement_outputs_list:
            # Extract layer outputs
            layer_outputs = []
            for i in range(self.config.enhancement_layer.num_layers):
                layer_i_outputs = []
                for t in range(seq_length):
                    if t < len(enhancement_outputs_list):
                        layer_output = enhancement_outputs_list[t]["layer_outputs"][i]["output"]
                        layer_i_outputs.append(layer_output)
                    else:
                        # Pad with zeros if needed
                        layer_i_outputs.append(torch.zeros_like(reconstructed_frames[:, 0]))

                layer_outputs.append(torch.stack(layer_i_outputs, dim=1))

            outputs["enhancement_outputs"] = layer_outputs

        # Add latents if available
        if enhancement_latents_list:
            # Organize latents by layer
            enhance_latents = []
            for i in range(self.config.enhancement_layer.num_layers):
                layer_i_latents = []
                for t in range(seq_length):
                    if t < len(enhancement_latents_list):
                        layer_latent = enhancement_latents_list[t][i]
                        layer_i_latents.append(layer_latent)
                    else:
                        # Pad with zeros if needed
                        layer_i_latents.append(torch.zeros_like(base_latents[:, 0]))

                enhance_latents.append(torch.stack(layer_i_latents, dim=1))

            outputs["enhancement_latents"] = enhance_latents

        return outputs
