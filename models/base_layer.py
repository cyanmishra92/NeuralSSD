"""
Base layer encoder and decoder modules for the layered neural codec.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation."""

    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: int = 3,
                stride: int = 1,
                activation: str = "relu"):
        """
        Initialize the convolutional block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            stride: Stride
            activation: Activation function
        """
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(
            num_groups=min(32, out_channels), num_channels=out_channels)

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leakyrelu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SelfAttention(nn.Module):
    """Self-attention module for feature maps."""

    def __init__(self, channels: int, reduction: int = 8):
        """
        Initialize the self-attention module.

        Args:
            channels: Number of input channels
            reduction: Channel reduction factor
        """
        super().__init__()

        self.query = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to feature maps.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Attention-weighted tensor of shape [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Compute query, key, value projections
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, H*W, C//r]
        k = self.key(x).view(B, -1, H * W)  # [B, C//r, H*W]
        v = self.value(x).view(B, -1, H * W)  # [B, C, H*W]

        # Compute attention weights
        attn = F.softmax(torch.bmm(q, k), dim=-1)  # [B, H*W, H*W]

        # Apply attention weights to value
        out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, H*W]
        out = out.view(B, C, H, W)

        # Apply learnable weight
        out = self.gamma * out + x

        return out


class BaseLayerEncoder(nn.Module):
    """Base layer encoder for the layered neural codec."""

    def __init__(self, config):
        """
        Initialize the base layer encoder.

        Args:
            config: Configuration object
        """
        super().__init__()

        self.input_channels = 3  # RGB images
        self.latent_dim = config.base_layer.latent_dim
        self.channels = config.base_layer.encoder_channels
        self.kernel_size = config.base_layer.kernel_size
        self.activation = config.base_layer.activation
        self.use_attention = config.base_layer.use_attention

        # Initial convolution
        self.initial_conv = ConvBlock(
            self.input_channels, self.channels[0],
            kernel_size=self.kernel_size, activation=self.activation
        )

        # Downsampling layers
        self.down_layers = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.down_layers.append(nn.Sequential(
                ConvBlock(
                    self.channels[i], self.channels[i],
                    kernel_size=self.kernel_size, activation=self.activation
                ),
                ConvBlock(
                    self.channels[i], self.channels[i + 1],
                    kernel_size=self.kernel_size, stride=2, activation=self.activation
                )
            ))

        # Attention module
        if self.use_attention:
            self.attention = SelfAttention(self.channels[-1])

        # Projection to latent space
        self.to_latent = nn.Conv2d(
            self.channels[-1], self.latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Dictionary with 'latent' and intermediate feature maps
        """
        features = {}

        # Initial convolution
        x = self.initial_conv(x)
        features["initial"] = x

        # Downsampling
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            features[f"down_{i}"] = x

        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
            features["attention"] = x

        # Project to latent space
        latent = self.to_latent(x)
        features["latent"] = latent

        return {
            "latent": latent,
            "features": features
        }


class BaseLayerDecoder(nn.Module):
    """Base layer decoder for the layered neural codec."""

    def __init__(self, config):
        """
        Initialize the base layer decoder.

        Args:
            config: Configuration object
        """
        super().__init__()

        self.latent_dim = config.base_layer.latent_dim
        self.channels = list(reversed(config.base_layer.decoder_channels))
        self.kernel_size = config.base_layer.kernel_size
        self.activation = config.base_layer.activation
        self.use_attention = config.base_layer.use_attention

        # From latent to features
        self.from_latent = ConvBlock(
            self.latent_dim, self.channels[0],
            kernel_size=1, activation=self.activation
        )

        # Attention module
        if self.use_attention:
            self.attention = SelfAttention(self.channels[0])

        # Upsampling layers
        self.up_layers = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.up_layers.append(nn.Sequential(
                ConvBlock(
                    self.channels[i], self.channels[i],
                    kernel_size=self.kernel_size, activation=self.activation
                ),
                nn.ConvTranspose2d(
                    self.channels[i], self.channels[i + 1],
                    kernel_size=4, stride=2, padding=1
                ),
                nn.GroupNorm(
                    num_groups=min(32, self.channels[i + 1]),
                    num_channels=self.channels[i + 1]
                ),
                #getattr(nn, self.activation.upper())()
                getattr(nn, "ReLU")()
            ))

        # Final convolution
        self.final_conv = nn.Sequential(
            ConvBlock(
                self.channels[-1], self.channels[-1],
                kernel_size=self.kernel_size, activation=self.activation
            ),
            nn.Conv2d(self.channels[-1], 3, kernel_size=3, padding=1)
        )

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode latent representation to image.

        Args:
            latent: Latent tensor of shape [B, C, H, W]

        Returns:
            Dictionary with 'output' and intermediate feature maps
        """
        features = {}

        # From latent to features
        x = self.from_latent(latent)
        features["from_latent"] = x

        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
            features["attention"] = x

        # Upsampling
        for i, layer in enumerate(self.up_layers):
            x = layer(x)
            features[f"up_{i}"] = x

        # Final convolution
        output = self.final_conv(x)
        features["output"] = output

        return {
            "output": output,
            "features": features
        }


class BaseLayerCodec(nn.Module):
    """Combined base layer encoder and decoder."""

    def __init__(self, config):
        """
        Initialize the base layer codec.

        Args:
            config: Configuration object
        """
        super().__init__()

        self.encoder = BaseLayerEncoder(config)
        self.decoder = BaseLayerDecoder(config)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Encode and decode input.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Dictionary with encoder and decoder outputs
        """
        # Encode
        encoder_out = self.encoder(x)
        latent = encoder_out["latent"]

        # Add quantization noise during training (straight-through estimator)
        if self.training:
            noise = torch.randn_like(latent) * 0.01
            latent_noisy = latent + noise
            latent_quantized = latent_noisy
        else:
            # In evaluation, we could apply actual quantization here
            latent_quantized = latent

        # Decode
        decoder_out = self.decoder(latent_quantized)

        return {
            "latent": latent,
            "latent_quantized": latent_quantized,
            "output": decoder_out["output"],
            "encoder_features": encoder_out["features"],
            "decoder_features": decoder_out["features"]
        }
