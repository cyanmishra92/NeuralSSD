"""
Visualization utilities for the layered neural codec.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import io
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor to a numpy array for visualization.

    Args:
        tensor: Input tensor of shape [C, H, W] or [H, W]

    Returns:
        Numpy array suitable for visualization
    """
    # Detach from graph and move to CPU
    array = tensor.detach().cpu().numpy()

    # Handle different number of dimensions
    if array.ndim == 2:  # [H, W]
        return array
    elif array.ndim == 3:  # [C, H, W]
        if array.shape[0] == 1:  # Single-channel image
            return array[0]
        elif array.shape[0] == 3:  # RGB image
            return np.transpose(array, (1, 2, 0))
        else:
            # For other channel counts, take first three or first one
            if array.shape[0] >= 3:
                return np.transpose(array[:3], (1, 2, 0))
            else:
                return array[0]
    else:
        raise ValueError(f"Unsupported tensor shape: {array.shape}")


def visualize_frame_comparison(original: torch.Tensor,
                              reconstructed: torch.Tensor,
                              base_reconstruction: Optional[torch.Tensor] = None,
                              title: str = "Frame Comparison",
                              save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Visualize original and reconstructed frames side by side.

    Args:
        original: Original frame tensor of shape [C, H, W]
        reconstructed: Reconstructed frame tensor of shape [C, H, W]
        base_reconstruction: Optional base layer reconstruction of shape [C, H, W]
        title: Plot title
        save_path: Path to save the plot (if None, return the figure)

    Returns:
        Matplotlib figure if save_path is None, otherwise None
    """
    # Convert tensors to numpy arrays
    orig_np = tensor_to_numpy(original)
    recon_np = tensor_to_numpy(reconstructed)

    # Create figure
    if base_reconstruction is not None:
        base_np = tensor_to_numpy(base_reconstruction)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(np.clip(orig_np, 0, 1))
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(np.clip(base_np, 0, 1))
        axes[1].set_title("Base Reconstruction")
        axes[1].axis("off")

        axes[2].imshow(np.clip(recon_np, 0, 1))
        axes[2].set_title("Final Reconstruction")
        axes[2].axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(np.clip(orig_np, 0, 1))
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(np.clip(recon_np, 0, 1))
        axes[1].set_title("Reconstructed")
        axes[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    # Save or return figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig


def visualize_sequence(frames: torch.Tensor,
                      title: str = "Video Sequence",
                      max_frames: int = 8,
                      save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Visualize a sequence of frames.

    Args:
        frames: Sequence of frames of shape [T, C, H, W]
        title: Plot title
        max_frames: Maximum number of frames to display
        save_path: Path to save the plot (if None, return the figure)

    Returns:
        Matplotlib figure if save_path is None, otherwise None
    """
    # Limit number of frames
    num_frames = min(frames.shape[0], max_frames)

    # Create figure
    fig, axes = plt.subplots(1, num_frames, figsize=(2 * num_frames, 3))

    if num_frames == 1:
        axes = [axes]

    for i in range(num_frames):
        frame_np = tensor_to_numpy(frames[i])
        axes[i].imshow(np.clip(frame_np, 0, 1))
        axes[i].set_title(f"Frame {i}")
        axes[i].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    # Save or return figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig


def visualize_flow(flow: torch.Tensor,
                  original: Optional[torch.Tensor] = None,
                  title: str = "Optical Flow",
                  save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Visualize optical flow using color coding.

    Args:
        flow: Optical flow tensor of shape [2, H, W]
        original: Optional original frame for overlay of shape [C, H, W]
        title: Plot title
        save_path: Path to save the plot (if None, return the figure)

    Returns:
        Matplotlib figure if save_path is None, otherwise None
    """
    # Convert flow to numpy
    flow_np = flow.detach().cpu().numpy()

    # Calculate magnitude and angle
    fx, fy = flow_np[0], flow_np[1]
    magnitude = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)

    # Normalize magnitude for better visualization
    magnitude_norm = Normalize()(magnitude)

    # Create HSV image (hue=angle, saturation=1, value=magnitude)
    hsv = np.zeros((flow_np.shape[1], flow_np.shape[2], 3), dtype=np.float32)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi)  # Hue (0-1)
    hsv[..., 1] = 1.0  # Saturation
    hsv[..., 2] = magnitude_norm  # Value

    # Convert HSV to RGB
    from matplotlib.colors import hsv_to_rgb
    flow_rgb = hsv_to_rgb(hsv)

    # Create figure
    if original is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Original frame
        orig_np = tensor_to_numpy(original)
        axes[0].imshow(np.clip(orig_np, 0, 1))
        axes[0].set_title("Original Frame")
        axes[0].axis("off")

        # Flow visualization
        axes[1].imshow(flow_rgb)
        axes[1].set_title("Optical Flow")
        axes[1].axis("off")
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(flow_rgb)
        ax.set_title("Optical Flow")
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    # Add colorbar with magnitude reference
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=magnitude.max())),
                       cax=cbar_ax)
    cbar.set_label("Magnitude")

    # Save or return figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig


def visualize_latent(latent: torch.Tensor,
                    title: str = "Latent Representation",
                    save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Visualize latent representation.

    Args:
        latent: Latent tensor of shape [C, H, W]
        title: Plot title
        save_path: Path to save the plot (if None, return the figure)

    Returns:
        Matplotlib figure if save_path is None, otherwise None
    """
    # Get number of channels
    channels = latent.shape[0]

    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(channels)))

    # Create figure
    fig = plt.figure(figsize=(2 * grid_size, 2 * grid_size))
    gs = gridspec.GridSpec(grid_size, grid_size)

    # Normalize latent for better visualization
    latent_np = latent.detach().cpu().numpy()

    for i in range(min(channels, grid_size * grid_size)):
        ax = plt.subplot(gs[i])

        # Get channel and normalize
        channel = latent_np[i]
        vmin, vmax = channel.min(), channel.max()

        # Display channel
        ax.imshow(channel, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Ch {i}")
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    # Save or return figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig


def visualize_enhancement_layers(original: torch.Tensor,
                                base_output: torch.Tensor,
                                enhancement_outputs: List[torch.Tensor],
                                title: str = "Enhancement Layers",
                                save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Visualize output of each enhancement layer.

    Args:
        original: Original frame tensor of shape [C, H, W]
        base_output: Base layer output tensor of shape [C, H, W]
        enhancement_outputs: List of enhancement layer outputs, each of shape [C, H, W]
        title: Plot title
        save_path: Path to save the plot (if None, return the figure)

    Returns:
        Matplotlib figure if save_path is None, otherwise None
    """
    # Determine total number of images to display
    num_images = 2 + len(enhancement_outputs)  # Original, base, enhancements

    # Create figure
    fig, axes = plt.subplots(1, num_images, figsize=(3 * num_images, 3))

    # Original frame
    orig_np = tensor_to_numpy(original)
    axes[0].imshow(np.clip(orig_np, 0, 1))
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Base layer output
    base_np = tensor_to_numpy(base_output)
    axes[1].imshow(np.clip(base_np, 0, 1))
    axes[1].set_title("Base Layer")
    axes[1].axis("off")

    # Enhancement layer outputs
    for i, enhancement in enumerate(enhancement_outputs):
        enhance_np = tensor_to_numpy(enhancement)
        axes[i + 2].imshow(np.clip(enhance_np, 0, 1))
        axes[i + 2].set_title(f"Enhancement {i+1}")
        axes[i + 2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    # Save or return figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig


def visualize_rd_curve(results: Dict[str, List[float]],
                      title: str = "Rate-Distortion Curve",
                      save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Visualize rate-distortion curve.

    Args:
        results: Dictionary with 'bpp', 'psnr', and 'msssim' lists
        title: Plot title
        save_path: Path to save the plot (if None, return the figure)

    Returns:
        Matplotlib figure if save_path is None, otherwise None
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR vs BPP
    axes[0].plot(results["bpp"], results["psnr"], "o-", linewidth=2)
    axes[0].set_xlabel("Bits Per Pixel (BPP)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Rate-Distortion Curve (PSNR)")
    axes[0].grid(True)

    # MS-SSIM vs BPP
    axes[1].plot(results["bpp"], results["msssim"], "o-", linewidth=2)
    axes[1].set_xlabel("Bits Per Pixel (BPP)")
    axes[1].set_ylabel("MS-SSIM")
    axes[1].set_title("Rate-Distortion Curve (MS-SSIM)")
    axes[1].grid(True)

    fig.suptitle(title)
    plt.tight_layout()

    # Save or return figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig


def create_comparison_grid(original_frames: torch.Tensor,
                          reconstructed_frames: torch.Tensor,
                          row_step: int = 2,
                          col_step: int = 2,
                          max_rows: int = 3,
                          max_cols: int = 4,
                          title: str = "Reconstruction Comparison",
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Create a grid comparing original and reconstructed frames from a sequence.

    Args:
        original_frames: Original frames tensor of shape [T, C, H, W]
        reconstructed_frames: Reconstructed frames tensor of shape [T, C, H, W]
        row_step: Step size for selecting frames for rows
        col_step: Step size for selecting different parts of frames for columns
        max_rows: Maximum number of rows (frames) to display
        max_cols: Maximum number of columns (crops) to display
        title: Plot title
        save_path: Path to save the plot (if None, return the figure)

    Returns:
        Matplotlib figure if save_path is None, otherwise None
    """
    # Get sequence length and frame dimensions
    seq_length, C, H, W = original_frames.shape

    # Select frames for rows
    row_indices = list(range(0, seq_length, row_step))[:max_rows]
    num_rows = len(row_indices)

    # Determine crop regions for columns
    crop_size = min(H, W) // 4
    col_centers = []

    # Create crops across the diagonal of the image
    for i in range(0, max_cols):
        center_h = (H // 2) + ((i - max_cols // 2) * H // (max_cols + 1))
        center_w = (W // 2) + ((i - max_cols // 2) * W // (max_cols + 1))
        col_centers.append((center_h, center_w))

    # Create figure with gridspec
    fig = plt.figure(figsize=(3 * 2 * len(col_centers), 3 * num_rows))
    gs = gridspec.GridSpec(num_rows, len(col_centers) * 2)

    for row_idx, frame_idx in enumerate(row_indices):
        # Get original and reconstructed frames
        orig = original_frames[frame_idx]
        recon = reconstructed_frames[frame_idx]

        orig_np = tensor_to_numpy(orig)
        recon_np = tensor_to_numpy(recon)

        for col_idx, (center_h, center_w) in enumerate(col_centers):
            # Define crop region
            h_start = max(0, center_h - crop_size // 2)
            h_end = min(H, center_h + crop_size // 2)
            w_start = max(0, center_w - crop_size // 2)
            w_end = min(W, center_w + crop_size // 2)

            # Crop original and reconstructed frames
            orig_crop = orig_np[h_start:h_end, w_start:w_end]
            recon_crop = recon_np[h_start:h_end, w_start:w_end]

            # Create subplots for original and reconstructed crops
            ax_orig = plt.subplot(gs[row_idx, col_idx * 2])
            ax_recon = plt.subplot(gs[row_idx, col_idx * 2 + 1])

            # Display crops
            ax_orig.imshow(np.clip(orig_crop, 0, 1))
            ax_orig.set_title(f"Original (F{frame_idx})")
            ax_orig.axis("off")

            ax_recon.imshow(np.clip(recon_crop, 0, 1))
            ax_recon.set_title(f"Reconstructed (F{frame_idx})")
            ax_recon.axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    # Save or return figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig


def save_gif(frames: torch.Tensor,
            filepath: str,
            fps: int = 10,
            max_size: Optional[Tuple[int, int]] = None):
    """
    Save a sequence of frames as an animated GIF.

    Args:
        frames: Sequence of frames of shape [T, C, H, W]
        filepath: Path to save the GIF
        fps: Frames per second
        max_size: Maximum size (width, height) to resize to
    """
    # Convert frames to PIL images
    pil_frames = []
    for i in range(frames.shape[0]):
        frame_np = tensor_to_numpy(frames[i])
        frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)

        if frame_np.shape[-1] == 3:  # RGB
            pil_img = Image.fromarray(frame_np)
        else:  # Grayscale
            pil_img = Image.fromarray(frame_np, mode="L")

        # Resize if necessary
        if max_size is not None:
            pil_img.thumbnail(max_size, Image.LANCZOS)

        pil_frames.append(pil_img)

    # Save as GIF
    duration = 1000 / fps  # Convert fps to duration in ms
    pil_frames[0].save(
        filepath,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=True,
        duration=duration,
        loop=0
    )


def visualize_metrics(metrics: Dict[str, List[float]],
                     title: str = "Training Metrics",
                     save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Visualize training or evaluation metrics.

    Args:
        metrics: Dictionary of metric names and values
        title: Plot title
        save_path: Path to save the plot (if None, return the figure)

    Returns:
        Matplotlib figure if save_path is None, otherwise None
    """
    # Filter out metrics with no values
    metrics = {k: v for k, v in metrics.items() if len(v) > 0}

    # Determine number of metrics to plot
    num_metrics = len(metrics)

    if num_metrics == 0:
        return None

    # Create figure
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 3 * num_metrics))

    if num_metrics == 1:
        axes = [axes]

    # Plot each metric
    for i, (name, values) in enumerate(metrics.items()):
        axes[i].plot(values)
        axes[i].set_title(name)
        axes[i].set_xlabel("Iterations")
        axes[i].grid(True)

    fig.suptitle(title)
    plt.tight_layout()

    # Save or return figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig
