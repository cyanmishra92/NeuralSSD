"""
Logging utilities for the layered neural codec.
"""
import os
import time
import json
import logging
from typing import Dict, Any, Optional, Union, List
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime


class Logger:
    """Logger for training and evaluation metrics."""

    def __init__(self, config):
        """
        Initialize the logger.

        Args:
            config: Configuration object
        """
        self.config = config

        # Create log directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{config.logging.experiment_name}_{timestamp}"
        self.log_dir = os.path.join(config.logging.log_dir, self.experiment_name)
        self.plot_dir = os.path.join(config.logging.plot_save_path, self.experiment_name)

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        # Setup file logging
        self.logger = logging.getLogger("layered_codec")
        self.logger.setLevel(logging.INFO)

        # Add file handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, "training.log"))
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(console_handler)

        # Setup TensorBoard
        self.writer = SummaryWriter(self.log_dir)

        # Metrics tracking
        self.metrics = {m: [] for m in config.logging.metrics}
        self.metrics.update({
            "loss": [],
            "loss_rec": [],
            "loss_perc": [],
            "loss_temp": [],
            "loss_rate": [],
            "lr": [],
            "time_per_iter": [],
            "time_data_loading": [],
            "time_forward": [],
            "time_backward": [],
        })

        # Save config
        self.save_config()

        self.global_step = 0
        self.epoch = 0
        self.best_val_metrics = {}

    def save_config(self):
        """Save configuration to a JSON file."""
        config_dict = {}

        # Convert config to dictionary
        for section_name in dir(self.config):
            if not section_name.startswith("__"):
                section = getattr(self.config, section_name)
                if hasattr(section, "__dataclass_fields__"):
                    section_dict = {}
                    for field_name in section.__dataclass_fields__:
                        field_value = getattr(section, field_name)
                        # Handle non-serializable types
                        if isinstance(field_value, (list, tuple)) and len(field_value) > 0 and not isinstance(field_value[0], (int, float, str, bool)):
                            field_value = str(field_value)
                        section_dict[field_name] = field_value
                    config_dict[section_name] = section_dict

        # Save to file
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

    def log_metrics(self, metrics_dict: Dict[str, Union[float, torch.Tensor]],
                   step: Optional[int] = None, phase: str = "train"):
        """
        Log metrics to TensorBoard and internal tracking.

        Args:
            metrics_dict: Dictionary of metric names and values
            step: Global step (if None, use internal counter)
            phase: Either 'train' or 'val'
        """
        step = step if step is not None else self.global_step

        # Log to TensorBoard
        for name, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()

            # Log to internal metrics tracking
            metric_key = f"{phase}_{name}" if phase == "val" else name
            if metric_key in self.metrics:
                self.metrics[metric_key].append(value)

            # Log to TensorBoard
            self.writer.add_scalar(f"{phase}/{name}", value, step)

        # Increment global step for train phase
        if phase == "train":
            self.global_step += 1

    def log_images(self, images_dict: Dict[str, torch.Tensor],
                  step: Optional[int] = None, phase: str = "train"):
        """
        Log images to TensorBoard.

        Args:
            images_dict: Dictionary of image names and tensors (B, C, H, W)
            step: Global step (if None, use internal counter)
            phase: Either 'train' or 'val'
        """
        step = step if step is not None else self.global_step

        for name, images in images_dict.items():
            # Ensure images are on CPU and in correct format
            if isinstance(images, torch.Tensor):
                # Normalize to [0, 1] if needed
                if images.min() < 0 or images.max() > 1:
                    images = (images - images.min()) / (images.max() - images.min() + 1e-8)

                # Convert to uint8 if needed
                if images.dtype != torch.uint8 and images.max() <= 1.0:
                    images = (images * 255).to(torch.uint8)

                # Log to TensorBoard
                self.writer.add_images(f"{phase}/{name}", images, step)

    def save_reconstructions(self, original: torch.Tensor, reconstructed: torch.Tensor,
                            step: int, max_samples: int = 4):
        """
        Save original and reconstructed images to disk.

        Args:
            original: Original frames tensor (B, T, C, H, W)
            reconstructed: Reconstructed frames tensor (B, T, C, H, W)
            step: Global step
            max_samples: Maximum number of samples to save
        """
        batch_size = min(original.size(0), max_samples)
        seq_length = original.size(1)

        # Select a subset of frames to visualize (first, middle, last)
        frame_indices = [0, seq_length // 2, seq_length - 1]

        for batch_idx in range(batch_size):
            fig, axes = plt.subplots(2, len(frame_indices), figsize=(4 * len(frame_indices), 8))

            for i, frame_idx in enumerate(frame_indices):
                # Original frame
                orig_frame = original[batch_idx, frame_idx].cpu().permute(1, 2, 0).numpy()
                orig_frame = np.clip(orig_frame, 0, 1)
                axes[0, i].imshow(orig_frame)
                axes[0, i].set_title(f"Original (Frame {frame_idx})")
                axes[0, i].axis("off")

                # Reconstructed frame
                recon_frame = reconstructed[batch_idx, frame_idx].cpu().permute(1, 2, 0).numpy()
                recon_frame = np.clip(recon_frame, 0, 1)
                axes[1, i].imshow(recon_frame)
                axes[1, i].set_title(f"Reconstructed (Frame {frame_idx})")
                axes[1, i].axis("off")

            plt.tight_layout()
            save_path = os.path.join(
                self.plot_dir, f"recon_batch{batch_idx}_step{step}.png")
            plt.savefig(save_path)
            plt.close(fig)

    def plot_metrics(self, save_path: Optional[str] = None):
        """
        Plot training metrics and save to disk.

        Args:
            save_path: Path to save the plot (if None, use default)
        """
        if not save_path:
            save_path = os.path.join(self.log_dir, "metrics.png")

        # Get metrics to plot (filter out timing metrics)
        plot_metrics = {k: v for k, v in self.metrics.items()
                      if len(v) > 0 and not k.startswith("time_")}

        n_metrics = len(plot_metrics)
        if n_metrics == 0:
            return

        # Create plot
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics))
        if n_metrics == 1:
            axes = [axes]

        for ax, (name, values) in zip(axes, plot_metrics.items()):
            ax.plot(values)
            ax.set_title(name)
            ax.set_xlabel("Iterations")
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def start_epoch(self, epoch: int):
        """
        Start a new epoch.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch
        self.logger.info(f"Starting epoch {epoch}")
        self.epoch_start_time = time.time()

    def end_epoch(self, learning_rate: float):
        """
        End an epoch and log summary statistics.

        Args:
            learning_rate: Current learning rate
        """
        epoch_duration = time.time() - self.epoch_start_time

        # Calculate average metrics for the epoch
        epoch_metrics = {}
        for name, values in self.metrics.items():
            if name.startswith("time_") or name == "lr":
                continue
            if len(values) > 0:
                # Get values from this epoch only
                epoch_values = values[-self.global_step:]
                if len(epoch_values) > 0:
                    epoch_metrics[name] = np.mean(epoch_values)

        # Log metrics
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items())
        self.logger.info(
            f"Epoch {self.epoch} completed in {epoch_duration:.2f}s, lr={learning_rate:.6f}, {metrics_str}")

        # Add learning rate to metrics
        self.metrics["lr"].append(learning_rate)

        # Plot metrics
        if self.epoch % 5 == 0:
            self.plot_metrics()

    def update_best_metrics(self, metrics_dict: Dict[str, float]) -> bool:
        """
        Update best validation metrics and return whether model improved.

        Args:
            metrics_dict: Dictionary of metric names and values

        Returns:
            True if model improved on any metric, False otherwise
        """
        improved = False

        for name, value in metrics_dict.items():
            # For metrics where higher is better (e.g., PSNR, MS-SSIM)
            if name in ["psnr", "msssim"]:
                if name not in self.best_val_metrics or value > self.best_val_metrics[name]:
                    self.best_val_metrics[name] = value
                    improved = True
                    self.logger.info(f"New best {name}: {value:.4f}")

            # For metrics where lower is better (e.g., loss, bpp)
            elif name in ["loss", "bpp"]:
                if name not in self.best_val_metrics or value < self.best_val_metrics[name]:
                    self.best_val_metrics[name] = value
                    improved = True
                    self.logger.info(f"New best {name}: {value:.4f}")

        return improved

    def close(self):
        """Close the logger and release resources."""
        self.writer.close()

        # Plot final metrics
        self.plot_metrics()

        # Log best metrics
        if self.best_val_metrics:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.best_val_metrics.items())
            self.logger.info(f"Best validation metrics: {metrics_str}")
