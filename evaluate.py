"""
Evaluation script for the layered neural codec.
"""
import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

from config.default_config import Config, default_config
from data.dataloader import create_dataloaders, prepare_batch
from models.codec import LayeredNeuralCodec
from utils.checkpoint import load_checkpoint
from utils.metrics import calculate_metrics, calculate_rd_curve


def evaluate_model(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device,
                  save_dir: str,
                  prefix: str = "") -> Dict[str, float]:
    """
    Evaluate the model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to evaluate on
        save_dir: Directory to save results
        prefix: Prefix for saved files

    Returns:
        Dictionary of average metrics
    """
    model.eval()

    # Initialize metrics
    metrics_list = []
    sequence_metrics = {}

    # Create output directory
    os.makedirs(os.path.join(save_dir, "reconstructions"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "metrics"), exist_ok=True)

    # Disable gradient computation
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move batch to device
            batch = prepare_batch(batch, device)
            frames = batch["frames"]
            sequence_names = batch["sequence_name"]

            # Forward pass
            outputs = model(frames)

            # Calculate metrics
            metrics = calculate_metrics(
                outputs["reconstructed_frames"], frames,
                [outputs["base_latent"]] + outputs.get("enhancement_latents", [])
            )

            # Store metrics
            for i in range(len(sequence_names)):
                seq_name = sequence_names[i]

                # Extract metrics for this sequence
                seq_metrics = {
                    "psnr": metrics["psnr"][i].item(),
                    "ssim": metrics["ssim"][i].item(),
                    "msssim": metrics["msssim"][i].item(),
                    "bpp": metrics["bpp"][i].item()
                }

                # Store metrics
                if seq_name not in sequence_metrics:
                    sequence_metrics[seq_name] = []

                sequence_metrics[seq_name].append(seq_metrics)
                metrics_list.append(seq_metrics)

                # Save reconstructions
                if batch_idx % 10 == 0:  # Save every 10th batch to avoid excessive storage
                    save_reconstructions(
                        frames[i:i+1], outputs["reconstructed_frames"][i:i+1],
                        os.path.join(save_dir, "reconstructions"),
                        f"{prefix}_{seq_name}_{batch_idx}_{i}"
                    )

    # Calculate average metrics
    avg_metrics = {}
    for key in ["psnr", "ssim", "msssim", "bpp"]:
        avg_metrics[key] = np.mean([m[key] for m in metrics_list])

    # Calculate per-sequence metrics
    per_sequence_metrics = {}
    for seq_name, metrics in sequence_metrics.items():
        per_sequence_metrics[seq_name] = {}
        for key in ["psnr", "ssim", "msssim", "bpp"]:
            per_sequence_metrics[seq_name][key] = np.mean([m[key] for m in metrics])

    # Save metrics
    save_metrics(avg_metrics, per_sequence_metrics, os.path.join(save_dir, "metrics"), prefix)

    # Print average metrics
    print(f"Average metrics:")
    print(f"PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"SSIM: {avg_metrics['ssim']:.4f}")
    print(f"MS-SSIM: {avg_metrics['msssim']:.4f}")
    print(f"BPP: {avg_metrics['bpp']:.4f}")

    return avg_metrics


def evaluate_model_rd(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     device: torch.device,
                     save_dir: str,
                     rate_points: List[float] = [0.1, 0.2, 0.5, 1.0]) -> Dict[str, List[float]]:
    """
    Evaluate the model at different rate points to generate a rate-distortion curve.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to evaluate on
        save_dir: Directory to save results
        rate_points: List of target bits per pixel values

    Returns:
        Dictionary of metrics for different rate points
    """
    # Initialize metrics
    rd_metrics = {
        "bpp": [],
        "psnr": [],
        "ssim": [],
        "msssim": []
    }

    # Set baseline lambda_rate
    original_lambda_rate = model.config.loss.lambda_rate

    # Evaluate at different rate points
    for rate_point in rate_points:
        # Adjust lambda_rate to target different rate points
        # Higher lambda_rate leads to lower bitrate
        if rate_point < 0.2:
            model.config.loss.lambda_rate = original_lambda_rate * 5.0
        elif rate_point < 0.5:
            model.config.loss.lambda_rate = original_lambda_rate * 2.0
        elif rate_point < 0.8:
            model.config.loss.lambda_rate = original_lambda_rate
        else:
            model.config.loss.lambda_rate = original_lambda_rate * 0.5

        # Create prefix for this rate point
        prefix = f"rd_bpp{rate_point:.2f}"

        # Evaluate model
        metrics = evaluate_model(model, dataloader, device, save_dir, prefix)

        # Store metrics
        for key in rd_metrics:
            rd_metrics[key].append(metrics[key])

    # Reset lambda_rate
    model.config.loss.lambda_rate = original_lambda_rate

    # Save RD curve
    plot_rd_curve(rd_metrics, os.path.join(save_dir, "rd_curve.png"))

    # Calculate BD-Rate and BD-PSNR
    bd_metrics = calculate_rd_curve(rd_metrics)

    # Print BD metrics
    print(f"BD-Rate: {bd_metrics['bd_rate']:.2f}%")
    print(f"BD-PSNR: {bd_metrics['bd_psnr']:.2f} dB")

    return rd_metrics


def save_reconstructions(original: torch.Tensor,
                        reconstructed: torch.Tensor,
                        save_dir: str,
                        prefix: str):
    """
    Save original and reconstructed frames.

    Args:
        original: Original frames tensor of shape [B, T, C, H, W]
        reconstructed: Reconstructed frames tensor of shape [B, T, C, H, W]
        save_dir: Directory to save images
        prefix: Prefix for saved files
    """
    batch_size, seq_length = original.shape[:2]

    # Select frames to save (first, middle, last)
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
        save_path = os.path.join(save_dir, f"{prefix}_frames.png")
        plt.savefig(save_path)
        plt.close(fig)


def save_metrics(avg_metrics: Dict[str, float],
                per_sequence_metrics: Dict[str, Dict[str, float]],
                save_dir: str,
                prefix: str):
    """
    Save metrics to CSV files.

    Args:
        avg_metrics: Dictionary of average metrics
        per_sequence_metrics: Dictionary of per-sequence metrics
        save_dir: Directory to save metrics
        prefix: Prefix for saved files
    """
    # Save average metrics
    with open(os.path.join(save_dir, f"{prefix}_avg_metrics.csv"), "w") as f:
        f.write("Metric,Value\n")
        for key, value in avg_metrics.items():
            f.write(f"{key},{value:.6f}\n")

    # Save per-sequence metrics
    with open(os.path.join(save_dir, f"{prefix}_per_sequence_metrics.csv"), "w") as f:
        # Write header
        f.write("Sequence")
        for key in avg_metrics:
            f.write(f",{key}")
        f.write("\n")

        # Write values
        for seq_name, metrics in per_sequence_metrics.items():
            f.write(seq_name)
            for key in avg_metrics:
                f.write(f",{metrics[key]:.6f}")
            f.write("\n")


def plot_rd_curve(rd_metrics: Dict[str, List[float]], save_path: str):
    """
    Plot rate-distortion curve.

    Args:
        rd_metrics: Dictionary of metrics for different rate points
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    # Plot PSNR vs BPP
    plt.subplot(1, 2, 1)
    plt.plot(rd_metrics["bpp"], rd_metrics["psnr"], "o-", linewidth=2)
    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel("PSNR (dB)")
    plt.title("Rate-Distortion Curve (PSNR)")
    plt.grid(True)

    # Plot MS-SSIM vs BPP
    plt.subplot(1, 2, 2)
    plt.plot(rd_metrics["bpp"], rd_metrics["msssim"], "o-", linewidth=2)
    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel("MS-SSIM")
    plt.title("Rate-Distortion Curve (MS-SSIM)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate(config: Config, checkpoint_path: str, save_dir: str, mode: str = "standard"):
    """
    Evaluate the layered neural codec.

    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        save_dir: Directory to save results
        mode: Evaluation mode ("standard" or "rd_curve")
    """
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Create dataloaders (use validation loader)
    _, val_loader = create_dataloaders(config)

    # Create model
    model = LayeredNeuralCodec(config).to(device)

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint_data = load_checkpoint(checkpoint_path, model, map_location=device)
    print(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}")

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Evaluate
    if mode == "standard":
        evaluate_model(model, val_loader, device, save_dir)
    elif mode == "rd_curve":
        evaluate_model_rd(model, val_loader, device, save_dir)
    else:
        raise ValueError(f"Unsupported evaluation mode: {mode}")

    print(f"Evaluation results saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate layered neural codec")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "rd_curve"],
                      help="Evaluation mode")
    args = parser.parse_args()

    # Use default config
    config = default_config

    # Override with config file if provided
    if args.config is not None and os.path.exists(args.config):
        # Load config from file (implementation depends on config format)
        pass

    # Evaluate the model
    evaluate(config, args.checkpoint, args.output, args.mode)
