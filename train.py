"""
Training script for the layered neural codec.
"""
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Tuple, Optional, Union

from config.default_config import Config, default_config
from data.dataloader import create_dataloaders, prepare_batch, get_anchor_frame_mask
from models.codec import LayeredNeuralCodec
from losses.composite_loss import get_loss_function
from utils.logger import Logger
from utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from utils.metrics import calculate_metrics


def train_epoch(model: nn.Module,
               train_loader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.Module,
               device: torch.device,
               logger: Logger,
               epoch: int,
               config: Config,
               scaler: Optional[GradScaler] = None) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        logger: Logger
        epoch: Current epoch
        config: Configuration
        scaler: Gradient scaler for mixed precision training

    Returns:
        Dictionary of average metrics
    """
    model.train()

    # Initialize metrics
    batch_metrics = {
        "loss": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "bpp": 0.0
    }

    # Start epoch
    logger.start_epoch(epoch)
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        # Track data loading time
        data_time = time.time() - start_time

        # Move batch to device
        batch = prepare_batch(batch, device)
        frames = batch["frames"]

        # Record start time for forward pass
        forward_start = time.time()

        # Forward pass with mixed precision if enabled
        if config.mixed_precision and scaler is not None:
            with autocast():
                outputs = model(frames)
                losses = loss_fn(outputs, batch)
                total_loss = losses["total"]
        else:
            outputs = model(frames)
            losses = loss_fn(outputs, batch)
            total_loss = losses["total"]

        # Record forward pass time
        forward_time = time.time() - forward_start

        # Backward pass and optimization
        optimizer.zero_grad()

        # Record start time for backward pass
        backward_start = time.time()

        if config.mixed_precision and scaler is not None:
            scaler.scale(total_loss).backward()

            # Clip gradients
            if config.training.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()

            # Clip gradients
            if config.training.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.clip_grad_norm)

            optimizer.step()

        # Record backward pass time
        backward_time = time.time() - backward_start

        # Calculate metrics
        with torch.no_grad():
            metrics = calculate_metrics(
                outputs["reconstructed_frames"], frames,
                [outputs["base_latent"]] + outputs.get("enhancement_latents", [])
            )

        # Update batch metrics
        batch_metrics["loss"] += total_loss.item()
        batch_metrics["psnr"] += metrics["psnr"].mean().item()
        batch_metrics["ssim"] += metrics["ssim"].mean().item()
        batch_metrics["bpp"] += metrics["bpp"].mean().item()

        # Log metrics
        if batch_idx % config.training.log_interval == 0:
            # Prepare metrics for logging
            log_metrics = {
                "loss": total_loss.item(),
                "loss_rec": losses.get("rec", torch.tensor(0.0)).item(),
                "loss_perc": losses.get("perc", torch.tensor(0.0)).item(),
                "loss_temp": losses.get("temp", torch.tensor(0.0)).item(),
                "loss_rate": losses.get("rate", torch.tensor(0.0)).item(),
                "psnr": metrics["psnr"].mean().item(),
                "msssim": metrics["msssim"].mean().item(),
                "bpp": metrics["bpp"].mean().item(),
                "time_data_loading": data_time,
                "time_forward": forward_time,
                "time_backward": backward_time,
                "time_per_iter": time.time() - start_time
            }

            # Add learning rate
            log_metrics["lr"] = optimizer.param_groups[0]["lr"]

            # Log metrics
            logger.log_metrics(log_metrics, step=epoch * len(train_loader) + batch_idx)

            # Print progress
            print(f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} "
                 f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                 f"Loss: {total_loss.item():.6f}\t"
                 f"PSNR: {metrics['psnr'].mean().item():.2f} dB\t"
                 f"BPP: {metrics['bpp'].mean().item():.4f}")

        # Save visualizations periodically
        if batch_idx % config.logging.plot_interval == 0:
            logger.save_reconstructions(
                frames, outputs["reconstructed_frames"],
                step=epoch * len(train_loader) + batch_idx
            )

            # Log images
            logger.log_images({
                "original": frames[:, 0],
                "reconstructed": outputs["reconstructed_frames"][:, 0],
                "base_reconstruction": outputs["base_reconstructions"][:, 0]
            }, phase="train")

        # Reset timer for next iteration
        start_time = time.time()

    # Calculate average metrics
    for key in batch_metrics:
        batch_metrics[key] /= len(train_loader)

    return batch_metrics


def validate(model: nn.Module,
            val_loader: torch.utils.data.DataLoader,
            loss_fn: nn.Module,
            device: torch.device,
            logger: Logger,
            epoch: int,
            config: Config) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to validate on
        logger: Logger
        epoch: Current epoch
        config: Configuration

    Returns:
        Dictionary of average metrics
    """
    model.eval()

    # Initialize metrics
    val_metrics = {
        "loss": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "msssim": 0.0,
        "bpp": 0.0
    }

    # Disable gradient computation
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move batch to device
            batch = prepare_batch(batch, device)
            frames = batch["frames"]

            # Forward pass
            outputs = model(frames)
            losses = loss_fn(outputs, batch)
            total_loss = losses["total"]

            # Calculate metrics
            metrics = calculate_metrics(
                outputs["reconstructed_frames"], frames,
                [outputs["base_latent"]] + outputs.get("enhancement_latents", [])
            )

            # Update metrics
            val_metrics["loss"] += total_loss.item()
            val_metrics["psnr"] += metrics["psnr"].mean().item()
            val_metrics["ssim"] += metrics["ssim"].mean().item()
            val_metrics["msssim"] += metrics["msssim"].mean().item()
            val_metrics["bpp"] += metrics["bpp"].mean().item()

            # Save visualizations periodically
            if batch_idx % max(1, len(val_loader) // 5) == 0:
                logger.save_reconstructions(
                    frames, outputs["reconstructed_frames"],
                    step=epoch * len(val_loader) + batch_idx,
                    max_samples=2
                )

                # Log images
                logger.log_images({
                    "val_original": frames[:, 0],
                    "val_reconstructed": outputs["reconstructed_frames"][:, 0],
                    "val_base_reconstruction": outputs["base_reconstructions"][:, 0]
                }, phase="val")

    # Calculate average metrics
    for key in val_metrics:
        val_metrics[key] /= len(val_loader)

    # Log validation metrics
    logger.log_metrics(val_metrics, phase="val")

    # Print validation results
    print(f"Validation Epoch: {epoch}\t"
         f"Loss: {val_metrics['loss']:.6f}\t"
         f"PSNR: {val_metrics['psnr']:.2f} dB\t"
         f"MS-SSIM: {val_metrics['msssim']:.4f}\t"
         f"BPP: {val_metrics['bpp']:.4f}")

    return val_metrics


def train(config: Config):
    """
    Train the layered neural codec.

    Args:
        config: Configuration object
    """
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    # Create model
    model = LayeredNeuralCodec(config).to(device)

    # Create loss function
    loss_fn = get_loss_function(config, loss_type="layered")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate,
                         weight_decay=config.training.weight_decay)

    # Create learning rate scheduler
    if config.training.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.num_epochs,
            eta_min=config.training.learning_rate * 0.1)
    elif config.training.scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.training.scheduler_params["step_size"],
            gamma=config.training.scheduler_params["gamma"])
    elif config.training.scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.training.scheduler_params["factor"],
            patience=config.training.scheduler_params["patience"])
    else:
        scheduler = None

    # Create gradient scaler for mixed precision training
    scaler = GradScaler() if config.mixed_precision else None

    # Create logger
    logger = Logger(config)

    # Load checkpoint if available
    start_epoch = 0
    best_val_psnr = 0.0
    checkpoint_path = get_latest_checkpoint(config.logging.save_dir)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_data = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device)

        start_epoch = checkpoint_data["epoch"] + 1
        if "metrics" in checkpoint_data and "psnr" in checkpoint_data["metrics"]:
            best_val_psnr = checkpoint_data["metrics"]["psnr"]

    # Train loop
    for epoch in range(start_epoch, config.training.num_epochs):
        # Train for one epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, logger, epoch, config, scaler)

        # Validate
        if epoch % config.training.val_interval == 0:
            val_metrics = validate(
                model, val_loader, loss_fn, device, logger, epoch, config)

            # Check if model improved
            is_best = logger.update_best_metrics(val_metrics)

            # Save checkpoint
            if epoch % config.training.save_interval == 0 or is_best:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, epoch * len(train_loader),
                    val_metrics, config.logging.save_dir, is_best,
                    f"checkpoint_epoch{epoch}.pth"
                )

        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        # Log epoch results
        logger.end_epoch(optimizer.param_groups[0]["lr"])

    # Close logger
    logger.close()

    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train layered neural codec")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()

    # Use default config
    config = default_config

    # Override with config file if provided
    if args.config is not None and os.path.exists(args.config):
        # Load config from file (implementation depends on config format)
        pass

    # Create directories
    os.makedirs(config.logging.log_dir, exist_ok=True)
    os.makedirs(config.logging.save_dir, exist_ok=True)
    os.makedirs(config.logging.plot_save_path, exist_ok=True)

    # Train the model
    train(config)
