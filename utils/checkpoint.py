"""
Model checkpointing utilities for the layered neural codec.
"""
import os
import torch
import logging
from typing import Dict, Any, Optional, Union


logger = logging.getLogger("layered_codec")


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[Any],
                   epoch: int,
                   step: int,
                   metrics: Dict[str, float],
                   save_dir: str,
                   is_best: bool = False,
                   filename: Optional[str] = None):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        epoch: Current epoch
        step: Current step
        metrics: Current metrics
        save_dir: Directory to save checkpoint to
        is_best: Whether this is the best model so far
        filename: Custom filename (if None, use default)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'metrics': metrics,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()

    # Save regular checkpoint
    if filename is None:
        filename = f"checkpoint_epoch{epoch}.pth"

    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save best checkpoint if required
    if is_best:
        best_path = os.path.join(save_dir, "checkpoint_best.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint to {best_path}")

    # Remove old checkpoints to save space (keep only latest 3 and best)
    clean_old_checkpoints(save_dir, keep_last=3)


def load_checkpoint(checkpoint_path: str,
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None,
                   map_location: Optional[str] = None) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into
        map_location: Device to map model to

    Returns:
        Dictionary containing checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Load model weights
    model.load_state_dict(checkpoint['model'])
    logger.info(f"Loaded model weights from {checkpoint_path}")

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Loaded optimizer state")

    # Load scheduler state if provided
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded scheduler state")

    # Return metadata
    metadata = {k: v for k, v in checkpoint.items()
               if k not in ['model', 'optimizer', 'scheduler']}

    return metadata


def clean_old_checkpoints(save_dir: str, keep_last: int = 3):
    """
    Remove old checkpoints, keeping only the latest ones and the best.

    Args:
        save_dir: Directory containing checkpoints
        keep_last: Number of latest checkpoints to keep
    """
    checkpoints = [f for f in os.listdir(save_dir)
                 if f.startswith("checkpoint_epoch") and f.endswith(".pth")]

    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.split("epoch")[1].split(".")[0]), reverse=True)

    # Keep best checkpoint
    best_checkpoint = "checkpoint_best.pth"

    # Remove excess checkpoints
    for checkpoint in checkpoints[keep_last:]:
        if checkpoint != best_checkpoint:
            os.remove(os.path.join(save_dir, checkpoint))


def get_latest_checkpoint(save_dir: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint in a directory.

    Args:
        save_dir: Directory containing checkpoints

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(save_dir):
        return None

    checkpoints = [f for f in os.listdir(save_dir)
                 if f.startswith("checkpoint_epoch") and f.endswith(".pth")]

    if not checkpoints:
        return None

    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.split("epoch")[1].split(".")[0]), reverse=True)

    return os.path.join(save_dir, checkpoints[0])
