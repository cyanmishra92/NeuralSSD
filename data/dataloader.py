"""
Data loading utilities for the layered neural codec.
"""
import torch
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, Any, Optional

from .datasets import get_dataset, get_transforms


def create_dataloaders(config):
    """
    Create train and validation dataloaders.

    Args:
        config: Configuration object

    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Get transforms
    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)

    # Create datasets
    train_dataset = get_dataset(config, split="train")
    train_dataset.transform = train_transform

    val_dataset = get_dataset(config, split="val")
    val_dataset.transform = val_transform

    # Create samplers for distributed training if needed
    train_sampler = None
    val_sampler = None

    if config.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader


def prepare_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move batch to device and perform any necessary preprocessing.

    Args:
        batch: Dictionary containing batch data
        device: Device to move data to

    Returns:
        Processed batch on the specified device
    """
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            result[k] = [t.to(device) for t in v]
        else:
            result[k] = v

    return result


def get_anchor_frame_mask(batch_size: int, seq_length: int,
                          anchor_interval: int = 5,
                          device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create a boolean mask indicating which frames are anchor frames.

    Args:
        batch_size: Batch size
        seq_length: Sequence length
        anchor_interval: Interval between anchor frames
        device: Device to create tensor on

    Returns:
        Boolean mask of shape [batch_size, seq_length] where True indicates an anchor frame
    """
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)

    # Mark frames at positions divisible by anchor_interval as anchor frames
    # First frame is always an anchor frame
    for i in range(0, seq_length, anchor_interval):
        mask[:, i] = True

    return mask
