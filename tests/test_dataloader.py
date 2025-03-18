"""
Unit tests for the data loading components.
"""
import os
import unittest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from config.default_config import Config, default_config
from data.datasets import VideoFrameDataset, CityscapesVideoDataset, KITTIVideoDataset, get_dataset, get_transforms
from data.dataloader import create_dataloaders, prepare_batch, get_anchor_frame_mask


class MockDataset(VideoFrameDataset):
    """Mock dataset for testing."""

    def _get_sequence_dirs(self):
        return ["seq1", "seq2", "seq3"]

    def _get_frame_sequences(self):
        # Create simple mock frame sequences with dummy paths
        sequences = []
        for i in range(3):
            sequence_frames = [f"frame_{i}_{j}.jpg" for j in range(self.sequence_length)]
            sequences.append({
                "sequence_name": f"seq{i+1}",
                "frame_paths": sequence_frames,
                "frame_indices": list(range(self.sequence_length)),
            })
        return sequences

    def __getitem__(self, idx):
        # Override to create dummy tensors instead of loading images
        sequence_info = self.frame_sequences[idx]

        # Create dummy frames
        frames = []
        for _ in range(len(sequence_info["frame_paths"])):
            # Create a random colored frame
            dummy_frame = torch.rand(3, *self.frame_size)
            frames.append(dummy_frame)

        # Stack frames along new dimension
        frames = torch.stack(frames, dim=0)

        return {
            "frames": frames,
            "sequence_name": sequence_info["sequence_name"],
            "frame_indices": sequence_info["frame_indices"],
        }


class TestDataLoader(unittest.TestCase):
    """Test data loading components."""

    def setUp(self):
        """Set up test environment."""
        self.config = default_config
        self.config.data.sequence_length = 4
        self.config.data.frame_size = (128, 256)
        self.config.data.batch_size = 2

        # Create mock dataset directory
        os.makedirs("./test_data", exist_ok=True)

    def tearDown(self):
        """Clean up after tests."""
        # Remove mock dataset directory
        import shutil
        if os.path.exists("./test_data"):
            shutil.rmtree("./test_data")

    def test_video_frame_dataset(self):
        """Test the base VideoFrameDataset."""
        dataset = MockDataset(
            root_dir="./test_data",
            sequence_length=self.config.data.sequence_length,
            frame_size=self.config.data.frame_size,
            split="train"
        )

        # Check dataset length
        self.assertEqual(len(dataset), 3)

        # Check item
        item = dataset[0]
        self.assertIn("frames", item)
        self.assertIn("sequence_name", item)
        self.assertIn("frame_indices", item)

        # Check frames shape
        frames = item["frames"]
        self.assertEqual(frames.shape, (self.config.data.sequence_length, 3, *self.config.data.frame_size))

    def test_get_transforms(self):
        """Test the transform generation."""
        # Test training transforms
        train_transform = get_transforms(self.config, is_train=True)
        self.assertIsNotNone(train_transform)

        # Test validation transforms
        val_transform = get_transforms(self.config, is_train=False)
        self.assertIsNotNone(val_transform)

    def test_prepare_batch(self):
        """Test batch preparation."""
        # Create a dummy batch
        batch = {
            "frames": torch.rand(2, 4, 3, 128, 256),
            "sequence_name": ["seq1", "seq2"],
            "frame_indices": [torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3])]
        }

        # Prepare batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prepared_batch = prepare_batch(batch, device)

        # Check that tensors are moved to the device
        self.assertEqual(prepared_batch["frames"].device, device)

        # Check that non-tensors are preserved
        self.assertEqual(prepared_batch["sequence_name"], batch["sequence_name"])

    def test_get_anchor_frame_mask(self):
        """Test anchor frame mask generation."""
        # Create anchor frame mask
        batch_size = 2
        seq_length = 8
        anchor_interval = 2

        mask = get_anchor_frame_mask(batch_size, seq_length, anchor_interval)

        # Check mask shape
        self.assertEqual(mask.shape, (batch_size, seq_length))

        # Check that frames at positions 0, 2, 4, 6 are marked as anchor frames
        for i in range(0, seq_length, anchor_interval):
            self.assertTrue(mask[:, i].all())

        # Check that other frames are not marked as anchor frames
        for i in range(1, seq_length, 2):
            self.assertFalse(mask[:, i].any())


if __name__ == "__main__":
    unittest.main()
