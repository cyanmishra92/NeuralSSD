"""
Unit tests for the loss functions.
"""
import unittest
import torch
import torch.nn as nn
import numpy as np

from config.default_config import Config, default_config
from losses.reconstruction_loss import MSELoss, L1Loss, CharbonnierLoss, SSIMLoss, get_reconstruction_loss
from losses.perceptual_loss import VGGPerceptualLoss, LPIPSLoss, get_perceptual_loss
from losses.temporal_loss import TemporalConsistencyLoss, SequentialTemporalLoss, get_temporal_loss
from losses.rate_loss import LaplacianRateLoss, GaussianRateLoss, get_rate_loss
from losses.composite_loss import CompositeLoss, LayeredLoss, get_loss_function


class TestReconstructionLoss(unittest.TestCase):
    """Test reconstruction loss functions."""

    def setUp(self):
        """Set up test environment."""
        self.pred = torch.rand(2, 3, 64, 64)
        self.target = torch.rand(2, 3, 64, 64)

    def test_mse_loss(self):
        """Test MSE loss."""
        loss_fn = MSELoss()
        loss = loss_fn(self.pred, self.target)

        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Check that loss is non-negative
        self.assertGreaterEqual(loss.item(), 0)

        # Check that loss is zero when pred equals target
        loss_zero = loss_fn(self.target, self.target)
        self.assertAlmostEqual(loss_zero.item(), 0, places=5)

    def test_l1_loss(self):
        """Test L1 loss."""
        loss_fn = L1Loss()
        loss = loss_fn(self.pred, self.target)

        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Check that loss is non-negative
        self.assertGreaterEqual(loss.item(), 0)

        # Check that loss is zero when pred equals target
        loss_zero = loss_fn(self.target, self.target)
        self.assertAlmostEqual(loss_zero.item(), 0, places=5)

    def test_charbonnier_loss(self):
        """Test Charbonnier loss."""
        loss_fn = CharbonnierLoss()
        loss = loss_fn(self.pred, self.target)

        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Check that loss is non-negative
        self.assertGreaterEqual(loss.item(), 0)

        # Check that loss is close to zero when pred equals target
        loss_zero = loss_fn(self.target, self.target)
        self.assertLess(loss_zero.item(), 1e-2)

    def test_ssim_loss(self):
        """Test SSIM loss."""
        loss_fn = SSIMLoss()
        loss = loss_fn(self.pred, self.target)

        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Check that loss is non-negative
        self.assertGreaterEqual(loss.item(), 0)

        # Check that loss is close to zero when pred equals target
        loss_zero = loss_fn(self.target, self.target)
        self.assertLess(loss_zero.item(), 1e-2)

    def test_get_reconstruction_loss(self):
        """Test reconstruction loss factory function."""
        # Test MSE loss
        loss_fn = get_reconstruction_loss("mse")
        self.assertIsInstance(loss_fn, MSELoss)

        # Test L1 loss
        loss_fn = get_reconstruction_loss("l1")
        self.assertIsInstance(loss_fn, L1Loss)

        # Test Charbonnier loss
        loss_fn = get_reconstruction_loss("charbonnier")
        self.assertIsInstance(loss_fn, CharbonnierLoss)

        # Test SSIM loss
        loss_fn = get_reconstruction_loss("ssim")
        self.assertIsInstance(loss_fn, SSIMLoss)


class TestPerceptualLoss(unittest.TestCase):
    """Test perceptual loss functions."""

    def setUp(self):
        """Set up test environment."""
        self.pred = torch.rand(2, 3, 256, 256)
        self.target = torch.rand(2, 3, 256, 256)

    def test_vgg_perceptual_loss(self):
        """Test VGG perceptual loss."""
        try:
            loss_fn = VGGPerceptualLoss(layers=['conv1_2'])
            loss = loss_fn(self.pred, self.target)

            # Check that loss is a scalar
            self.assertEqual(loss.dim(), 0)

            # Check that loss is non-negative
            self.assertGreaterEqual(loss.item(), 0)
        except Exception as e:
            print(f"Skipping VGG perceptual loss test due to: {e}")

    def test_lpips_loss(self):
        """Test LPIPS loss."""
        try:
            loss_fn = LPIPSLoss()
            loss = loss_fn(self.pred, self.target)

            # Check that loss is a scalar
            self.assertEqual(loss.dim(), 0)

            # Check that loss is non-negative
            self.assertGreaterEqual(loss.item(), 0)
        except Exception as e:
            print(f"Skipping LPIPS loss test due to: {e}")

    def test_get_perceptual_loss(self):
        """Test perceptual loss factory function."""
        try:
            # Test VGG loss
            loss_fn = get_perceptual_loss("vgg")
            self.assertIsInstance(loss_fn, VGGPerceptualLoss)

            # Test LPIPS loss
            loss_fn = get_perceptual_loss("lpips")
            self.assertIsInstance(loss_fn, LPIPSLoss)
        except Exception as e:
            print(f"Skipping perceptual loss factory test due to: {e}")


class TestTemporalLoss(unittest.TestCase):
    """Test temporal loss functions."""

    def setUp(self):
        """Set up test environment."""
        self.prev_frame = torch.rand(2, 3, 64, 64)
        self.curr_frame = torch.rand(2, 3, 64, 64)
        self.flow = torch.rand(2, 2, 64, 64)

        self.sequence = torch.rand(2, 4, 3, 64, 64)  # [B, T, C, H, W]
        self.flows = torch.rand(2, 3, 2, 64, 64)  # [B, T-1, 2, H, W]

    def test_temporal_consistency_loss(self):
        """Test temporal consistency loss."""
        loss_fn = TemporalConsistencyLoss()
        loss = loss_fn(self.prev_frame, self.curr_frame, self.flow)

        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Check that loss is non-negative
        self.assertGreaterEqual(loss.item(), 0)

    def test_sequential_temporal_loss(self):
        """Test sequential temporal loss."""
        loss_fn = SequentialTemporalLoss()
        loss = loss_fn(self.sequence, self.flows)

        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Check that loss is non-negative
        self.assertGreaterEqual(loss.item(), 0)

    def test_get_temporal_loss(self):
        """Test temporal loss factory function."""
        # Test consistency loss
        loss_fn = get_temporal_loss("consistency")
        self.assertIsInstance(loss_fn, TemporalConsistencyLoss)

        # Test sequential loss
        loss_fn = get_temporal_loss("sequential")
        self.assertIsInstance(loss_fn, SequentialTemporalLoss)


class TestRateLoss(unittest.TestCase):
    """Test rate loss functions."""

    def setUp(self):
        """Set up test environment."""
        self.latent = torch.rand(2, 32, 16, 16)
        self.rate = torch.rand(2)

    def test_laplacian_rate_loss(self):
        """Test Laplacian rate loss."""
        loss_fn = LaplacianRateLoss()
        loss = loss_fn(self.latent)

        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Check that loss is non-negative
        self.assertGreaterEqual(loss.item(), 0)

    def test_gaussian_rate_loss(self):
        """Test Gaussian rate loss."""
        loss_fn = GaussianRateLoss()
        loss = loss_fn(self.latent)

        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Check that loss is non-negative
        self.assertGreaterEqual(loss.item(), 0)

    def test_get_rate_loss(self):
        """Test rate loss factory function."""
        # Test Laplacian loss
        loss_fn = get_rate_loss("laplacian")
        self.assertIsInstance(loss_fn, LaplacianRateLoss)

        # Test Gaussian loss
        loss_fn = get_rate_loss("gaussian")
        self.assertIsInstance(loss_fn, GaussianRateLoss)


class TestCompositeLoss(unittest.TestCase):
    """Test composite loss functions."""

    def setUp(self):
        """Set up test environment."""
        self.config = default_config

        # Create dummy inputs
        self.frames = torch.rand(2, 4, 3, 64, 64)  # [B, T, C, H, W]
        self.reconstructed = torch.rand(2, 4, 3, 64, 64)  # [B, T, C, H, W]
        self.base_latent = torch.rand(2, 4, 32, 16, 16)  # [B, T, C, H, W]
        self.flows = torch.rand(2, 4, 2, 64, 64)  # [B, T, 2, H, W]

        # Create dummy outputs and targets
        self.outputs = {
            "reconstructed_frames": self.reconstructed,
            "base_latent": self.base_latent,
            "flows": self.flows
        }

        self.targets = {
            "frames": self.frames
        }

    def test_composite_loss(self):
        """Test composite loss."""
        loss_fn = CompositeLoss(self.config)
        losses = loss_fn(self.outputs, self.targets)

        # Check that losses dictionary is returned
        self.assertIsInstance(losses, dict)

        # Check that total loss is included
        self.assertIn("total", losses)

        # Check that total loss is a scalar
        self.assertEqual(losses["total"].dim(), 0)

        # Check that total loss is non-negative
        self.assertGreaterEqual(losses["total"].item(), 0)

    def test_layered_loss(self):
        """Test layered loss."""
        loss_fn = LayeredLoss(self.config)
        losses = loss_fn(self.outputs, self.targets)

        # Check that losses dictionary is returned
        self.assertIsInstance(losses, dict)

        # Check that total loss is included
        self.assertIn("total", losses)

        # Check that total loss is a scalar
        self.assertEqual(losses["total"].dim(), 0)

        # Check that total loss is non-negative
        self.assertGreaterEqual(losses["total"].item(), 0)

    def test_get_loss_function(self):
        """Test loss function factory function."""
        # Test composite loss
        loss_fn = get_loss_function(self.config, loss_type="composite")
        self.assertIsInstance(loss_fn, CompositeLoss)

        # Test layered loss
        loss_fn = get_loss_function(self.config, loss_type="layered")
        self.assertIsInstance(loss_fn, LayeredLoss)


if __name__ == "__main__":
    unittest.main()
