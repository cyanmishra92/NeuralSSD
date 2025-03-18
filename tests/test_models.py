"""
Unit tests for the model components.
"""
import unittest
import torch
import torch.nn as nn
import numpy as np

from config.default_config import Config, default_config
from models.feature_extraction import MobileNetFeatureExtractor, TransformerFeatureExtractor, get_feature_extractor
from models.motion_estimation import MotionEstimation, warp_frame
from models.base_layer import BaseLayerEncoder, BaseLayerDecoder, BaseLayerCodec
from models.enhancement_layer import EnhancementLayer, EnhancementLayerStack
from models.codec import LayeredNeuralCodec


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction modules."""

    def setUp(self):
        """Set up test environment."""
        self.config = default_config
        self.config.feature_extraction.model_type = "mobilenet"
        self.config.feature_extraction.feature_dim = 128
        self.config.feature_extraction.pretrained = False  # Faster for testing

        # Create a dummy input
        self.input = torch.rand(2, 3, 256, 256)

    def test_mobilenet_feature_extractor(self):
        """Test MobileNet feature extractor."""
        model = MobileNetFeatureExtractor(
            feature_dim=self.config.feature_extraction.feature_dim,
            pretrained=self.config.feature_extraction.pretrained
        )

        # Forward pass
        outputs = model(self.input)

        # Check outputs
        self.assertIn("features", outputs)
        self.assertIn("spatial_features", outputs)

        # Check shapes
        features = outputs["features"]
        spatial_features = outputs["spatial_features"]

        self.assertEqual(features.shape, (2, self.config.feature_extraction.feature_dim))
        self.assertEqual(spatial_features.shape[0], 2)  # Batch size
        self.assertEqual(spatial_features.shape[1], self.config.feature_extraction.feature_dim)  # Feature dim

    def test_transformer_feature_extractor(self):
        """Test transformer feature extractor."""
        # Skip if not enough memory
        try:
            model = TransformerFeatureExtractor(
                feature_dim=self.config.feature_extraction.feature_dim,
                pretrained=self.config.feature_extraction.pretrained
            )

            # Forward pass
            outputs = model(self.input)

            # Check outputs
            self.assertIn("features", outputs)
            self.assertIn("spatial_features", outputs)

            # Check shapes
            features = outputs["features"]
            self.assertEqual(features.shape, (2, self.config.feature_extraction.feature_dim))
        except RuntimeError as e:
            print(f"Skipping transformer test due to: {e}")

    def test_get_feature_extractor(self):
        """Test feature extractor factory function."""
        # Test MobileNet
        self.config.feature_extraction.model_type = "mobilenet"
        model = get_feature_extractor(self.config)
        self.assertIsInstance(model, MobileNetFeatureExtractor)

        # Test Transformer
        try:
            self.config.feature_extraction.model_type = "transformer"
            model = get_feature_extractor(self.config)
            self.assertIsInstance(model, TransformerFeatureExtractor)
        except RuntimeError as e:
            print(f"Skipping transformer test due to: {e}")


class TestMotionEstimation(unittest.TestCase):
    """Test motion estimation modules."""

    def setUp(self):
        """Set up test environment."""
        self.config = default_config
        self.config.motion_estimation.method = "neural"
        self.config.motion_estimation.refinement_layers = 2
        self.config.motion_estimation.refinement_channels = 32

        # Create dummy inputs
        self.reference = torch.rand(2, 3, 128, 128)
        self.target = torch.rand(2, 3, 128, 128)
        self.flow = torch.rand(2, 2, 128, 128)

    def test_warp_frame(self):
        """Test frame warping."""
        # Warp the reference frame
        warped = warp_frame(self.reference, self.flow)

        # Check shape
        self.assertEqual(warped.shape, self.reference.shape)

    def test_motion_estimation(self):
        """Test motion estimation module."""
        model = MotionEstimation(self.config)

        # Forward pass
        outputs = model(self.reference, self.target)

        # Check outputs
        self.assertIn("flow", outputs)
        self.assertIn("warped", outputs)
        self.assertIn("residual", outputs)

        # Check shapes
        flow = outputs["flow"]
        warped = outputs["warped"]
        residual = outputs["residual"]

        self.assertEqual(flow.shape, (2, 2, 128, 128))
        self.assertEqual(warped.shape, self.reference.shape)
        self.assertEqual(residual.shape, self.reference.shape)


class TestBaseLayer(unittest.TestCase):
    """Test base layer modules."""

    def setUp(self):
        """Set up test environment."""
        self.config = default_config
        self.config.base_layer.latent_dim = 64
        self.config.base_layer.encoder_channels = [32, 64, 128]
        self.config.base_layer.decoder_channels = [128, 64, 32]
        self.config.base_layer.kernel_size = 3
        self.config.base_layer.activation = "relu"
        self.config.base_layer.use_attention = True

        # Create a dummy input
        self.input = torch.rand(2, 3, 128, 128)

    def test_base_layer_encoder(self):
        """Test base layer encoder."""
        model = BaseLayerEncoder(self.config)

        # Forward pass
        outputs = model(self.input)

        # Check outputs
        self.assertIn("latent", outputs)
        self.assertIn("features", outputs)

        # Check shapes
        latent = outputs["latent"]
        self.assertEqual(latent.shape[0], 2)  # Batch size
        self.assertEqual(latent.shape[1], self.config.base_layer.latent_dim)  # Latent dim

    def test_base_layer_decoder(self):
        """Test base layer decoder."""
        # Create encoder to get latent
        encoder = BaseLayerEncoder(self.config)
        encoder_outputs = encoder(self.input)
        latent = encoder_outputs["latent"]

        # Create decoder
        decoder = BaseLayerDecoder(self.config)

        # Forward pass
        outputs = decoder(latent)

        # Check outputs
        self.assertIn("output", outputs)
        self.assertIn("features", outputs)

        # Check shapes
        output = outputs["output"]
        self.assertEqual(output.shape, self.input.shape)

    def test_base_layer_codec(self):
        """Test base layer codec."""
        model = BaseLayerCodec(self.config)

        # Forward pass
        outputs = model(self.input)

        # Check outputs
        self.assertIn("latent", outputs)
        self.assertIn("output", outputs)

        # Check shapes
        latent = outputs["latent"]
        output = outputs["output"]

        self.assertEqual(latent.shape[0], 2)  # Batch size
        self.assertEqual(latent.shape[1], self.config.base_layer.latent_dim)  # Latent dim
        self.assertEqual(output.shape, self.input.shape)


class TestEnhancementLayer(unittest.TestCase):
    """Test enhancement layer modules."""

    def setUp(self):
        """Set up test environment."""
        self.config = default_config
        self.config.enhancement_layer.num_layers = 2
        self.config.enhancement_layer.latent_dims = [32, 16]
        self.config.enhancement_layer.roi_enabled = True
        self.config.enhancement_layer.roi_weight_factor = 2.0

        # Create dummy inputs
        self.input = torch.rand(2, 3, 128, 128)
        self.prev_reconstruction = torch.rand(2, 3, 128, 128)

    def test_enhancement_layer(self):
        """Test single enhancement layer."""
        model = EnhancementLayer(
            input_channels=6,  # RGB + previous reconstruction
            latent_dim=32,
            use_roi=True,
            roi_weight_factor=2.0
        )

        # Forward pass
        outputs = model(self.input, self.prev_reconstruction)

        # Check outputs
        self.assertIn("latent", outputs)
        self.assertIn("refinement", outputs)
        self.assertIn("output", outputs)

        # Check shapes
        latent = outputs["latent"]
        refinement = outputs["refinement"]
        output = outputs["output"]

        self.assertEqual(refinement.shape, self.input.shape)
        self.assertEqual(output.shape, self.input.shape)

    def test_enhancement_layer_stack(self):
        """Test enhancement layer stack."""
        model = EnhancementLayerStack(self.config)

        # Forward pass
        outputs = model(self.input, self.prev_reconstruction)

        # Check outputs
        self.assertIn("final_output", outputs)
        self.assertIn("layer_outputs", outputs)
        self.assertIn("latents", outputs)

        # Check shapes
        final_output = outputs["final_output"]
        layer_outputs = outputs["layer_outputs"]
        latents = outputs["latents"]

        self.assertEqual(final_output.shape, self.input.shape)
        self.assertEqual(len(layer_outputs), self.config.enhancement_layer.num_layers)
        self.assertEqual(len(latents), self.config.enhancement_layer.num_layers)


class TestLayeredCodec(unittest.TestCase):
    """Test the complete layered codec."""

    def setUp(self):
        """Set up test environment."""
        self.config = default_config

        # Reduce model size for testing
        self.config.feature_extraction.model_type = "mobilenet"
        self.config.feature_extraction.feature_dim = 64
        self.config.feature_extraction.pretrained = False

        self.config.base_layer.latent_dim = 32
        self.config.base_layer.encoder_channels = [16, 32, 64]
        self.config.base_layer.decoder_channels = [64, 32, 16]

        self.config.enhancement_layer.num_layers = 1
        self.config.enhancement_layer.latent_dims = [16]

        self.config.training.anchor_frame_interval = 2

        # Create a dummy input sequence
        self.frames = torch.rand(2, 4, 3, 128, 128)  # [B, T, C, H, W]

    def test_layered_neural_codec(self):
        """Test the complete layered neural codec."""
        model = LayeredNeuralCodec(self.config)

        # Forward pass
        outputs = model(self.frames)

        # Check outputs
        self.assertIn("reconstructed_frames", outputs)
        self.assertIn("base_reconstructions", outputs)
        self.assertIn("base_latent", outputs)
        self.assertIn("flows", outputs)

        # Check shapes
        reconstructed_frames = outputs["reconstructed_frames"]
        base_reconstructions = outputs["base_reconstructions"]
        base_latent = outputs["base_latent"]
        flows = outputs["flows"]

        self.assertEqual(reconstructed_frames.shape, self.frames.shape)
        self.assertEqual(base_reconstructions.shape, self.frames.shape)
        self.assertEqual(base_latent.shape[0:2], self.frames.shape[0:2])  # Batch and sequence dims
        self.assertEqual(flows.shape[0:3], (2, 4, 2))  # [B, T, 2, H, W]


if __name__ == "__main__":
    unittest.main()
