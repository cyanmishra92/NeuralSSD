"""
Configuration settings for the layered neural codec.
"""
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "cityscapes"
    dataset_path: str = "./data/cityscapes"
    frame_size: Tuple[int, int] = (256, 512)  # Height, Width
    sequence_length: int = 16  # Number of frames in a sequence
    batch_size: int = 8
    num_workers: int = 4
    train_val_split: float = 0.8  # Percentage of data for training
    augmentations: bool = True


@dataclass
class FeatureExtractionConfig:
    """Feature extraction configuration."""
    model_type: str = "mobilenet"  # Options: "mobilenet", "transformer"
    pretrained: bool = True
    freeze_backbone: bool = False
    feature_dim: int = 512


@dataclass
class MotionEstimationConfig:
    """Motion estimation configuration."""
    method: str = "raft"  # Options: "raft", "pwc", "neural"
    flow_checkpoint: Optional[str] = None
    refinement_layers: int = 3
    refinement_channels: int = 64
    warp_padding_mode: str = "border"


@dataclass
class BaseLayerConfig:
    """Base layer encoder/decoder configuration."""
    latent_dim: int = 128
    encoder_channels: List[int] = (64, 128, 256)
    decoder_channels: List[int] = (256, 128, 64)
    kernel_size: int = 3
    activation: str = "relu"  # Options: "relu", "leakyrelu", "gelu"
    use_attention: bool = True


@dataclass
class EnhancementLayerConfig:
    """Enhancement layer configuration."""
    num_layers: int = 2
    latent_dims: List[int] = (64, 32)
    roi_enabled: bool = True
    roi_weight_factor: float = 2.0  # How much to emphasize ROI regions


@dataclass
class LossConfig:
    """Loss configuration."""
    lambda_perceptual: float = 0.1
    lambda_temporal: float = 0.05
    lambda_rate: float = 0.001
    perceptual_type: str = "vgg"  # Options: "vgg", "lpips"
    perceptual_layers: List[str] = ("conv1_2", "conv2_2", "conv3_3")


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_type: str = "cosine"  # Options: "cosine", "step", "plateau"
    scheduler_params: dict = None
    clip_grad_norm: Optional[float] = 1.0
    anchor_frame_interval: int = 5  # Every Nth frame is an anchor frame
    log_interval: int = 10  # Log every N batches
    save_interval: int = 5  # Save model every N epochs
    val_interval: int = 1   # Validate every N epochs

    def __post_init__(self):
        if self.scheduler_params is None:
            if self.scheduler_type == "cosine":
                self.scheduler_params = {"T_max": self.num_epochs}
            elif self.scheduler_type == "step":
                self.scheduler_params = {"step_size": 30, "gamma": 0.1}
            elif self.scheduler_type == "plateau":
                self.scheduler_params = {"patience": 5, "factor": 0.5}


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "./logs"
    experiment_name: str = "layered_codec"
    save_dir: str = "./checkpoints"
    metrics: List[str] = ("psnr", "msssim", "bpp")  # Metrics to track
    plot_interval: int = 100  # Plot reconstructions every N batches
    plot_save_path: str = "./plots"


@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = DataConfig()
    feature_extraction: FeatureExtractionConfig = FeatureExtractionConfig()
    motion_estimation: MotionEstimationConfig = MotionEstimationConfig()
    base_layer: BaseLayerConfig = BaseLayerConfig()
    enhancement_layer: EnhancementLayerConfig = EnhancementLayerConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    logging: LoggingConfig = LoggingConfig()
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"
    mixed_precision: bool = True
    distributed: bool = False


# Default configuration
default_config = Config()
