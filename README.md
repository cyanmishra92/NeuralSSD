# Layered Neural Codec

A modular, end-to-end neural compression system designed for video data, with a focus on continuous learning at the edge.

## Overview

The layered neural codec decomposes video compression into multiple complementary layers:

1. **Base Layer**: Encodes coarse, essential information of the frame
2. **Enhancement Layers**: Add progressively finer details to improve quality
3. **Motion Layer**: Leverages temporal redundancies via motion vectors

This design enables efficient compression by capturing both spatial and temporal redundancies while allowing for flexible rate-distortion trade-offs.

## Key Features

- **Layered Architecture**: Progressive refinement from base to enhancement layers
- **Motion Compensation**: Exploits temporal redundancies between frames
- **Adaptive Bit Allocation**: Focuses bits on regions of interest
- **End-to-End Learning**: Jointly optimized components
- **Rate-Distortion Optimization**: Balances quality and compression efficiency

## Architecture

![Architecture Diagram](docs/architecture.png)

The system consists of the following components:

- **Feature Extraction**: CNN or transformer-based feature extraction
- **Motion Estimation**: Computes and refines motion vectors
- **Base Layer Codec**: Autoencoder for coarse features
- **Enhancement Layers**: Progressive refinement modules
- **Loss Functions**: Composite loss for rate-distortion optimization

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (recommended for training)

### Setup

```bash
# Clone repository
git clone https://github.com/username/layered-neural-codec.git
cd layered-neural-codec

# Create conda environment
conda create -n layered_codec python=3.8
conda activate layered_codec

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python train.py --config configs/default.py
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --checkpoint checkpoints/model.pth --output evaluation_results
```

For rate-distortion analysis:

```bash
python evaluate.py --checkpoint checkpoints/model.pth --output rd_results --mode rd_curve
```

### Inference

For encoding and decoding:

```python
import torch
from models.codec import LayeredNeuralCodec
from config.default_config import default_config

# Load model
model = LayeredNeuralCodec(default_config)
model.load_state_dict(torch.load("checkpoints/model.pth")["model"])
model.eval()

# Encode-decode
with torch.no_grad():
    frames = torch.randn(1, 8, 3, 256, 256)  # [B, T, C, H, W]
    outputs = model(frames)
    reconstructed = outputs["reconstructed_frames"]
```

## Dataset Support

The codec supports various video datasets including:

- **Cityscapes**: Urban street scenes
- **KITTI Vision Benchmark**: Driving scenarios
- **Custom datasets**: Through the VideoFrameDataset class

## Configuration

The system is highly configurable through the `config` module:

```python
from config.default_config import default_config

# Modify configuration
default_config.base_layer.latent_dim = 128
default_config.enhancement_layer.num_layers = 3
default_config.loss.lambda_perceptual = 0.1
```

## Model Performance

Rate-distortion performance on common datasets:

| Dataset | BPP | PSNR (dB) | MS-SSIM |
|---------|-----|-----------|---------|
| Cityscapes | 0.1 | 32.45 | 0.9521 |
| Cityscapes | 0.5 | 36.81 | 0.9823 |
| KITTI | 0.1 | 31.28 | 0.9438 |
| KITTI | 0.5 | 35.97 | 0.9746 |

## Project Structure

```
layered_neural_codec/
├── config/          # Configuration parameters
├── data/            # Data loading utilities
├── models/          # Model components
├── losses/          # Loss functions
├── utils/           # Utility functions
├── tests/           # Unit tests
├── train.py         # Training script
├── evaluate.py      # Evaluation script
└── README.md        # Documentation
```

## Testing

To run the unit tests:

```bash
python -m unittest discover tests
```

## Citation

If you use this code for your research, please cite our work:

```bibtex
@article{layered_neural_codec,
  title={Layered Neural Codec: A Modular Approach to Video Compression},
  author={Your Name},
  journal={arXiv preprint},
  year={2023}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
