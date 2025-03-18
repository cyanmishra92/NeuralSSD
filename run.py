#!/usr/bin/env python
"""
Convenience script for running the layered neural codec.
"""
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional, Union


def train_model(args):
    """Run training with the specified arguments."""
    from train import train
    from config.default_config import default_config

    # Modify config based on arguments
    if args.learning_rate:
        default_config.training.learning_rate = args.learning_rate

    if args.epochs:
        default_config.training.num_epochs = args.epochs

    if args.batch_size:
        default_config.data.batch_size = args.batch_size

    if args.dataset:
        default_config.data.dataset_name = args.dataset

    if args.dataset_path:
        default_config.data.dataset_path = args.dataset_path

    if args.sequence_length:
        default_config.data.sequence_length = args.sequence_length

    if args.frame_size:
        assert len(args.frame_size) == 2, "Frame size must be [height, width]"
        default_config.data.frame_size = tuple(args.frame_size)

    # Create output directories
    os.makedirs(default_config.logging.log_dir, exist_ok=True)
    os.makedirs(default_config.logging.save_dir, exist_ok=True)
    os.makedirs(default_config.logging.plot_save_path, exist_ok=True)

    # Run training
    train(default_config)


def evaluate_model(args):
    """Run evaluation with the specified arguments."""
    from evaluate import evaluate
    from config.default_config import default_config

    # Modify config based on arguments
    if args.dataset:
        default_config.data.dataset_name = args.dataset

    if args.dataset_path:
        default_config.data.dataset_path = args.dataset_path

    if args.batch_size:
        default_config.data.batch_size = args.batch_size

    if args.sequence_length:
        default_config.data.sequence_length = args.sequence_length

    if args.frame_size:
        assert len(args.frame_size) == 2, "Frame size must be [height, width]"
        default_config.data.frame_size = tuple(args.frame_size)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run evaluation
    evaluate(default_config, args.checkpoint, args.output, args.mode)


def visualize_results(args):
    """Visualize results with the specified arguments."""
    import torch
    from models.codec import LayeredNeuralCodec
    from config.default_config import default_config
    from utils.visualization import (
        visualize_frame_comparison,
        visualize_sequence,
        visualize_flow,
        visualize_latent,
        visualize_enhancement_layers,
        create_comparison_grid,
        save_gif
    )
    from utils.checkpoint import load_checkpoint

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = LayeredNeuralCodec(default_config).to(device)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    # Load sample input
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file not found at {args.input}")
            return

        try:
            sample_data = torch.load(args.input)
            frames = sample_data["frames"].to(device)
        except:
            print(f"Error: Failed to load frames from {args.input}")
            return
    else:
        # Create dummy input
        frames = torch.rand(1, 8, 3, 256, 256).to(device)

    # Process frames
    with torch.no_grad():
        outputs = model(frames)

    # Create visualizations
    reconstructed = outputs["reconstructed_frames"]
    base_reconstructions = outputs["base_reconstructions"]
    flows = outputs["flows"]

    # Save individual frame comparisons
    for t in range(min(frames.shape[1], 5)):  # First 5 frames
        visualize_frame_comparison(
            frames[0, t], reconstructed[0, t], base_reconstructions[0, t],
            title=f"Frame {t} Comparison",
            save_path=os.path.join(args.output, f"frame_{t}_comparison.png")
        )

    # Save sequence visualization
    visualize_sequence(
        frames[0], title="Original Sequence",
        save_path=os.path.join(args.output, "original_sequence.png")
    )

    visualize_sequence(
        reconstructed[0], title="Reconstructed Sequence",
        save_path=os.path.join(args.output, "reconstructed_sequence.png")
    )

    # Save flow visualization
    for t in range(min(flows.shape[1], 4)):  # First 4 flows
        visualize_flow(
            flows[0, t], frames[0, t],
            title=f"Flow for Frame {t}",
            save_path=os.path.join(args.output, f"flow_{t}.png")
        )

    # Save latent visualization
    visualize_latent(
        outputs["base_latent"][0, 0], title="Base Layer Latent",
        save_path=os.path.join(args.output, "base_latent.png")
    )

    # Save comparison grid
    create_comparison_grid(
        frames[0], reconstructed[0],
        title="Original vs Reconstructed Detail Comparison",
        save_path=os.path.join(args.output, "detail_comparison.png")
    )

    # Save GIFs
    save_gif(
        frames[0], os.path.join(args.output, "original.gif"), fps=5
    )

    save_gif(
        reconstructed[0], os.path.join(args.output, "reconstructed.gif"), fps=5
    )

    print(f"Visualizations saved to {args.output}")


def run_tests(args):
    """Run unit tests."""
    import unittest

    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layered Neural Codec")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--dataset", type=str, help="Dataset name")
    train_parser.add_argument("--dataset-path", type=str, help="Path to dataset")
    train_parser.add_argument("--sequence-length", type=int, help="Number of frames in a sequence")
    train_parser.add_argument("--frame-size", type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'), help="Frame size as (height width)")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--output", type=str, default="./evaluation_results", help="Output directory")
    eval_parser.add_argument("--mode", type=str, default="standard", choices=["standard", "rd_curve"],
                           help="Evaluation mode")
    eval_parser.add_argument("--dataset", type=str, help="Dataset name")
    eval_parser.add_argument("--dataset-path", type=str, help="Path to dataset")
    eval_parser.add_argument("--batch-size", type=int, help="Batch size")
    eval_parser.add_argument("--sequence-length", type=int, help="Number of frames in a sequence")
    eval_parser.add_argument("--frame-size", type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'), help="Frame size as (height width)")

    # Visualize command
    vis_parser = subparsers.add_parser("visualize", help="Visualize model outputs")
    vis_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    vis_parser.add_argument("--input", type=str, help="Path to input data file")
    vis_parser.add_argument("--output", type=str, default="./visualizations", help="Output directory")
    vis_parser.add_argument("--device", type=str, default="cuda", help="Device to run on")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run unit tests")

    # Parse arguments and run appropriate function
    args = parser.parse_args()

    if args.command == "train":
        train_model(args)
    elif args.command == "evaluate":
        evaluate_model(args)
    elif args.command == "visualize":
        visualize_results(args)
    elif args.command == "test":
        run_tests(args)
    else:
        parser.print_help()
