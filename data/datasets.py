"""
Dataset implementations for the layered neural codec.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Optional, Callable, Dict, Any
import torchvision.transforms as transforms
import random
import cv2
from tqdm import tqdm


class VideoFrameDataset(Dataset):
    """Dataset for loading sequences of video frames."""

    def __init__(
        self,
        root_dir: str,
        sequence_length: int = 16,
        frame_size: Tuple[int, int] = (256, 512),
        transform: Optional[Callable] = None,
        split: str = "train",
        train_val_split: float = 0.8,
        seed: int = 42,
    ):
        """
        Args:
            root_dir: Directory containing the dataset
            sequence_length: Number of consecutive frames to load
            frame_size: Size to resize frames to (height, width)
            transform: Optional transform to apply to the frames
            split: Either 'train' or 'val'
            train_val_split: Fraction of data to use for training
            seed: Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.split = split
        self.seed = seed

        # Create default transform if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(frame_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

        # Get all video sequence directories
        self.sequence_dirs = self._get_sequence_dirs()

        # Split into train and validation sets
        random.seed(seed)
        random.shuffle(self.sequence_dirs)
        split_idx = int(len(self.sequence_dirs) * train_val_split)

        if split == "train":
            self.sequence_dirs = self.sequence_dirs[:split_idx]
        elif split == "val":
            self.sequence_dirs = self.sequence_dirs[split_idx:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")

        # Get all valid frame sequences
        self.frame_sequences = self._get_frame_sequences()

    def _get_sequence_dirs(self) -> List[str]:
        """Get all video sequence directories."""
        raise NotImplementedError("Subclasses must implement this method")

    def _get_frame_sequences(self) -> List[Dict[str, Any]]:
        """Get all valid frame sequences."""
        raise NotImplementedError("Subclasses must implement this method")

    def __len__(self) -> int:
        return len(self.frame_sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence_info = self.frame_sequences[idx]
        frame_paths = sequence_info["frame_paths"]

        # Load and transform frames
        frames = []
        for path in frame_paths:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                frames.append(img)

        # Stack frames along new dimension
        frames = torch.stack(frames, dim=0)

        return {
            "frames": frames,
            "sequence_name": sequence_info["sequence_name"],
            "frame_indices": sequence_info["frame_indices"],
        }


class VideoFileDataset(VideoFrameDataset):
    """Dataset for loading frames from video files."""

    def _get_sequence_dirs(self) -> List[str]:
        """Get all valid video files."""
        print(f"Looking for video files in: {self.root_dir}")
        video_files = glob.glob(os.path.join(self.root_dir, "*.mp4"))
        video_files.extend(glob.glob(os.path.join(self.root_dir, "**", "*.mp4"), recursive=True))
        print(f"Found {len(video_files)} potential video files")

        # Pre-check videos to filter out corrupted ones
        valid_videos = []
        print("Checking video files for corruption (this may take a while)...")
        for video_path in tqdm(video_files[:min(1000, len(video_files))]):  # Limit to first 1000 to avoid long scan
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    # Read the first frame to ensure video is valid
                    ret, _ = cap.read()
                    if ret:
                        valid_videos.append(video_path)
                cap.release()
            except Exception as e:
                # Skip this video
                continue

        print(f"Found {len(valid_videos)} valid video files out of {min(1000, len(video_files))} checked")
        return valid_videos

    def _get_frame_sequences(self) -> List[Dict[str, Any]]:
        """Create frame sequences from videos."""
        sequences = []

        print("Creating frame sequences from videos...")
        for video_path in tqdm(self.sequence_dirs):
            # Extract video name
            video_name = os.path.basename(video_path).split('.')[0]

            try:
                # Open video file
                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    print(f"Warning: Could not open video {video_path}")
                    continue

                # Get total frames
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Check if video has enough frames
                if total_frames < self.sequence_length:
                    print(f"Warning: Video {video_name} has only {total_frames} frames, skipping")
                    cap.release()
                    continue

                # Check that frames can actually be read
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, _ = cap.read()
                if not ret:
                    print(f"Warning: Cannot read frames from {video_name}, skipping")
                    cap.release()
                    continue

                # Create sequences with overlap
                stride = max(1, self.sequence_length // 2)  # 50% overlap
                for start_idx in range(0, total_frames - self.sequence_length + 1, stride):
                    sequences.append({
                        "sequence_name": f"{video_name}_{start_idx}",
                        "video_path": video_path,
                        "start_frame": start_idx,
                        "frame_indices": list(range(start_idx, start_idx + self.sequence_length)),
                    })

                cap.release()

            except Exception as e:
                print(f"Error processing video {video_path}: {str(e)}")
                continue

        print(f"Created {len(sequences)} sequences from {len(self.sequence_dirs)} videos")

        # If no sequences were created, raise an error
        if len(sequences) == 0:
            raise RuntimeError(
                "No valid video sequences found. Please check that your video files are not corrupted."
            )

        return sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a video sequence."""
        sequence_info = self.frame_sequences[idx]

        # Create frames list
        frames = []

        try:
            # Open video
            cap = cv2.VideoCapture(sequence_info["video_path"])

            if not cap.isOpened():
                raise RuntimeError(f"Could not open video {sequence_info['video_path']}")

            # Read frames
            for frame_idx in sequence_info["frame_indices"]:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    # If reading fails, create a black frame
                    frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image
                pil_img = Image.fromarray(frame)

                # Apply transform
                if self.transform:
                    pil_img = self.transform(pil_img)
                else:
                    # Default transform if none provided
                    pil_img = transforms.Compose([
                        transforms.Resize(self.frame_size),
                        transforms.ToTensor(),
                    ])(pil_img)

                frames.append(pil_img)

            cap.release()

        except Exception as e:
            print(f"Error loading sequence {sequence_info['sequence_name']}: {str(e)}")
            # Create dummy frames
            for _ in range(self.sequence_length):
                dummy_frame = torch.zeros(3, self.frame_size[0], self.frame_size[1])
                frames.append(dummy_frame)

        # Stack frames along new dimension
        frames = torch.stack(frames, dim=0)

        return {
            "frames": frames,
            "sequence_name": sequence_info["sequence_name"],
            "frame_indices": sequence_info["frame_indices"],
        }


class DummyDataset(VideoFrameDataset):
    """Dummy dataset for testing."""

    def _get_sequence_dirs(self):
        return ["seq1", "seq2", "seq3", "seq4", "seq5"]

    def _get_frame_sequences(self):
        # Create mock frame sequences with random tensors
        sequences = []
        for i in range(5):
            sequences.append({
                "sequence_name": f"seq{i+1}",
                "frame_paths": [f"frame_{i}_{j}" for j in range(self.sequence_length)],
                "frame_indices": list(range(self.sequence_length)),
            })
        return sequences

    def __getitem__(self, idx):
        sequence_info = self.frame_sequences[idx]

        # Create random frames
        frames = []
        for _ in range(len(sequence_info["frame_paths"])):
            dummy_frame = torch.rand(3, *self.frame_size)
            frames.append(dummy_frame)

        # Stack frames along new dimension
        frames = torch.stack(frames, dim=0)

        return {
            "frames": frames,
            "sequence_name": sequence_info["sequence_name"],
            "frame_indices": sequence_info["frame_indices"],
        }


class CityscapesVideoDataset(VideoFrameDataset):
    """Cityscapes dataset adapter for video sequences."""

    def _get_sequence_dirs(self) -> List[str]:
        """Get all city directories containing sequences."""
        base_path = os.path.join(self.root_dir, "leftImg8bit")
        print(f"Looking for Cityscapes data in: {base_path}")

        if not os.path.exists(base_path):
            print(f"ERROR: Path doesn't exist: {base_path}")
            return []

        cities_dirs = glob.glob(os.path.join(base_path, "*", "*"))
        print(f"Found {len(cities_dirs)} potential sequence directories")

        valid_dirs = [d for d in cities_dirs if os.path.isdir(d)]
        print(f"Found {len(valid_dirs)} valid sequence directories")
        return valid_dirs

    def _get_frame_sequences(self) -> List[Dict[str, Any]]:
        """Get valid frame sequences from Cityscapes."""
        frame_sequences = []

        for seq_dir in self.sequence_dirs:
            # Get all frame files sorted by name
            frame_files = sorted(glob.glob(os.path.join(seq_dir, "*_leftImg8bit.png")))

            # Skip if not enough frames
            if len(frame_files) < self.sequence_length:
                continue

            # Create sequences
            for i in range(len(frame_files) - self.sequence_length + 1):
                sequence_frames = frame_files[i:i+self.sequence_length]

                # Extract sequence info
                seq_path_parts = os.path.normpath(seq_dir).split(os.sep)
                city_name = seq_path_parts[-2] if len(seq_path_parts) >= 2 else "unknown"
                seq_name = seq_path_parts[-1] if len(seq_path_parts) >= 1 else "unknown"

                frame_sequences.append({
                    "sequence_name": f"{city_name}/{seq_name}",
                    "frame_paths": sequence_frames,
                    "frame_indices": list(range(i, i+self.sequence_length)),
                })

        return frame_sequences


class KITTIVideoDataset(VideoFrameDataset):
    """KITTI dataset adapter for video sequences."""

    def _get_sequence_dirs(self) -> List[str]:
        """Get all sequence directories."""
        return glob.glob(os.path.join(self.root_dir, "data_scene_flow", "*", "image_2"))

    def _get_frame_sequences(self) -> List[Dict[str, Any]]:
        """Get valid frame sequences from KITTI."""
        frame_sequences = []

        for seq_dir in self.sequence_dirs:
            # Get all frame files sorted by name
            frame_files = sorted(glob.glob(os.path.join(seq_dir, "*.png")))

            # Skip if not enough frames
            if len(frame_files) < self.sequence_length:
                continue

            # Create sequences
            for i in range(len(frame_files) - self.sequence_length + 1):
                sequence_frames = frame_files[i:i+self.sequence_length]

                # Extract sequence info
                seq_path_parts = os.path.normpath(seq_dir).split(os.sep)
                seq_name = seq_path_parts[-2] if len(seq_path_parts) >= 2 else "unknown"

                frame_sequences.append({
                    "sequence_name": seq_name,
                    "frame_paths": sequence_frames,
                    "frame_indices": list(range(i, i+self.sequence_length)),
                })

        return frame_sequences


def get_dataset(config, split="train"):
    """Helper function to get the appropriate dataset."""
    dataset_name = config.data.dataset_name.lower()

    if dataset_name == "dummy":
        return DummyDataset(
            root_dir="./dummy",
            sequence_length=config.data.sequence_length,
            frame_size=config.data.frame_size,
            split=split,
            train_val_split=config.data.train_val_split,
            seed=config.seed,
        )
    elif dataset_name == "video":
        return VideoFileDataset(
            root_dir=config.data.dataset_path,
            sequence_length=config.data.sequence_length,
            frame_size=config.data.frame_size,
            split=split,
            train_val_split=config.data.train_val_split,
            seed=config.seed,
        )
    elif dataset_name == "cityscapes":
        return CityscapesVideoDataset(
            root_dir=config.data.dataset_path,
            sequence_length=config.data.sequence_length,
            frame_size=config.data.frame_size,
            split=split,
            train_val_split=config.data.train_val_split,
            seed=config.seed,
        )
    elif dataset_name == "kitti":
        return KITTIVideoDataset(
            root_dir=config.data.dataset_path,
            sequence_length=config.data.sequence_length,
            frame_size=config.data.frame_size,
            split=split,
            train_val_split=config.data.train_val_split,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_transforms(config, is_train=True):
    """Get transforms based on configuration."""
    transforms_list = [
        transforms.Resize(config.data.frame_size),
    ]

    if is_train and config.data.augmentations:
        transforms_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])

    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transforms.Compose(transforms_list)
