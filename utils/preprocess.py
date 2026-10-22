"""
Frame preprocessing utilities for DOOM vision.

Handles frame resizing, stacking, normalization, and format conversion
for ViZDoom observations.
"""

import numpy as np
from typing import List, Optional, Tuple

import cv2


def resize_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (84, 84),
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize a DOOM frame to target size.

    Args:
        frame: Input frame (H, W, C) — RGB or RGBA
        target_size: Target (width, height) — Note: cv2 uses (W, H)
        interpolation: OpenCV interpolation method

    Returns:
        Resized frame (target_height, target_width, C)
    """
    # cv2.resize expects (width, height)
    return cv2.resize(frame, target_size, interpolation=interpolation)


def stack_frames(
    frames: List[np.ndarray],
    stack_size: int = 4,
    channels_last: bool = True,
) -> np.ndarray:
    """Stack multiple frames along the channel axis.

    Maintains temporal context by stacking recent frames,
    similar to the Atari DQN approach.

    Args:
        frames: List of frames to stack (each: H, W, C or H, W)
        stack_size: Number of frames to keep in stack
        channels_last: If True, stack along last axis (H, W, C*stack)
                      If False, stack along first axis (C*stack, H, W)

    Returns:
        Stacked frame array

    Raises:
        ValueError: If frames list is empty
    """
    if not frames:
        raise ValueError("Frames list cannot be empty")

    # Take the most recent `stack_size` frames
    recent = frames[-stack_size:]

    if channels_last:
        return np.concatenate(recent, axis=-1)
    else:
        return np.concatenate([f.transpose(2, 0, 1) for f in recent], axis=0)


class FrameProcessor:
    """Configurable frame preprocessing pipeline for ViZDoom observations.

    Usage:
        processor = FrameProcessor(target_size=(84, 84), stack_size=4)
        stacked = processor.process(raw_frame)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (84, 84),
        stack_size: int = 4,
        normalize: bool = True,
        grayscale: bool = False,
    ):
        """Initialize frame processor.

        Args:
            target_size: Target (width, height) for frame resizing
            stack_size: Number of frames to stack
            normalize: Whether to normalize pixel values to [0, 1]
            grayscale: Whether to convert to grayscale
        """
        self.target_size = target_size
        self.stack_size = stack_size
        self.normalize = normalize
        self.grayscale = grayscale
        self._frame_buffer: List[np.ndarray] = []

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame: resize, grayscale, stack, normalize.

        Args:
            frame: Raw frame from ViZDoom (64x48x3 RGB)

        Returns:
            Processed stacked observation
        """
        # Resize
        frame = resize_frame(frame, self.target_size)

        # Grayscale if requested
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = frame[..., np.newaxis]  # Add channel dim

        # Add to buffer
        self._frame_buffer.append(frame)
        if len(self._frame_buffer) > self.stack_size:
            self._frame_buffer.pop(0)

        # Stack
        stacked = stack_frames(self._frame_buffer, self.stack_size, channels_last=True)

        # Normalize
        if self.normalize:
            stacked = stacked.astype(np.float32) / 255.0

        return stacked

    def reset(self) -> None:
        """Clear the frame buffer."""
        self._frame_buffer = []

    def preprocess_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 4,
    ) -> np.ndarray:
        """Process a batch of frames into training-ready tensors.

        Args:
            frames: List of raw frames
            batch_size: Number of frames per batch

        Returns:
            Processed batch (N, C, H, W) format for PyTorch
        """
        processed = []
        for frame in frames:
            p = self.process(frame)
            processed.append(p.transpose(2, 0, 1))  # HWC -> CHW

        return np.stack(processed, axis=0)
