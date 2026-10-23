"""Unit tests for frame preprocessing utilities.

Tests resize, stacking, and FrameProcessor output shapes.
"""

import pytest
import numpy as np

from utils.preprocess import resize_frame, stack_frames, FrameProcessor


class TestResizeFrame:
    """Test frame resizing."""

    def test_resize_output_shape(self):
        """resize_frame returns correct target size."""
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        resized = resize_frame(frame, target_size=(84, 84))
        assert resized.shape == (84, 84, 3)

    def test_resize_preserves_channels(self):
        """resize_frame preserves channel count."""
        frame = np.zeros((48, 64, 4), dtype=np.uint8)
        resized = resize_frame(frame, target_size=(100, 80))
        assert resized.shape == (80, 100, 4)


class TestStackFrames:
    """Test frame stacking."""

    def test_stack_channels_last(self):
        """stack_frames stacks along channel axis (channels_last=True)."""
        frame = np.zeros((84, 84, 3), dtype=np.uint8)
        frames = [frame] * 4
        stacked = stack_frames(frames, stack_size=4, channels_last=True)
        assert stacked.shape == (84, 84, 12)  # 4 frames × 3 channels

    def test_stack_channels_first(self):
        """stack_frames stacks along channel axis (channels_last=False)."""
        frame = np.zeros((84, 84, 3), dtype=np.uint8)
        frames = [frame] * 4
        stacked = stack_frames(frames, stack_size=4, channels_last=False)
        assert stacked.shape == (12, 84, 84)

    def test_empty_frames_raises(self):
        """stack_frames raises ValueError for empty frames list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            stack_frames([], stack_size=4)

    def test_stack_size_limit(self):
        """stack_frames keeps only the most recent stack_size frames."""
        frames = [np.zeros((84, 84, 3)) for _ in range(10)]
        stacked = stack_frames(frames, stack_size=4, channels_last=True)
        assert stacked.shape[2] == 12  # 4 × 3 channels


class TestFrameProcessor:
    """Test FrameProcessor pipeline."""

    @pytest.fixture
    def processor(self):
        """Create a FrameProcessor with default settings."""
        return FrameProcessor(target_size=(84, 84), stack_size=4, normalize=True)

    def test_process_output_shape(self, processor):
        """FrameProcessor.process returns correct output shape."""
        # Simulate a raw DOOM frame (64×48×3)
        frame = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
        processed = processor.process(frame)
        assert processed.shape == (84, 84, 12)  # 4 stacked × 3 channels
        assert processed.dtype == np.float32

    def test_normalized_range(self, processor):
        """FrameProcessor normalizes to [0, 1]."""
        frame = np.full((48, 64, 3), 255, dtype=np.uint8)
        # Fill buffer with enough frames
        for _ in range(4):
            processed = processor.process(frame)
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0

    def test_reset_clears_buffer(self, processor):
        """FrameProcessor.reset clears the frame buffer."""
        frame = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
        processor.process(frame)
        assert len(processor._frame_buffer) > 0
        processor.reset()
        assert len(processor._frame_buffer) == 0

    def test_preprocess_batch_shape(self, processor):
        """FrameProcessor.preprocess_batch returns correct shape."""
        frames = [np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8) for _ in range(3)]
        batch = processor.preprocess_batch(frames)
        assert batch.shape == (3, 12, 84, 84)  # (N, C, H, W)
