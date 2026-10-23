"""
Visualization utilities for DOOM AI agents.

Handles frame recording, spike visualization, and training metric plotting.

Usage:
    from eval.visualize import FrameRecorder, plot_training_metrics
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Optional
from PIL import Image

try:
    import cv2

    _HAS_OPENCV = True
except ImportError:
    _HAS_OPENCV = False


class FrameRecorder:
    """Records DOOM gameplay frames to video.

    Usage:
        recorder = FrameRecorder(output_path="eval/videos/simple.mp4", fps=35)
        recorder.start()
        for obs in observation_stream:
            recorder.record(obs)
        recorder.stop()
    """

    def __init__(
        self,
        output_path: str = "eval/videos/gameplay.mp4",
        fps: int = 35,
        screen_size: tuple = (64, 48),
    ):
        """Initialize frame recorder.

        Args:
            output_path: Output video file path
            fps: Frames per second
            screen_size: (width, height) of source frames
        """
        self.output_path = output_path
        self.fps = fps
        self.screen_size = screen_size
        self._frames: List[np.ndarray] = []
        self._recording = False

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def start(self) -> None:
        """Start recording frames."""
        self._frames = []
        self._recording = True

    def record(self, frame: np.ndarray) -> None:
        """Record a single frame.

        Args:
            frame: RGB frame array (H, W, 3)
        """
        if not self._recording:
            return

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]  # Remove alpha channel

        self._frames.append(frame)

    def stop(self) -> Optional[str]:
        """Stop recording and save video.

        Returns:
            Path to saved video file, or None if no frames recorded
        """
        if not self._frames:
            return None

        self._save_video()
        self._recording = False
        return self.output_path

    def _save_video(self) -> None:
        """Save recorded frames to video file.

        Uses OpenCV VideoWriter for MP4 encoding. Falls back to
        individual PNG export if OpenCV is not available.
        """
        if _HAS_OPENCV and len(self._frames) > 0:
            # Determine frame size from first frame
            h, w = self._frames[0].shape[:2]
            frame_size = (w, h)

            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, frame_size
            )

            for frame in self._frames:
                # Ensure frame is 3-channel and correct dtype
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.shape[2] == 4:
                    frame = frame[:, :, :3]  # Remove alpha

                frame_uint8 = frame.astype(np.uint8)
                if frame_uint8.shape[:2] != (h, w):
                    frame_uint8 = cv2.resize(frame_uint8, (w, h))

                video_writer.write(frame_uint8)

            video_writer.release()
            print(f"  Video saved to {self.output_path} "
                  f"({len(self._frames)} frames, {self.fps} fps)")
        else:
            # Fallback: save as individual PNG frames
            frame_dir = Path(self.output_path).parent / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)

            for i, frame in enumerate(self._frames):
                img = Image.fromarray(frame.astype(np.uint8))
                img.save(frame_dir / f"frame_{i:04d}.png")

            print(f"  Saved {len(self._frames)} frames to {frame_dir}")
            print(f"  To create video: ffmpeg -r {self.fps} -i "
                  f"'{frame_dir}/frame_%04d.png' -c:v libx264 "
                  f"'{self.output_path}'")

    def record_episode(self, env, model=None, n_frames: int = 1000) -> Optional[str]:
        """Record a complete episode.

        Args:
            env: Gymnasium environment
            model: Optional model for playing (None = random)
            n_frames: Max frames to record

        Returns:
            Path to saved video, or None
        """
        self.start()
        obs, info = env.reset()
        done = False
        frames_recorded = 0

        while not done and frames_recorded < n_frames:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record screen buffer
            screen = env.render()
            if screen is not None:
                self.record(screen)

            frames_recorded += 1

        return self.stop()


def plot_firing_rates(firing_rates: List[float], output_path: str = "eval/firing_rates.png") -> None:
    """Plot neuron firing rates across layers.

    Args:
        firing_rates: List of per-layer firing rates
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    layer_names = ["Conv1", "Conv2", "Conv3", "FC1", "Policy", "Value"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(layer_names, firing_rates, color="steelblue")
    ax.set_title("Neuron Firing Rates Across Layers")
    ax.set_ylabel("Firing Rate")
    ax.set_xlabel("Layer")
    ax.set_ylim(0, max(firing_rates) * 1.1)

    # Add value labels on bars
    for i, rate in enumerate(firing_rates):
        ax.text(i, rate + 0.001, f"{rate:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Firing rate plot saved to {output_path}")
    plt.close()


def plot_training_progress(
    timesteps: List[int],
    rewards: List[float],
    output_path: str = "eval/training_progress.png",
) -> None:
    """Plot training reward progress.

    Args:
        timesteps: List of timestep values
        rewards: List of corresponding mean rewards
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timesteps, rewards, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Progress")
    ax.grid(True, alpha=0.3)

    # Format x-axis with K/M notation
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"{x/1e6:.0f}M")
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Training progress plot saved to {output_path}")
    plt.close()
