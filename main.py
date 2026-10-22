"""
Main entry point for PeraMorphic-DOOM-SNN.

Provides a CLI for training, converting, and evaluating DOOM agents.

Usage:
    # Train ANN baseline
    python main.py --mode train-ann --scenario simple

    # Evaluate trained agent
    python main.py --mode evaluate --model models/ann_simple --episodes 100

    # Convert ANN to SNN
    python main.py --mode convert --checkpoint models/ann_simple

    # Fine-tune SNN
    python main.py --mode train-snn --checkpoint models/snn_converted

    # Test random agent
    python main.py --mode test-random --scenario simple --episodes 10
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_ann import main as train_ann_main
from training.convert_and_simulate import main as convert_main
from training.train_snn import main as train_snn_main
from eval.evaluate import main as evaluate_main


def main():
    parser = argparse.ArgumentParser(
        description="PeraMorphic-DOOM-SNN: Autonomous DOOM via SNNs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  train-ann      Train PPO ANN baseline on ViZDoom
  train-snn      Fine-tune SNN with surrogate gradients
  convert        Convert trained ANN to SNN
  evaluate       Evaluate trained agent across episodes
  test-random    Test random agent for environment verification

Examples:
  python main.py --mode train-ann --scenario simple
  python main.py --mode evaluate --model models/ann_simple --episodes 100
  python main.py --mode convert --checkpoint models/ann_simple
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["train-ann", "train-snn", "convert", "evaluate", "test-random"],
        required=True,
        help="Operation mode",
    )

    # Common args
    parser.add_argument(
        "--scenario",
        choices=["simple", "deadly_corridor", "deathmatch"],
        default="simple",
        help="ViZDoom scenario (default: simple)",
    )
    parser.add_argument(
        "--checkpoint", "--model",
        default="models/ann_simple",
        help="Path to model checkpoint (default: models/ann_simple)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes for evaluation (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--screen-size",
        nargs=2,
        type=int,
        default=[84, 84],
        help="Screen size (width height) for preprocessing (default: 84 84)",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Include depth buffer in observations",
    )

    # Train-ANN specific args
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total timesteps for ANN training",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )

    # Train-SNN specific args
    parser.add_argument(
        "--snn-tau",
        type=float,
        default=16.0,
        help="LIF neuron tau for SNN (default: 16.0)",
    )
    parser.add_argument(
        "--simulation-steps",
        type=int,
        default=16,
        help="Number of simulation steps per env step (default: 16)",
    )
    parser.add_argument(
        "--snn-learning-rate",
        type=float,
        default=1e-4,
        help="SNN fine-tuning learning rate (default: 1e-4)",
    )

    args = parser.parse_args()

    if args.mode == "train-ann":
        train_ann_main(args)
    elif args.mode == "train-snn":
        train_snn_main(args)
    elif args.mode == "convert":
        convert_main(args)
    elif args.mode == "evaluate":
        evaluate_main(args)
    elif args.mode == "test-random":
        test_random(args)


def test_random(args):
    """Test environment with random actions.

    Verifies that ViZDoom, the Gym wrapper, and preprocessing all work.
    """
    from env.vizdoom_env import DoomEnv
    from utils.preprocess import FrameProcessor

    print(f"Testing random agent on scenario '{args.scenario}'...")

    screen_size = tuple(args.screen_size)
    env = DoomEnv(
        scenario=args.scenario,
        screen_size=screen_size,
        use_depth=args.use_depth,
        seed=args.seed,
    )

    processor = FrameProcessor(
        target_size=tuple(args.screen_size),
        stack_size=4,
        normalize=True,
    )

    total_reward = 0.0
    stats = {"total_reward": 0.0, "total_kills": 0, "total_steps": 0}

    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        stats["total_reward"] += episode_reward
        stats["total_kills"] += info.get("kills", 0)
        stats["total_steps"] += info.get("episode_steps", 0)

        if (episode + 1) % 10 == 0:
            print(
                f"  Episode {episode + 1}/{args.episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Kills: {info.get('kills', 0)} | "
                f"Steps: {info.get('episode_steps', 0)}"
            )

    env.close()

    print("\n=== Random Agent Results ===")
    print(f"  Episodes: {args.episodes}")
    print(f"  Avg Reward: {stats['total_reward'] / args.episodes:.2f}")
    print(f"  Avg Kills: {stats['total_kills'] / args.episodes:.2f}")
    print(f"  Avg Steps: {stats['total_steps'] / args.episodes:.0f}")
    print("\nEnvironment setup verified successfully!")


if __name__ == "__main__":
    main()
