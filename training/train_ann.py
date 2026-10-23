"""
ANN Baseline Training — PPO on ViZDoom.

Trains a standard neural network (PPO) to play DOOM using Stable-Baselines3.
Uses the DoomNet architecture from `models.ann.ppo_network`.

Usage:
    python main.py --mode train-ann --scenario simple --episodes 100

    # Custom parameters
    python main.py --mode train-ann --scenario deadly_corridor \\
        --total-timesteps 10000000 --n-envs 4
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

from env.vizdoom_env import DoomEnv


class ProgressCallback(BaseCallback):
    """Callback that prints training progress every N timesteps.

    Usage:
        cb = ProgressCallback(print_every=1_000_000)
        model.learn(5_000_000, callback=cb)
    """

    def __init__(self, print_every: int = 1_000_000):
        super().__init__()
        self.print_every = print_every
        self._last_print_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print_step >= self.print_every:
            self._last_print_step = self.num_timesteps
            local_info = self.locals.get("infos", [{}])
            avg_reward = local_info[0].get("reward", 0) if local_info else 0
            print(
                f"  [Step {self.num_timesteps:,}] "
                f"Last ep reward: {avg_reward:.1f}"
            )
        return True


class EvaluationCallback(BaseCallback):
    """Periodically evaluate the agent during training.

    Usage:
        cb = EvaluationCallback(eval_env, n_eval_episodes=10, verbose=1)
    """

    def __init__(
        self,
        eval_env,
        n_eval_episodes: int = 10,
        callback_after_step: int = 1_000_000,
        verbose: int = 0,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.callback_after_step = callback_after_step
        self.verbose = verbose

    def _on_step(self) -> bool:
        if self.num_timesteps % self.callback_after_step == 0:
            # Evaluate
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
            )
            print(
                f"  [Step {self.num_timesteps:,}] "
                f"Mean eval reward: {mean_reward:.2f} ± {std_reward:.2f}"
            )

            # Save best model — read scenario from env metadata
            scenario = self.model.env.metadata.get("scenario", "simple")
            if mean_reward > getattr(self, "best_mean_reward", -1e9):
                self.best_mean_reward = mean_reward
                save_path = f"models/best_{scenario}"
                self.model.save(save_path)
                if self.verbose:
                    print(f"  New best model saved (reward: {mean_reward:.2f})")

        return True


def get_scenario_info(scenario: str) -> dict:
    """Get scenario-specific configuration.

    Args:
        scenario: Scenario name

    Returns:
        Dict with n_actions, timesteps, and config path
    """
    configs = {
        "simple": {
            "n_actions": 12,
            "timesteps": 5_000_000,
            "config": "config/train/ppo_simple.yaml",
        },
        "deadly_corridor": {
            "n_actions": 12,
            "timesteps": 10_000_000,
            "config": "config/train/ppo_deadly_corridor.yaml",
        },
        "deathmatch": {
            "n_actions": 12,
            "timesteps": 20_000_000,
            "config": "config/train/ppo_deathmatch.yaml",
        },
    }

    if scenario not in configs:
        raise ValueError(f"Unknown scenario: {scenario}. Choose from {list(configs.keys())}")

    return configs[scenario]


def make_env(scenario: str, env_idx: int, screen_size: tuple, use_depth: bool):
    """Create an environment with proper seeding for parallel training.

    Args:
        scenario: Scenario name
        env_idx: Environment index (for different random seeds)
        screen_size: (width, height) for screen observations
        use_depth: Whether to include depth buffer

    Returns:
        Configured DoomEnv instance
    """
    def _init():
        env = DoomEnv(
            scenario=scenario,
            screen_size=screen_size,
            frame_skip=1,
            use_depth=use_depth,
            seed=env_idx,
        )
        return env

    return _init


def train(args: argparse.Namespace, config: dict = None) -> PPO:
    """Train PPO agent on DOOM.

    Args:
        args: Parsed command line arguments

    Returns:
        Trained PPO model
    """
    scenario_info = get_scenario_info(args.scenario)
    total_timesteps = args.total_timesteps or scenario_info["timesteps"]
    screen_size = tuple(args.screen_size)

    print(f"=== Training PPO on {args.scenario} ===")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Screen size: {screen_size}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  Depth buffer: {args.use_depth}")

    # Create parallel environments
    env = SubprocVecEnv(
        [make_env(args.scenario, i, screen_size, args.use_depth)
         for i in range(args.n_envs)],
    )
    env = VecMonitor(env)  # Monitor for TensorBoard

    # Build custom DoomNet policy
    from models.ann.ppo_network import DoomNet
    from stable_baselines3.common.policies import ActorCriticCnnPolicy

    class DoomPolicy(ActorCriticCnnPolicy):
        """Custom policy that uses DoomNet architecture."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Replace SB3's feature extractor with DoomNet
            self.features_extractor = DoomNet(
                input_channels=kwargs.get("features_extractor_channels", 8),
                n_actions=self.n_actions,
                hidden_dim=512,
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass through DoomNet, returning policy and value outputs."""
            policy_logits, values = self.features_extractor(x)
            return policy_logits, values

    # Create PPO model
    lr = (
        args.learning_rate
        if hasattr(args, "learning_rate") and args.learning_rate
        else 3e-4
    )
    model = PPO(
        DoomPolicy,
        env,
        learning_rate=lr,
        n_steps=config.get("n_steps", scenario_info.get("n_steps", 2048)),
        batch_size=config.get("batch_size", scenario_info.get("batch_size", 64)),
        n_epochs=config.get("n_epochs", scenario_info.get("n_epochs", 10)),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.01),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        verbose=1,
    )

    # Callbacks
    callbacks = [
        ProgressCallback(print_every=max(500_000, total_timesteps // 10)),
    ]

    # TensorBoard
    if args.tensorboard:
        log_dir = f"./logs/{args.scenario}"
        os.makedirs(log_dir, exist_ok=True)
        model.tensorboard_log = log_dir

    # Save scenario for callbacks
    model.env.metadata["scenario"] = args.scenario

    # Learn
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    save_path = f"models/ann_{args.scenario}"
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    # Evaluate
    print("Running evaluation...")
    eval_env = DoomEnv(
        scenario=args.scenario,
        screen_size=screen_size,
        use_depth=args.use_depth,
        seed=123,
    )
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    eval_env.close()

    return model


def main(args):
    """CLI entry point for training."""
    # Load config if available
    scenario_info = get_scenario_info(args.scenario)
    config_path = scenario_info["config"]

    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # Override with command line args
        if args.total_timesteps:
            config["total_timesteps"] = args.total_timesteps
        if args.n_envs:
            config["n_envs"] = args.n_envs
    else:
        config = {}

    train(args, config=config)


if __name__ == "__main__":
    main(None)  # Will be called via argparse in main.py
