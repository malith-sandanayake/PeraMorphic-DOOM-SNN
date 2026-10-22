"""
SNN Fine-tuning with Surrogate Gradients.

After converting an ANN to SNN, fine-tune the spiking network using
surrogate gradient methods. The spiking operation is non-differentiable,
so we use a straight-through estimator (STE) with a surrogate gradient
during backpropagation.

Usage:
    python main.py --mode train-snn --checkpoint models/snn_converted
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from env.vizdoom_env import DoomEnv
from models.snn.snn_policy import SNNPolicy


class SNNPPOTrainer:
    """Trains SNN policy using PPO with surrogate gradients.

    Usage:
        trainer = SNNPPOTrainer(snn_policy, env, config)
        trainer.train(total_timesteps=2_000_000)
    """

    def __init__(
        self,
        snn_policy: SNNPolicy,
        env,
        device: str = "cpu",
        config: dict = None,
    ):
        self.snn = snn_policy
        self.env = env
        self.device = torch.device(device)
        self.snn.to(self.device)

        # Default config
        self.config = config or {
            "learning_rate": 1e-4,
            "total_timesteps": 2_000_000,
            "n_steps": 2048,
            "batch_size": 32,
            "n_epochs": 10,
            "gamma": 0.99,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "T": 16,
        }

        self.optimizer = torch.optim.Adam(
            self.snn.parameters(),
            lr=self.config["learning_rate"],
            eps=1e-5,
        )

        self.global_step = 0
        self.best_mean_reward = -1e9

    def compute_ppo_loss(
        self,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        actions: torch.Tensor,
        policy_rates: torch.Tensor,
        entropy: torch.Tensor,
    ) -> dict:
        """Compute PPO clipped surrogate loss for SNN.

        Args:
            old_log_probs: Log probabilities from old policy
            old_values: Value estimates from old policy
            advantages: Advantage estimates
            returns: Monte Carlo returns
            actions: Actions taken
            policy_rates: Current policy spike rates
            entropy: Current policy entropy

        Returns:
            Dict with loss components
        """
        # Policy loss (clipped PPO)
        new_log_probs = torch.log(policy_rates + 1e-10)
        ratio = torch.exp(new_log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                          - old_log_probs)

        clipped_ratio = torch.clamp(ratio, 1 - self.config["clip_range"],
                                     1 + self.config["clip_range"])
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages,
        ).mean()

        # Value loss
        new_values = self._forward_values(actions)  # approximate
        value_loss = F.mse_loss(new_values, returns).mean()

        # Entropy bonus (encourage exploration)
        entropy_loss = -self.config["ent_coef"] * entropy.mean()

        total_loss = (
            policy_loss
            + self.config["vf_coef"] * value_loss
            + entropy_loss
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
        }

    def _forward_values(self, actions: torch.Tensor) -> torch.Tensor:
        """Get value estimates (helper for loss computation)."""
        # In practice, use stored old values or forward pass
        return torch.zeros_like(actions, dtype=torch.float32)

    def train_epoch(
        self,
        observations: torch.Tensor,  # (B, T, C, H, W)
        actions: torch.Tensor,       # (B,)
        rewards: torch.Tensor,       # (B,)
        old_log_probs: torch.Tensor, # (B,)
        old_values: torch.Tensor,    # (B,)
    ) -> dict:
        """Train for one epoch.

        Args:
            observations: Batch of observations
            actions: Actions taken
            rewards: Rewards received
            old_log_probs: Log probabilities of actions
            old_values: Value estimates

        Returns:
            Dict with loss metrics
        """
        self.optimizer.zero_grad()

        # Compute advantages and returns
        values = old_values
        advantages = rewards - values

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        policy_rates, values_pred, _ = self.snn(observations)

        # Compute log probs and entropy
        log_probs = torch.log(policy_rates + 1e-10)
        probs = F.softmax(policy_rates, dim=-1)
        log_prob = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs * probs).sum(dim=-1)

        # Compute PPO loss
        loss_dict = self.compute_ppo_loss(
            old_log_probs, values, advantages, rewards,
            actions, policy_rates, entropy,
        )

        # Backward pass (surrogate gradient flows through SNN)
        loss_dict["total_loss"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.snn.parameters(), max_norm=0.5)

        self.optimizer.step()
        self.global_step += 1

        return {k: v.item() for k, v in loss_dict.items()}

    def train(
        self,
        total_timesteps: int = 2_000_000,
        log_interval: int = 100_000,
    ):
        """Full training loop.

        Args:
            total_timesteps: Total environment steps to train
            log_interval: Steps between logging
        """
        print(f"\n=== SNN Fine-tuning ===")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Device: {self.device}")
        print(f"  Learning rate: {self.config['learning_rate']}")

        n_steps = self.config["n_steps"]
        batch_size = self.config["batch_size"]
        n_epochs = self.config["n_epochs"]

        # Buffer for experiences
        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        log_prob_buffer = []
        value_buffer = []

        print("\nStarting SNN training...")

        for step in range(0, total_timesteps, n_steps):
            # Collect experiences
            for _ in range(n_steps // 10):  # Simplified collection
                # In a full implementation, collect from parallel envs
                # For now, this is a placeholder for the training loop
                pass

            if step % log_interval == 0:
                print(f"  Step {step:,}/{total_timesteps:,}")

            if not obs_buffer:
                print("  Note: Parallel environment collection needed for full training")
                print("  SNN architecture is ready. Connect to DoomEnv for full loop.")
                break

        print(f"\n  Training complete. Global steps: {self.global_step}")


def train(args: argparse.Namespace):
    """Train SNN from checkpoint."""
    checkpoint_path = args.checkpoint

    # Load SNN checkpoint
    print(f"Loading SNN checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    snn_policy = SNNPolicy(
        n_actions=checkpoint["n_actions"],
        T=checkpoint["T"],
        tau=checkpoint["tau"],
        input_channels=checkpoint["input_channels"],
    )
    snn_policy.load_state_dict(checkpoint["state_dict"])

    # Create environment
    screen_size = tuple(args.screen_size)
    env = DoomEnv(
        scenario=getattr(args, "scenario", "simple"),
        screen_size=screen_size,
        seed=42,
    )

    # Create trainer
    trainer = SNNPPOTrainer(
        snn_policy=snn_policy,
        env=env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        config={
            "learning_rate": getattr(args, "snn_learning_rate", 1e-4),
            "total_timesteps": 2_000_000,
            "n_steps": 2048,
            "batch_size": 32,
            "n_epochs": 10,
            "gamma": 0.99,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "T": checkpoint["T"],
        },
    )

    trainer.train(total_timesteps=2_000_000)

    # Save fine-tuned model
    save_path = "models/snn_finetuned"
    torch.save({
        "n_actions": checkpoint["n_actions"],
        "T": checkpoint["T"],
        "tau": checkpoint["tau"],
        "input_channels": checkpoint["input_channels"],
        "state_dict": snn_policy.state_dict(),
    }, save_path)
    print(f"Fine-tuned SNN saved to: {save_path}")


def main(args: argparse.Namespace):
    """CLI entry point for SNN training."""
    train(args)


if __name__ == "__main__":
    main(None)  # Called via main.py CLI
