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
from typing import Tuple

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

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        terminated: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE).

        Computes advantage estimates and discounted returns using GAE,
        which balances bias-variance trade-off through the lambda
        parameter.

        Args:
            rewards: Reward sequence (T,)
            values: Value estimates for each timestep (T+1,)
                values[-1] is the bootstrap value
            terminated: Termination flags (T,) — True when episode ended
            gamma: Discount factor
            gae_lambda: GAE smoothing parameter (0=TD, 1=Monte Carlo)

        Returns:
            advantages: GAE advantage estimates (T,)
            returns: Discounted returns (T,)
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                # Last timestep: no bootstrap, handle episode end
                next_value = 0.0
            else:
                next_value = values[t + 1]

            # TD error: r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            next_values = next_value * (1 - terminated[t].float())
            delta = rewards[t] + gamma * next_values - values[t]
            advantages[t] = gae = delta + gamma * gae_lambda * gae * (1 - terminated[t].float())

        returns = advantages + values[:T]
        return advantages, returns

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
            policy_rates: Current policy spike rates (used as logits)
            entropy: Current policy entropy

        Returns:
            Dict with loss components
        """
        # Policy loss (clipped PPO)
        # policy_rates are spike counts averaged over T timesteps.
        # Treat them as action logits for categorical distribution.
        new_log_probs = F.log_softmax(policy_rates, dim=-1)
        ratio = torch.exp(
            new_log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            - old_log_probs
        )

        clipped_ratio = torch.clamp(
            ratio,
            1 - self.config["clip_range"],
            1 + self.config["clip_range"],
        )
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages,
        ).mean()

        # Value loss — use values from forward pass
        value_loss = F.mse_loss(old_values, returns)

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
        # In practice, values come from the forward pass in train_epoch.
        # This method is kept as a placeholder; actual values are threaded
        # through from train_epoch to avoid redundant forward passes.
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

        # Forward pass through SNN
        policy_rates, values_pred, _ = self.snn(observations)

        # Compute log probs and entropy
        log_probs = F.log_softmax(policy_rates, dim=-1)
        probs = F.softmax(policy_rates, dim=-1)
        log_prob = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs * probs).sum(dim=-1)

        # Compute PPO loss (values from forward pass, not zeros)
        loss_dict = self.compute_ppo_loss(
            old_log_probs, values_pred, advantages, rewards,
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

        Rolls out the SNN in the environment to collect experiences,
        then performs PPO updates over the collected batch.

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
        terminated_buffer = []

        T = self.config["T"]
        gamma = self.config["gamma"]
        gae_lambda = self.config.get("gae_lambda", 0.95)

        print("\nStarting SNN training...")

        # --- Collect experiences from environment ---
        obs, _ = self.env.reset()
        for step in range(0, total_timesteps, n_steps):
            for _ in range(n_steps):
                # Convert observation to tensor: (H, W, C) -> (T, C, H, W)
                obs_tensor = torch.from_numpy(obs).to(self.device)
                obs_tensor = obs_tensor.permute(2, 0, 1)  # (C, H, W)
                obs_tensor = obs_tensor.repeat(T, 1, 1)  # (T, C, H, W)

                # Act: select action, get log_prob and value
                with torch.no_grad():
                    # Use SNNPolicy.forward() for consistent log_prob
                    result = self.snn(obs_tensor.unsqueeze(0))
                    action = result["action"]
                    log_prob = result["log_prob"]
                    value = result["value"]

                # Step environment
                next_obs, reward, terminated_flag, truncated_flag, info = self.env.step(
                    int(action.item())
                )
                done = terminated_flag or truncated_flag

                # Collect experiences
                obs_buffer.append(obs_tensor)  # (T, C, H, W)
                action_buffer.append(action)
                reward_buffer.append(torch.tensor(reward, device=self.device))
                log_prob_buffer.append(log_prob)
                value_buffer.append(value)
                terminated_buffer.append(terminated_flag)

                obs = next_obs

                # Handle episode end
                if done:
                    obs, _ = self.env.reset()
                    if "episode" in info and "r" in info["episode"]:
                        pass  # reward tracked via reward_buffer

            # --- PPO update over collected batch ---
            # Compute GAE advantages and returns
            values_arr = torch.stack(value_buffer)
            rewards_arr = torch.stack(reward_buffer)
            terminated_arr = torch.tensor(
                terminated_buffer, dtype=torch.bool, device=self.device
            )

            advantages, returns = self.compute_gae(
                rewards=rewards_arr,
                values=values_arr,
                terminated=terminated_arr,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )

            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Run multiple epochs of PPO updates
            # Shuffle indices
            indices = torch.randperm(len(obs_buffer))

            for epoch in range(n_epochs):
                shuffled_indices = indices[torch.randperm(len(indices))]

                # Create batches
                batches = []
                for start in range(0, len(shuffled_indices), batch_size):
                    end = min(start + batch_size, len(shuffled_indices))
                    batches.append(shuffled_indices[start:end])

                epoch_losses = []
                for batch_idx in batches:
                    # Stack batch tensors: (T, C, H, W) -> (B, T, C, H, W)
                    batch_obs = torch.stack(
                        [obs_buffer[i] for i in batch_idx]
                    )  # (B, T, C, H, W)
                    batch_actions = torch.stack(
                        [action_buffer[i] for i in batch_idx]
                    )
                    batch_log_probs = torch.stack(
                        [log_prob_buffer[i] for i in batch_idx]
                    )
                    batch_values = torch.stack(
                        [value_buffer[i] for i in batch_idx]
                    )

                    # Compute advantages and returns for this batch
                    batch_advantages = advantages[batch_idx]
                    batch_returns = returns[batch_idx]

                    # Forward pass + backward (with surrogate gradient)
                    policy_rates, values_pred, _ = self.snn(batch_obs)

                    # Compute log_probs and entropy using log_softmax
                    log_probs = F.log_softmax(policy_rates, dim=-1)
                    probs = F.softmax(policy_rates, dim=-1)
                    log_prob = log_probs.gather(1, batch_actions.unsqueeze(-1)).squeeze(-1)
                    entropy = -(log_probs * probs).sum(dim=-1)

                    # PPO loss
                    ratio = torch.exp(log_prob - batch_log_probs)
                    clipped_ratio = torch.clamp(ratio, 1 - self.config["clip_range"], 1 + self.config["clip_range"])
                    policy_loss = -torch.min(
                        ratio * batch_advantages,
                        clipped_ratio * batch_advantages,
                    ).mean()

                    value_loss = F.mse_loss(values_pred, batch_returns)
                    entropy_loss = -self.config["ent_coef"] * entropy.mean()
                    total_loss = policy_loss + self.config["vf_coef"] * value_loss + entropy_loss

                    # Backward pass
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.snn.parameters(), max_norm=0.5)
                    self.optimizer.step()

                    epoch_losses.append({
                        "total_loss": total_loss.item(),
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy_loss": entropy_loss.item(),
                    })

                # Log epoch loss
                if epoch_losses:
                    mean_losses = {
                        k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
                        for k in epoch_losses[0]
                    }
                    if step % log_interval == 0:
                        print(f"  Epoch {epoch}: {mean_losses['total_loss']:.4f}")

            # Clear buffers for next collection
            obs_buffer.clear()
            action_buffer.clear()
            reward_buffer.clear()
            log_prob_buffer.clear()
            value_buffer.clear()
            terminated_buffer.clear()

            if step % log_interval == 0:
                print(f"  Step {step:,}/{total_timesteps:,} — global steps: {self.global_step}")

        print(f"\n  Training complete. Global steps: {self.global_step}")


def train(args: argparse.Namespace):
    """Train SNN from checkpoint."""
    checkpoint_path = args.checkpoint

    # Load SNN checkpoint
    print(f"Loading SNN checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

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
