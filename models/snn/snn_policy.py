"""
SNN Policy Network for DOOM AI.

Provides a clean policy interface for SNN-based reinforcement learning.
Handles action selection from spike rates and integrates with the
PPO training loop.

Usage:
    from models.snn.snn_policy import SNNPolicy
    policy = SNNPolicy(n_actions=12)
    action, log_prob, value = policy.act(observation, deterministic=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.snn.snn_cnn import SpikingCNN


class SNNPolicy(nn.Module):
    """SNN policy wrapper for action selection and PPO training.

    Wraps SpikingCNN to provide a consistent API for:
    - Action selection (with/without randomness)
    - Log probability computation for policy gradient
    - Value estimation for advantage computation
    """

    def __init__(
        self,
        n_actions: int = 12,
        T: int = 16,
        tau: float = 16.0,
        leak: float = 1.0,
        V_thresh: float = 1.0,
        input_channels: int = 8,
    ):
        super().__init__()
        self.network = SpikingCNN(
            n_actions=n_actions,
            T=T,
            tau=tau,
            leak=leak,
            V_thresh=V_thresh,
            input_channels=input_channels,
        )
        self.n_actions = n_actions
        self.T = T

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor = None,
    ) -> dict:
        """Full forward pass.

        Args:
            x: Input tensor (batch, T, C, H, W)
            action: Optional action tensor for loss computation

        Returns:
            Dictionary with action, log_prob, value, entropy (if action provided)
        """
        if action is None:
            policy_rate, values, hidden = self.network(x)
            # Sample action from rates (treated as logits)
            dist = torch.distributions.Categorical(
                F.softmax(policy_rate, dim=-1)
            )
            selected_action = dist.sample()
            log_prob = dist.log_prob(selected_action)
            entropy = dist.entropy()

            return {
                "action": selected_action,
                "log_prob": log_prob,
                "value": values,
                "entropy": entropy,
            }
        else:
            log_prob, values, entropy, hidden = self.network.get_action_and_value(
                x, action
            )
            return {
                "action": action,
                "log_prob": log_prob,
                "value": values,
                "entropy": entropy,
                "hidden": hidden,
            }

    def act(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select an action for a single observation.

        Accepts either:
        - Stacked frames: (T, C, H*stack_size, W) — shape 4
        - Already shaped sequence: (1, T, C, H, W) — shape 5

        Args:
            observation: Observation tensor
            deterministic: If True, always pick the action with highest rate

        Returns:
            action: Selected action (scalar tensor)
            value: State value estimate (scalar tensor)
        """
        with torch.no_grad():
            # Handle different input shapes
            if observation.ndim == 4:
                # (T, C, H*stack, W) — add batch dim
                x = observation.unsqueeze(0)
            elif observation.ndim == 5:
                # (B=1, T, C, H, W) — already shaped
                x = observation
            else:
                raise ValueError(
                    f"Expected 4D or 5D observation, got {observation.ndim}D"
                )

            policy_rate, value, _ = self.network(x)

            if deterministic:
                action = torch.argmax(policy_rate, dim=-1)
            else:
                dist = torch.distributions.Categorical(
                    F.softmax(policy_rate, dim=-1)
                )
                action = dist.sample()

            return action.squeeze(0), value.squeeze(0)

    def convert_from_ann(self, ann_model) -> None:
        """Convert ANN weights to SNN weights.

        Args:
            ann_model: Trained DoomNet model from models.ann.ppo_network
        """
        self.network.convert_from_ann(ann_model)
