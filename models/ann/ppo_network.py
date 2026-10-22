"""
PPO Network Architecture for DOOM AI.

Implements the DoomNet-style CNN policy network used with PPO for
ViZDoom environments. Architecture mirrors the standard DoomNet from
the ViZDoom paper (Budzianowski & Koutnar, 2016).

Architecture:
    Input: Stacked frames (4 frames × 2 channels = 8 channels)
    Conv1: 32 filters, 8×8, stride 4 → (19, 19, 32)
    Conv2: 64 filters, 4×4, stride 2 → (9, 9, 64)
    Conv3: 64 filters, 3×3, stride 1 → (7, 7, 64)
    FC1: 512 units
    Policy head: N actions (action logits)
    Value head: 1 scalar (state value)

Usage:
    from models.ann.ppo_network import DoomNet
    model = DoomNet(n_actions=12)
    policy_logits, values = model(observations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoomNet(nn.Module):
    """DoomNet CNN policy network for PPO in ViZDoom.

    A 3-layer CNN that extracts features from stacked DOOM frames,
    followed by two head networks (policy and value).

    Args:
        input_channels: Number of input channels (8 for 4-frame RGB stack)
        n_actions: Number of discrete actions in the action space
        hidden_dim: Size of the fully connected hidden layer (default: 512)
    """

    def __init__(
        self,
        input_channels: int = 8,
        n_actions: int = 12,
        hidden_dim: int = 512,
    ):
        super().__init__()

        # Convolutional feature extractor (DoomNet architecture)
        self.features = nn.Sequential(
            # Layer 1: (8, 84, 84) → (32, 19, 19)
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Layer 2: (32, 19, 19) → (64, 9, 9)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Layer 3: (64, 9, 9) → (64, 7, 7)
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.fc1_relu = nn.ReLU()

        # Policy head: output action logits
        self.policy_head = nn.Linear(hidden_dim, n_actions)

        # Value head: output state value
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, C, H, W)

        Returns:
            policy_logits: Action logits (batch, n_actions)
            values: State values (batch, 1)
        """
        features = self.features(x).flatten(1)
        fc_output = self.fc1_relu(self.fc1(features))

        policy_logits = self.policy_head(fc_output)
        values = self.value_head(fc_output)

        return policy_logits, values

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor = None,
    ) -> tuple:
        """Forward pass with optional action value computation.

        Args:
            x: Input tensor (batch, C, H, W)
            action: Optional action tensor (batch) — if provided,
                    returns log_prob, entropy, and value for that action

        Returns:
            If action is None: (policy_logits, values)
            If action is provided: (log_probs, values, entropy)
        """
        policy_logits, values = self(x)

        if action is None:
            return policy_logits, values

        distribution = torch.distributions.Categorical(
            F.softmax(policy_logits, dim=-1)
        )
        chosen_action = distribution.log_prob(action)
        entropy = distribution.entropy()

        return chosen_action, values.squeeze(-1), entropy


def test_doom_net():
    """Unit test: verify network shapes and forward pass."""
    n_actions = 12
    batch_size = 4
    input_channels = 8
    input_size = 84

    model = DoomNet(input_channels=input_channels, n_actions=n_actions)

    x = torch.randn(batch_size, input_channels, input_size, input_size)
    logits, values = model(x)

    assert logits.shape == (batch_size, n_actions), (
        f"Expected policy shape ({batch_size}, {n_actions}), got {logits.shape}"
    )
    assert values.shape == (batch_size, 1), (
        f"Expected value shape ({batch_size}, 1), got {values.shape}"
    )

    # Test action_and_value
    actions = torch.randint(0, n_actions, (batch_size,))
    log_probs, vals, entropy = model.get_action_and_value(x, actions)
    assert log_probs.shape == (batch_size,)
    assert vals.shape == (batch_size,)
    assert entropy.shape == (batch_size,)

    print("All DoomNet tests passed!")


if __name__ == "__main__":
    test_doom_net()
