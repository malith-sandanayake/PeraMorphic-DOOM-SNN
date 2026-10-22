"""
Spiking CNN Model for DOOM AI.

Direct spiking equivalent of the DoomNet architecture from
`models.ann.ppo_network`. Each ANN layer maps to a spiking layer:

    ANN Conv2d + ReLU  →  Conv2d + LIF neuron
    ANN Linear + ReLU  →  Linear + LIF neuron
    ANN Linear (output) → Linear (no activation, spike count = policy)

The model processes a sequence of frames over T timesteps, with the
action policy determined by the total spike count across timesteps
(rate coding).

Architecture:
    Input: (batch, T, 8, 84, 84) — T timesteps of 4-frame RGB stacks
    Conv1: Conv2d + LIF → (batch, T, 32, 19, 19)
    Conv2: Conv2d + LIF → (batch, T, 64, 9, 9)
    Conv3: Conv2d + LIF → (batch, T, 64, 7, 7)
    FC1: Linear + LIF → (batch, T, 512)
    Policy: Linear (no activation) → (batch, T, n_actions)
    Value: Linear (no activation) → (batch, T, 1)

Usage:
    from models.snn.snn_cnn import SpikingCNN
    snn = SpikingCNN(n_actions=12, T=16, tau=16.0)
    policy_spikes, value_spikes = snn(input_frames)
    action_probs = policy_spikes.sum(1) / T  # rate coding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.snn.lif_neuron import LIFNode


class SpikingCNN(nn.Module):
    """Spiking CNN policy network for PPO-based DOOM agents.

    Uses rate coding: the fraction of timesteps a neuron spikes
    represents its output value. Actions are selected based on
    total spike counts across the simulation horizon.

    Args:
        n_actions: Number of discrete actions
        T: Number of simulation time steps per environment step
        tau: Membrane time constant for LIF neurons
        leak: Leak factor for LIF neurons
        V_thresh: Firing threshold for LIF neurons
        input_channels: Input channels (8 for 4-frame RGB stack)
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

        self.n_actions = n_actions
        self.T = T

        # === Convolutional layers with LIF activation ===
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # === Fully connected layers with LIF activation ===
        self.fc1 = nn.Linear(64 * 7 * 7, 512)

        # === Output heads (no activation — spike count = probability proxy) ===
        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

        # === LIF neurons ===
        self.lif1 = LIFNode(tau=tau, leak=leak, V_thresh=V_thresh)
        self.lif2 = LIFNode(tau=tau, leak=leak, V_thresh=V_thresh)
        self.lif3 = LIFNode(tau=tau, leak=leak, V_thresh=V_thresh)
        self.lif4 = LIFNode(tau=tau, leak=leak, V_thresh=V_thresh)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass through spiking CNN.

        Processes the input through T timesteps of simulation.

        Args:
            x: Input tensor of shape (batch, T, C, H, W)
               where C = 8 (4 RGB frames stacked), H = W = 84

        Returns:
            policy_spikes: Action spikes (batch, T, n_actions)
            value_spikes: Value spikes (batch, T, 1)
            hidden_state: Dict of hidden states for BPTT training
        """
        B, T, C, H, W = x.shape
        hidden_state = {}

        # Accumulate policy and value spikes across timesteps
        policy_accum = torch.zeros(B, self.n_actions, device=x.device)
        value_accum = torch.zeros(B, 1, device=x.device)

        for t in range(T):
            # Extract frame for timestep t: (B, C, H, W)
            frame = x[:, t]

            # === Convolutional layers with LIF ===
            # Conv1: (B, C, H, W) → (B, 32, 19, 19)
            conv1_out = F.relu(self.conv1(frame))
            conv1_out, (spike1, hidden_state["lif1"]) = self.lif1(
                conv1_out, **hidden_state.get("lif1", {})
            )

            # Conv2: (B, 32, 19, 19) → (B, 64, 9, 9)
            conv2_out = F.relu(self.conv2(conv1_out))
            conv2_out, (spike2, hidden_state["lif2"]) = self.lif2(
                conv2_out, **hidden_state.get("lif2", {})
            )

            # Conv3: (B, 64, 9, 9) → (B, 64, 7, 7)
            conv3_out = F.relu(self.conv3(conv2_out))
            conv3_out, (spike3, hidden_state["lif3"]) = self.lif3(
                conv3_out, **hidden_state.get("lif3", {})
            )

            # Flatten
            flat = conv3_out.flatten(1)  # (B, 64*7*7) = (B, 3136)

            # === FC layers with LIF ===
            # FC1: (B, 3136) → (B, 512)
            fc1_out = F.relu(self.fc1(flat))
            fc1_out, (spike4, hidden_state["lif4"]) = self.lif4(
                fc1_out, **hidden_state.get("lif4", {})
            )

            # === Output heads ===
            policy_logits = self.policy_head(spike4)
            value_logits = self.value_head(spike4)

            policy_accum += policy_logits
            value_accum += value_logits

        # Normalize by T (average spike count = firing rate)
        policy_rate = policy_accum / T
        value_rate = value_accum / T

        return policy_rate, value_rate.squeeze(-1), hidden_state

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with optional action value computation.

        Args:
            x: Input tensor (batch, T, C, H, W)
            action: Optional action tensor (batch)

        Returns:
            If action is None: (policy_rates, values, hidden_state)
            If action is provided: (log_probs, values, entropy, hidden_state)
        """
        policy_rate, values, hidden = self(x)

        if action is None:
            return policy_rate, values, hidden

        # Convert rates to probabilities via softmax
        log_probs = torch.log(policy_rate + 1e-10)
        probs = F.softmax(policy_rate, dim=-1)
        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs * probs).sum(dim=-1)

        return log_prob, values, entropy, hidden

    def convert_from_ann(self, ann_model) -> None:
        """Load weights from an ANN DoomNet model.

        Copies convolutional and linear layer weights from the trained
        ANN model, preserving the same connectivity.

        Args:
            ann_model: An instance of DoomNet from models.ann.ppo_network
        """
        with torch.no_grad():
            # Copy convolutional weights (ann_model.features is a Sequential)
            snn_conv_layers = [self.conv1, self.conv2, self.conv3]
            for ann_idx, snn_conv in enumerate(snn_conv_layers, start=0):
                ann_conv = ann_model.features[ann_idx * 2]  # conv is at even indices (0, 2, 4)
                snn_conv.weight.copy_(ann_conv.weight)
                if ann_conv.bias is not None:
                    snn_conv.bias.copy_(ann_conv.bias)

            # Copy FC weights
            self.fc1.weight.copy_(ann_model.fc1.weight)
            if ann_model.fc1.bias is not None:
                self.fc1.bias.copy_(ann_model.fc1.bias)

            # Copy head weights
            self.policy_head.weight.copy_(ann_model.policy_head.weight)
            if ann_model.policy_head.bias is not None:
                self.policy_head.bias.copy_(ann_model.policy_head.bias)

            self.value_head.weight.copy_(ann_model.value_head.weight)
            if ann_model.value_head.bias is not None:
                self.value_head.bias.copy_(ann_model.value_head.bias)

        print("ANN weights copied to SNN model.")
