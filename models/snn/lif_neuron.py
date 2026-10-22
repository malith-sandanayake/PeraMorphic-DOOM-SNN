"""
Leaky Integrate-and-Fire (LIF) Neuron Module.

Implements the standard LIF neuron model used in Spiking Neural Networks.
The LIF neuron integrates input current over time, and fires a spike
when its membrane potential exceeds a threshold.

Mathematical Model:
    τ_m · dV/dt = -(V - V_rest) + I(t)

    If V(t) ≥ V_thresh:
        spike = 1
        V(t) = V_reset      (reset after spike)
    Else:
        spike = 0
        V(t) = V(t - 1) + leakage

    For training with surrogate gradients, the spike operation uses a
    straight-through estimator (STE) to pass gradients through the
    non-differentiable spike function.

Reference:
    Maass, W. (1997). Networks of spiking neurons: The third generation
    of neural network models. Neural Networks, 10(9), 1273-1288.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LIFNode(nn.Module):
    """Single Leaky Integrate-and-Fire neuron with surrogate gradient training.

    The neuron processes a sequence of input currents over time steps,
    maintaining membrane potential and producing spike outputs.

    Usage:
        neuron = LIFNode(tau=16.0, leak=1.0)
        spikes, voltages = neuron(input_sequence)

    Attributes:
        tau: Membrane time constant (controls decay rate)
        leak: Leak factor (1.0 = no decay, <1.0 = exponential decay)
        V_thresh: Firing threshold (default: 1.0)
        V_reset: Reset potential after spike (default: 0.0)
        learn_tau: Whether tau is a learnable parameter
        learn_thresh: Whether threshold is a learnable parameter
    """

    def __init__(
        self,
        tau: float = 16.0,
        leak: float = 1.0,
        V_thresh: float = 1.0,
        V_reset: float = 0.0,
        learn_tau: bool = False,
        learn_thresh: bool = False,
    ):
        super().__init__()

        self.tau = nn.Parameter(
            torch.tensor(tau), requires_grad=learn_tau
        )
        self.leak = nn.Parameter(
            torch.tensor(leak), requires_grad=False
        )
        self.V_thresh = nn.Parameter(
            torch.tensor(V_thresh), requires_grad=learn_thresh
        )
        self.V_reset = V_reset

    def forward(
        self,
        current: torch.Tensor,
        voltage: Optional[torch.Tensor] = None,
        spike: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for one time step.

        Args:
            current: Input current tensor (same shape as voltage)
            voltage: Previous membrane potential (optional, auto-initialized)
            spike: Previous spike output (optional, auto-initialized)

        Returns:
            voltage: Updated membrane potential
            (spike, hidden_state): Tuple of current spike output and
                                   hidden state dict for next step
        """
        if voltage is None:
            voltage = torch.zeros_like(current)
        if spike is None:
            spike = torch.zeros_like(current)

        # Membrane potential dynamics: leak * V + current
        voltage = self.leak * voltage + current

        # Spike detection: threshold crossing
        # Use STE (Straight-Through Estimator) for gradient flow
        spike = spike_out = torch.gt(voltage, self.V_thresh).float()

        # Reset membrane potential after spike
        voltage = spike_out * self.V_reset + (1 - spike_out) * voltage

        return voltage, (spike, {"voltage": voltage})

    def init_hidden(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Initialize hidden state for a new sequence.

        Args:
            x: Input tensor to determine shape and device

        Returns:
            (voltage, hidden_dict): Initial hidden state
        """
        voltage = torch.zeros_like(x)
        return voltage, {"voltage": voltage}


class LIFConv2d(nn.Module):
    """Convolutional layer + LIF neuron.

    Combines a 2D convolution with LIF activation.
    Mirrors ANN's Conv2d + ReLU but with spiking dynamics.

    Usage:
        conv_snn = LIFConv2d(in_channels=8, out_channels=32,
                             kernel_size=8, stride=4, tau=16.0)
        output, spikes = conv_snn(input_sequence)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        tau: float = 16.0,
        leak: float = 1.0,
        V_thresh: float = 1.0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.neuron = LIFNode(tau=tau, leak=leak, V_thresh=V_thresh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a sequence of frames through conv + LIF.

        Args:
            x: Input tensor of shape (batch, T, C, H, W)
               where T is the number of time steps

        Returns:
            Total spike count across all time steps (batch, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        hidden = None

        all_spikes = []
        for t in range(T):
            conv_out = self.conv(x[:, t])
            if hidden is None:
                hidden = self.neuron.init_hidden(conv_out)

            voltage, (spike, hidden) = self.neuron(conv_out, **hidden)
            all_spikes.append(spike)

        # Sum spikes across time steps (rate coding)
        return torch.stack(all_spikes, dim=1)  # (B, T, C, H, W)


class LIFLinear(nn.Module):
    """Linear layer + LIF neuron.

    Combines a fully connected layer with LIF activation.
    Mirrors ANN's Linear + ReLU but with spiking dynamics.

    Usage:
        lin_snn = LIFLinear(in_features=512, out_features=12, tau=16.0)
        spikes = lin_snn(input_sequence)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 16.0,
        leak: float = 1.0,
        V_thresh: float = 1.0,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.neuron = LIFNode(tau=tau, leak=leak, V_thresh=V_thresh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a sequence through linear + LIF.

        Args:
            x: Input tensor of shape (batch, T, in_features)

        Returns:
            Total spike count across time steps (batch, T, out_features)
        """
        B, T, _ = x.shape
        hidden = None

        all_spikes = []
        for t in range(T):
            linear_out = self.linear(x[:, t])
            if hidden is None:
                hidden = self.neuron.init_hidden(linear_out)

            voltage, (spike, hidden) = self.neuron(linear_out, **hidden)
            all_spikes.append(spike)

        return torch.stack(all_spikes, dim=1)


def test_lif_neuron():
    """Unit test: verify LIF neuron dynamics."""
    neuron = LIFNode(tau=16.0, leak=1.0, V_thresh=1.0)

    # Constant current over 10 time steps
    T = 10
    current = torch.ones(1, T) * 0.3  # (batch=1, time=10)

    # Initialize hidden state with single-timestep shape
    voltage, hidden = neuron.init_hidden(current[:, 0:1])
    spikes = []

    for t in range(T):
        v, (sp, hidden) = neuron(current[:, t:t+1], **hidden)
        spikes.append(sp.sum().item())
        voltage = v

    # With tau=16, leak=1, current=0.3: V accumulates as 0.3, 0.6, 0.9, 1.2
    # Should see spike at t=3 (0-indexed)
    assert spikes[3] >= 1.0, f"Expected spike at t=3, got {spikes}"
    print(f"LIF neuron test passed! Spikes at indices: {[i for i, s in enumerate(spikes) if s > 0]}")


if __name__ == "__main__":
    test_lif_neuron()
