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


class SurrogateGradient(torch.autograd.Function):
    """Surrogate gradient for the non-differentiable spike function.

    Uses a sigmoid approximation of the derivative of the Heaviside step
    function. Gradients flow through the threshold during backpropagation
    only when the voltage is near the threshold (where the sigmoid is
    steepest).

    Reference:
        Zenke, F. & Ganguli, S. (2018). Superspike: Supervised learning
        in spiking neural networks with surrogate gradients.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Forward: hard threshold (spike if x > 0)."""
        ctx.save_for_backward(x, torch.tensor(alpha))
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward: sigmoid surrogate gradient scaled by alpha."""
        x, alpha = ctx.saved_tensors
        sig_alpha = alpha
        surrogate_grad = sig_alpha * torch.sigmoid(sig_alpha * x)
        return surrogate_grad * grad_output, None


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
        surrogate_alpha: float = 5.0,
        learn_tau: bool = False,
        learn_thresh: bool = False,
    ):
        """Initialize LIF neuron.

        Args:
            tau: Membrane time constant (controls decay rate)
            leak: Leak factor (1.0 = no decay, <1.0 = exponential decay)
            V_thresh: Firing threshold
            V_reset: Reset potential after spike
            surrogate_alpha: Steepness of the surrogate gradient. Higher
                values produce a sharper approximation of the step
                function's derivative (default: 5.0).
            learn_tau: Whether tau is a learnable parameter
            learn_thresh: Whether threshold is a learnable parameter
        """
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
        self.surrogate_alpha = surrogate_alpha

    def forward(
        self,
        current: torch.Tensor,
        voltage: Optional[torch.Tensor] = None,
        spike: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for one time step.

        Uses a surrogate gradient to allow backpropagation through the
        threshold-crossing operation. Gradients flow only when the
        membrane potential is near the firing threshold.

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

        # Spike detection with surrogate gradient for backprop
        # Forward pass: hard threshold (spike if voltage >= threshold)
        # Backward pass: smooth sigmoid approximation of the derivative
        spike = spike_out = SurrogateGradient.apply(
            voltage - self.V_thresh, self.surrogate_alpha
        ).float()

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
        surrogate_alpha: float = 5.0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.neuron = LIFNode(
            tau=tau, leak=leak, V_thresh=V_thresh,
            surrogate_alpha=surrogate_alpha,
        )

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
        surrogate_alpha: float = 5.0,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.neuron = LIFNode(
            tau=tau, leak=leak, V_thresh=V_thresh,
            surrogate_alpha=surrogate_alpha,
        )

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


def test_surrogate_gradient():
    """Unit test: verify gradients flow through LIF neuron."""
    neuron = LIFNode(tau=1.0, leak=1.0, V_thresh=1.0, surrogate_alpha=5.0)

    # Create current that will cause voltage to cross threshold
    current = torch.ones(1, 1) * 1.5  # (batch=1) — above threshold
    current.requires_grad_(True)

    voltage, (spike, _) = neuron(current)

    # Backward pass — should produce non-zero gradient
    voltage.sum().backward()

    assert current.grad is not None, "Gradient should not be None"
    assert not torch.all(current.grad == 0), (
        f"Surrogate gradient should flow through LIF neuron, "
        f"got all-zero gradient: {current.grad}"
    )
    # Gradients should be non-zero where voltage is near threshold
    assert current.grad.abs().sum() > 0, "Gradient sum should be positive"
    print("Surrogate gradient test passed! "
          f"Gradient mean: {current.grad.mean():.4f}")


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
    test_surrogate_gradient()
    test_lif_neuron()
