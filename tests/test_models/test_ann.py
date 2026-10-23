"""Unit tests for DoomNet ANN architecture.

Tests output shapes, forward pass, and action value computation.
"""

import pytest
import torch
import torch.nn.functional as F

from models.ann.ppo_network import DoomNet


class TestDoomNet:
    """Test DoomNet model."""

    @pytest.fixture
    def model(self):
        """Create a DoomNet instance with test parameters."""
        return DoomNet(input_channels=8, n_actions=12, hidden_dim=512)

    def test_output_shapes(self, model):
        """DoomNet produces correct output shapes."""
        batch_size = 2
        x = torch.randn(batch_size, 8, 84, 84)
        policy_logits, values = model(x)
        assert policy_logits.shape == (batch_size, 12)
        assert values.shape == (batch_size, 1)

    def test_action_value_computation(self, model):
        """get_action_and_value returns correct shapes and valid probs."""
        batch_size = 4
        x = torch.randn(batch_size, 8, 84, 84)
        actions = torch.randint(0, 12, (batch_size,))

        result = model.get_action_and_value(x, actions)
        log_probs, values, entropy = result

        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size, 1)
        assert entropy.shape == (batch_size,)
        # Entropy must be non-negative for a valid distribution
        assert (entropy >= 0).all()
        # Log probs must be negative (log of [0, 1])
        assert (log_probs <= 0).all()

    def test_no_action_forward(self, model):
        """Forward without action returns logits and values."""
        batch_size = 1
        x = torch.randn(batch_size, 8, 84, 84)
        policy_logits, values = model(x)
        assert policy_logits.shape == (batch_size, 12)
        assert values.shape == (batch_size, 1)

    def test_parameter_count(self, model):
        """DoomNet has approximately 1.2M parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 1_000_000
        assert total_params < 2_000_000

    def test_gradient_flow(self, model):
        """Gradients flow through the model."""
        x = torch.randn(1, 8, 84, 84)
        policy_logits, values = model(x)
        loss = policy_logits.sum()
        loss.backward()
        # Check that at least some parameters have gradients
        has_grad = sum(1 for p in model.parameters() if p.grad is not None)
        assert has_grad > 0
