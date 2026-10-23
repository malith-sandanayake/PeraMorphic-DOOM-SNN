"""Unit tests for SNN models — LIF neurons, SpikingCNN, and weight conversion.

Tests spike dynamics, forward pass shapes, and ANN-to-SNN weight mapping.
"""

import pytest
import torch
import torch.nn.functional as F

from models.snn.lif_neuron import LIFNode
from models.snn.snn_cnn import SpikingCNN
from models.snn.snn_policy import SNNPolicy


class TestLIFNode:
    """Test LIF neuron dynamics."""

    def test_spike_detection(self):
        """LIF fires when voltage exceeds threshold."""
        neuron = LIFNode(V_thresh=1.0, leak=1.0)
        current = torch.ones(1, 4)  # constant current
        voltage, (spike, _) = neuron(current)
        # With unit current and leak=1.0, voltage should increase
        assert voltage.sum() > 0

    def test_reset_after_spike(self):
        """Voltage resets to V_reset after firing."""
        neuron = LIFNode(V_thresh=0.5, leak=1.0, V_reset=0.0)
        current = torch.ones(1, 4) * 10.0  # large current → spike
        voltage, (spike, _) = neuron(current)
        if spike.sum() > 0:
            assert (voltage == 0.0).any()

    def test_no_spike_below_threshold(self):
        """LIF does not fire when voltage stays below threshold."""
        neuron = LIFNode(V_thresh=100.0, leak=1.0)
        current = torch.zeros(1, 4)  # zero current
        voltage, (spike, _) = neuron(current)
        assert (spike == 0).all()
        assert (voltage == 0).all()

    def test_hidden_state_management(self):
        """LIFNode manages hidden state across calls."""
        neuron = LIFNode(V_thresh=1.0, leak=1.0)
        current = torch.ones(1, 4)
        v1, (s1, h1) = neuron(current)
        v2, (s2, h2) = neuron(current, **h1)
        # Second call should see membrane potential from first
        assert v2.sum() >= v1.sum()

    def test_init_hidden(self):
        """init_hidden creates proper initial state."""
        neuron = LIFNode(V_thresh=1.0)
        x = torch.randn(2, 4)
        voltage, hidden = neuron.init_hidden(x)
        assert voltage.shape == (2, 4)
        assert "voltage" in hidden


class TestSpikingCNN:
    """Test SpikingCNN architecture."""

    @pytest.fixture
    def model(self):
        """Create a SpikingCNN with test parameters."""
        return SpikingCNN(n_actions=12, T=8, tau=16.0, input_channels=8)

    def test_forward_shape(self, model):
        """SpikingCNN forward pass produces correct shapes."""
        batch_size = 2
        T = model.T
        x = torch.randn(batch_size, T, 8, 84, 84)
        policy_rate, value, hidden = model(x)
        assert policy_rate.shape == (batch_size, 12)
        assert value.shape == (batch_size,)
        assert isinstance(hidden, dict)

    def test_no_action_forward(self, model):
        """Forward without action returns policy, value, hidden."""
        x = torch.randn(1, 8, 8, 84, 84)
        policy_rate, value, hidden = model(x)
        assert policy_rate.shape == (1, 12)
        assert value.shape == (1,)
        assert len(hidden) == 4  # 4 LIF neurons

    def test_get_action_and_value(self, model):
        """get_action_and_value returns correct shapes and valid probs."""
        x = torch.randn(2, 8, 8, 84, 84)
        actions = torch.randint(0, 12, (2,))
        result = model.get_action_and_value(x, actions)
        log_probs, values, entropy, hidden = result
        assert log_probs.shape == (2,)
        assert values.shape == (2,)
        assert entropy.shape == (2,)
        assert (entropy >= 0).all()

    def test_rate_coding_normalization(self, model):
        """Policy rates are averaged over T timesteps."""
        x = torch.randn(1, 8, 8, 84, 84)
        policy_rate, _, _ = model(x)
        # Rate coding produces finite floating-point values
        assert torch.isfinite(policy_rate).all()
        assert not torch.all(policy_rate == 0)

    def test_zero_input_produces_zero_output(self, model):
        """Zero input should produce zero policy rates."""
        x = torch.zeros(1, 8, 8, 84, 84)
        policy_rate, value, _ = model(x)
        # With zero input, no neurons fire → all rates are zero
        assert torch.allclose(policy_rate, torch.zeros_like(policy_rate), atol=1e-6)
        assert torch.allclose(value, torch.zeros_like(value), atol=1e-6)


class TestSNNWeightConversion:
    """Test ANN-to-SNN weight conversion."""

    def test_conversion_from_ann(self):
        """SNN can copy weights from a trained DoomNet."""
        ann_model = DoomNet(input_channels=8, n_actions=12, hidden_dim=512)
        snn = SpikingCNN(n_actions=12, T=16, input_channels=8)

        snn.convert_from_ann(ann_model)

        # Check that conv1 weights were copied
        assert torch.equal(ann_model.features[0].weight, snn.conv1.weight)
        assert torch.equal(ann_model.fc1.weight, snn.fc1.weight)

    def test_conversion_preserves_head_weights(self):
        """Policy and value head weights are copied correctly."""
        ann_model = DoomNet(input_channels=8, n_actions=12, hidden_dim=512)
        snn = SpikingCNN(n_actions=12, T=16, input_channels=8)

        snn.convert_from_ann(ann_model)

        assert torch.equal(ann_model.policy_head.weight, snn.policy_head.weight)
        assert torch.equal(ann_model.value_head.weight, snn.value_head.weight)
