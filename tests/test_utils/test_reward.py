"""Unit tests for reward shaping utilities.

Tests RewardConfig.compute_reward and simple_reward correctness.
"""

import pytest

from utils.reward import RewardConfig, simple_reward


class TestRewardConfig:
    """Test RewardConfig reward computation."""

    @pytest.fixture
    def default_config(self):
        """RewardConfig with default weights."""
        return RewardConfig()

    def test_kill_reward(self, default_config):
        """Reward config gives positive reward for kills."""
        reward = default_config.compute_reward(
            prev_health=100, current_health=100,
            prev_armor=0, current_armor=0,
            prev_kills=0, current_kills=1,
            prev_weapon=1, current_weapon=1,
            died=False, episode_step=10,
        )
        assert reward > 0
        assert reward >= 10.0  # at minimum the kill_reward

    def test_death_penalty(self, default_config):
        """Reward config applies death penalty."""
        reward = default_config.compute_reward(
            prev_health=100, current_health=0,
            prev_armor=0, current_armor=0,
            prev_kills=0, current_kills=0,
            prev_weapon=1, current_weapon=1,
            died=True, episode_step=10,
        )
        assert reward < 0
        # Death penalty should be at least -100 + survival bonus
        assert reward <= -99.0

    def test_health_gain(self, default_config):
        """Reward config gives positive reward for health gain."""
        reward = default_config.compute_reward(
            prev_health=50, current_health=75,
            prev_armor=0, current_armor=0,
            prev_kills=0, current_kills=0,
            prev_weapon=1, current_weapon=1,
            died=False, episode_step=10,
        )
        assert reward > 0
        # Should include health gain (1.0 * 25 = 25) + survival bonus
        assert reward >= 25.0

    def test_health_loss(self, default_config):
        """Reward config penalizes health loss."""
        reward = default_config.compute_reward(
            prev_health=100, current_health=75,
            prev_armor=0, current_armor=0,
            prev_kills=0, current_kills=0,
            prev_weapon=1, current_weapon=1,
            died=False, episode_step=10,
        )
        assert reward < 0

    def test_survival_bonus(self, default_config):
        """Reward config gives small positive survival bonus."""
        # No changes — pure survival
        reward = default_config.compute_reward(
            prev_health=100, current_health=100,
            prev_armor=0, current_armor=0,
            prev_kills=0, current_kills=0,
            prev_weapon=1, current_weapon=1,
            died=False, episode_step=10,
        )
        assert reward > 0  # only survival bonus (0.01)

    def test_custom_weights(self):
        """RewardConfig respects custom weights."""
        config = RewardConfig(
            kill_reward=5.0,
            death_penalty=-50.0,
            survival_reward=0.1,
        )
        # Kill should use custom weight
        reward = config.compute_reward(
            prev_health=100, current_health=100,
            prev_armor=0, current_armor=0,
            prev_kills=0, current_kills=1,
            prev_weapon=1, current_weapon=1,
            died=False, episode_step=10,
        )
        assert reward >= 5.0

    def test_armor_gain(self, default_config):
        """Reward config gives positive reward for armor gain."""
        reward = default_config.compute_reward(
            prev_health=100, current_health=100,
            prev_armor=0, current_armor=25,
            prev_kills=0, current_kills=0,
            prev_weapon=1, current_weapon=1,
            died=False, episode_step=10,
        )
        assert reward > 0
        assert reward >= 0.01  # armor_gain (0.1 * 25) + survival
