"""
Reward function utilities for DOOM reinforcement learning.

Provides reward computation and shaping functions for ViZDoom scenarios.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    """Configuration for reward shaping.

    Attributes:
        health_gain_weight: Weight for health increase (default: 1.0)
        health_loss_weight: Weight for health decrease (default: 2.0)
        armor_gain_weight: Weight for armor increase (default: 0.1)
        kill_reward: Reward for enemy kill (default: 10.0)
        death_penalty: Penalty for agent death (default: -100.0)
        survival_reward: Per-step survival bonus (default: 0.01)
        damage_penalty: Penalty per health point lost (default: 0.5)
        time_penalty: Per-step time penalty to encourage speed (default: 0.0)
        weapon_change_penalty: Penalty for switching weapons (default: 0.0)
    """
    health_gain_weight: float = 1.0
    health_loss_weight: float = 2.0
    armor_gain_weight: float = 0.1
    kill_reward: float = 10.0
    death_penalty: float = -100.0
    survival_reward: float = 0.01
    damage_penalty: float = 0.5
    time_penalty: float = 0.0
    weapon_change_penalty: float = 0.0

    def compute_reward(
        self,
        prev_health: float,
        current_health: float,
        prev_armor: float,
        current_armor: float,
        prev_kills: int,
        current_kills: int,
        prev_weapon: int,
        current_weapon: int,
        died: bool,
        episode_step: int,
    ) -> float:
        """Compute shaped reward from delta in player statistics.

        Args:
            prev_health: Health value from previous step
            current_health: Current health value
            prev_armor: Armor value from previous step
            current_armor: Current armor value
            prev_kills: Kill count from previous step
            current_kills: Current kill count
            prev_weapon: Weapon index from previous step
            current_weapon: Current weapon index
            died: Whether the agent died this step
            episode_step: Current step number (for survival reward)

        Returns:
            Composed reward value
        """
        reward = 0.0

        # Health changes
        health_delta = current_health - prev_health
        if health_delta > 0:
            reward += self.health_gain_weight * health_delta
        elif health_delta < 0:
            reward -= self.health_loss_weight * abs(health_delta)

        # Armor changes
        armor_delta = current_armor - prev_armor
        if armor_delta > 0:
            reward += self.armor_gain_weight * armor_delta

        # Kills
        kill_delta = current_kills - prev_kills
        if kill_delta > 0:
            reward += self.kill_reward * kill_delta

        # Death
        if died:
            reward += self.death_penalty

        # Survival bonus (small per-step incentive to stay alive)
        reward += self.survival_reward

        # Time penalty (encourages speed)
        if self.time_penalty > 0:
            reward -= self.time_penalty

        # Weapon change penalty (discourage unnecessary switching)
        if (prev_weapon != current_weapon and prev_weapon >= 0 and current_weapon >= 0
                and prev_weapon != -1 and current_weapon != -1):
            reward -= self.weapon_change_penalty

        return reward


def simple_reward(
    prev_health: int,
    current_health: int,
    prev_kills: int,
    current_kills: int,
    died: bool,
) -> float:
    """Minimal reward function for quick testing.

    Args:
        prev_health: Previous health
        current_health: Current health
        prev_kills: Previous kill count
        current_kills: Current kill count
        died: Whether agent died

    Returns:
        Simple composed reward
    """
    reward = 0.0

    if current_kills > prev_kills:
        reward += 10.0
    if died:
        reward -= 100.0
    if current_health > prev_health:
        reward += 1.0

    return reward
