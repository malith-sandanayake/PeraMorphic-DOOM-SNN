"""
ViZDoom Gymnasium Environment Wrapper

Provides a clean gym.Env interface to ViZDoom for reinforcement learning.
Supports multiple DOOM scenarios with configurable observations and actions.

Usage:
    from env.vizdoom_env import DoomEnv
    env = DoomEnv(scenario="simple")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import os
import numpy as np
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

try:
    import vizdoom as vd
except ImportError:
    raise ImportError(
        "ViZDoom Python bindings not installed. "
        "Run: pip install vizdoom"
    )

# Scenario configuration paths
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "doom_cfg")

# Default ViZDoom actions (12-action set)
DEFAULT_ACTIONS = [
    [vd.Action.MOVE_FORWARD],
    [vd.Action.MOVE_BACKWARD],
    [vd.Action.STRAFE_LEFT],
    [vd.Action.STRAFE_RIGHT],
    [vd.Action.TURN_LEFT],
    [vd.Action.TURN_RIGHT],
    [vd.Action.ATTACK],
    [vd.Action.MOVE_FORWARD, vd.Action.TURN_LEFT],
    [vd.Action.MOVE_FORWARD, vd.Action.TURN_RIGHT],
    [vd.Action.MOVE_FORWARD, vd.Action.ATTACK],
    [vd.Action.MOVE_FORWARD, vd.Action.STRAFE_LEFT],
    [vd.Action.MOVE_FORWARD, vd.Action.STRAFE_RIGHT],
]

# Extended actions for Deathmatch (includes backward variants)
DEATHMATCH_ACTIONS = [
    [vd.Action.MOVE_FORWARD],
    [vd.Action.MOVE_BACKWARD],
    [vd.Action.STRAFE_LEFT],
    [vd.Action.STRAFE_RIGHT],
    [vd.Action.TURN_LEFT],
    [vd.Action.TURN_RIGHT],
    [vd.Action.ATTACK],
    [vd.Action.MOVE_FORWARD, vd.Action.TURN_LEFT],
    [vd.Action.MOVE_FORWARD, vd.Action.TURN_RIGHT],
    [vd.Action.MOVE_FORWARD, vd.Action.ATTACK],
    [vd.Action.MOVE_FORWARD, vd.Action.STRAFE_LEFT],
    [vd.Action.MOVE_FORWARD, vd.Action.STRAFE_RIGHT],
    [vd.Action.MOVE_BACKWARD, vd.Action.TURN_LEFT],
    [vd.Action.MOVE_BACKWARD, vd.Action.TURN_RIGHT],
    [vd.Action.MOVE_BACKWARD, vd.Action.ATTACK],
    [vd.Action.MOVE_BACKWARD, vd.Action.STRAFE_LEFT],
    [vd.Action.MOVE_BACKWARD, vd.Action.STRAFE_RIGHT],
]


class DoomEnv(gym.Env):
    """Gymnasium environment wrapping ViZDoom for DOOM AI research.

    Observations:
        - Screen buffer: RGB image (default 64x48, configurable)
        - Depth buffer: Grayscale depth map (optional)
        - Player info: Vector of game state variables

    Actions:
        - Discrete action space (4, 12, or 18 actions depending on scenario)

    Args:
        scenario: Name of the scenario ("simple", "deadly_corridor", "deathmatch")
        screen_size: Tuple of (width, height) for screen observations
        frame_skip: Number of environment steps per RL step (1 = no skipping)
        use_depth: Whether to include depth buffer in observations
        use_hud: Whether to include HUD elements in observations
        seed: Random seed for reproducibility
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 35}

    def __init__(
        self,
        scenario: str = "simple",
        screen_size: Tuple[int, int] = (64, 48),
        frame_skip: int = 1,
        use_depth: bool = False,
        use_hud: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.scenario = scenario
        self.screen_size = screen_size
        self.frame_skip = frame_skip
        self.use_depth = use_depth
        self.use_hud = use_hud

        # Load ViZDoom config
        cfg_path = os.path.join(CONFIG_DIR, f"{scenario}.cfg")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"Scenario config not found: {cfg_path}\n"
                f"Ensure ViZDoom scenario configs are in config/doom_cfg/"
            )

        self._game = vd.DoomGame()
        self._game.load_config(cfg_path)
        self._game.set_window_visible(False)
        self._game.set_mode(vd.Mode.ASYNC_PLAYER if use_hud else vd.Mode.ASYNC_SPARTAN)

        # Set action set
        if scenario == "deathmatch":
            self.actions = DEATHMATCH_ACTIONS
        else:
            self.actions = DEFAULT_ACTIONS

        self._game.set_available_buttons(self.actions)

        # Set observations
        self._game.set_screen_format(vd.ScreenFormat.RGB24)
        self._game.set_screen_resolution(vd.ScreenResolution.RES_64X48)

        if use_depth:
            self._game.set_depth_enabled(True)

        if use_hud:
            self._game.set_hud_enabled(True)
            self._game.set_available_hud_variables(
                [
                    vd.HudVariable.HEALTH,
                    vd.HudVariable.ARMOR,
                    vd.HudVariable.SELECTED_WEAPON,
                    vd.HudVariable.SELECTED_WEAPON_AMMO,
                    vd.HudVariable.KILLS,
                    vd.HudVariable.DEATHS,
                ]
            )

        # Initialize the game
        self._game.init()

        # Action space (discrete)
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space
        screen_channels = 3  # RGB
        if use_depth:
            screen_channels = 4  # RGB + depth
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(screen_size[1], screen_size[0], screen_channels),
            dtype=np.uint8,
        )

        # Frame stack buffer (for temporal context)
        self._frame_stack_size = 4
        self._frame_stack = np.zeros(
            (self._frame_stack_size, screen_size[1], screen_size[0], 3),
            dtype=np.uint8,
        )

        # Previous player info for reward computation
        self._prev_player_info = None

        # Episode stats
        self._episode_steps = 0
        self._total_reward = 0.0

        if seed is not None:
            self.seed(seed)

    def seed(self, seed: Optional[int] = None) -> list:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value

        Returns:
            List containing the seed that was set
        """
        self._game.set_seed(seed)
        return [seed]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the initial state.

        Args:
            seed: Optional seed for this reset (overrides __init__ seed)
            options: Additional reset options (not used currently)

        Returns:
            observation: Preprocessed observation (stacked frames)
            info: Dictionary with additional environment information
        """
        if seed is not None:
            self._game.set_seed(seed)

        self._game.new_episode()

        # Get initial observation
        screen = self._game.get_screen_buffer()
        depth = self._game.get_depth_buffer()
        player_info = self._game.get_player_stats()

        # Stack initial frames (duplicate first frame)
        if screen is not None:
            screen = self._preprocess_frame(screen)
            self._frame_stack = np.stack([screen] * self._frame_stack_size, axis=0)

        self._prev_player_info = player_info
        self._episode_steps = 0
        self._total_reward = 0.0

        info = {
            "kills": 0,
            "deaths": 0,
            "health": 0,
            "armor": 0,
            "frags": 0,
            "episode_steps": 0,
        }

        return self._frame_stack.copy(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Integer action index (0 to num_actions-1)

        Returns:
            observation: Preprocessed observation (stacked frames)
            reward: Computed reward from ViZDoom + delta reward
            terminated: Whether the episode ended (death or goal reached)
            truncated: Whether the episode was truncated (timeout)
            info: Dictionary with additional step information
        """
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.frame_skip):
            # Execute action in ViZDoom
            reward = self._game.make_action(self.actions[action])
            total_reward += reward

            # Check if episode is finished
            if self._game.is_episode_finished():
                terminated = True
                break

            # Get new observation
            screen = self._game.get_screen_buffer()
            if screen is not None:
                screen = self._preprocess_frame(screen)
                # Shift frame stack
                self._frame_stack = np.roll(self._frame_stack, -1, axis=0)
                self._frame_stack[-1] = screen

        # Get player stats for info and delta reward
        player_info = self._game.get_player_stats()

        if player_info is not None and self._prev_player_info is not None:
            # Compute delta reward
            delta_reward = self._compute_delta_reward(self._prev_player_info, player_info)
            total_reward += delta_reward

            self._prev_player_info = player_info

        self._episode_steps += 1
        self._total_reward += total_reward

        # Build info dict
        info = {
            "kills": getattr(player_info, "kills", 0) if player_info else 0,
            "deaths": getattr(player_info, "death_count", 0) if player_info else 0,
            "health": getattr(player_info, "health", 0) if player_info else 0,
            "armor": getattr(player_info, "armor", 0) if player_info else 0,
            "frags": getattr(player_info, "frags", 0) if player_info else 0,
            "episode_steps": self._episode_steps,
            "raw_reward": total_reward,
        }

        # Truncate if episode exceeds max steps
        if self._episode_steps >= 2100:
            truncated = True

        return self._frame_stack.copy(), total_reward, terminated, truncated, info

    def get_player_stats(self) -> Any:
        """Get current player statistics from ViZDoom.

        Returns:
            DoomPlayerStats object from ViZDoom
        """
        return self._game.get_player_stats()

    def render(self) -> Optional[np.ndarray]:
        """Render the current frame for visualization.

        Returns:
            RGB array of the current screen, or None if window is hidden
        """
        screen = self._game.get_screen_buffer()
        if screen is not None:
            return screen
        return None

    def close(self):
        """Close the ViZDoom game instance."""
        if self._game.is_running():
            self._game.shutdown()

    def __del__(self):
        """Ensure game is closed on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    # ---- Private Methods ----

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame from ViZDoom.

        Args:
            frame: Raw RGB frame from ViZDoom

        Returns:
            Preprocessed frame (uint8, same shape as input)
        """
        # Convert to numpy if it's a DoomImage
        if hasattr(frame, "to_ndarray"):
            frame = frame.to_ndarray()

        # Ensure correct shape
        if frame.ndim == 3:
            return frame.astype(np.uint8)

        return frame.astype(np.uint8)

    def _compute_delta_reward(
        self,
        prev: Any,
        current: Any,
    ) -> float:
        """Compute delta-based reward from changes in player statistics.

        This provides dense reward shaping beyond ViZDoom's built-in rewards.

        Args:
            prev: Previous player info (DoomPlayerStats)
            current: Current player info (DoomPlayerStats)

        Returns:
            Delta reward value
        """
        delta_health = getattr(current, "health", 0) - getattr(prev, "health", 0)
        delta_armor = getattr(current, "armor", 0) - getattr(prev, "armor", 0)
        delta_kills = getattr(current, "kills", 0) - getattr(prev, "kills", 0)
        died = 1 if (
            getattr(current, "death_count", 0) > getattr(prev, "death_count", 0)
        ) else 0

        return (
            +1.0 * delta_health
            + 0.1 * delta_armor
            + 10.0 * delta_kills
            - 100.0 * died
        )


class DoomStackedEnv(gym.ObservationWrapper):
    """Wrapper that stacks multiple frames for temporal context.

    Stacks 4 consecutive observations into a single observation,
    providing the agent with motion information without explicit
    recurrent connections.

    Usage:
        env = DoomStackedEnv(DoomEnv(scenario="simple"))
        # obs shape: (4, H, W, C) — 4-frame stack
    """

    def __init__(self, env: DoomEnv, stack_size: int = 4):
        super().__init__(env)
        self.stack_size = stack_size
        h, w = env.screen_size
        c = env.observation_space.shape[-1]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(stack_size * h, w, c),
            dtype=np.uint8,
        )
        self._stack = np.zeros((stack_size, h, w, c), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Stack the latest frame into the frame buffer.

        Args:
            observation: Single-frame observation (H, W, C)

        Returns:
            Stacked observation ((stack_size * H), W, C) — vertical stack for CNN
        """
        self._stack = np.roll(self._stack, -1, axis=0)
        self._stack[-1] = observation
        # Vertical stack for compatibility with CNN feature extractors
        return self._stack.reshape(self.stack_size * self._stack.shape[1],
                                    self._stack.shape[2], self._stack.shape[3])
