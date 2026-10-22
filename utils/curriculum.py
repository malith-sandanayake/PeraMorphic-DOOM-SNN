"""
Curriculum learning utility for scenario progression.

Manages progressive training across increasingly difficult DOOM scenarios,
loading previous checkpoints as initialization.

Usage:
    from utils.curriculum import CurriculumTrainer
    trainer = CurriculumTrainer()
    for scenario, timesteps, checkpoint in trainer.run():
        train_on_scenario(scenario, timesteps, checkpoint)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class CurriculumStep:
    """A single step in the curriculum.

    Attributes:
        scenario: Scenario name (e.g., "simple", "deadly_corridor")
        timesteps: Number of training timesteps
        description: Human-readable description
    """
    scenario: str
    timesteps: int
    description: str


# Default curriculum: progressively harder scenarios
DEFAULT_CURRICULUM = [
    CurriculumStep(
        scenario="simple",
        timesteps=5_000_000,
        description="Learn basic movement and shooting mechanics",
    ),
    CurriculumStep(
        scenario="deadly_corridor",
        timesteps=10_000_000,
        description="Navigate through corridors while eliminating enemies",
    ),
    CurriculumStep(
        scenario="deathmatch",
        timesteps=20_000_000,
        description="Full combat: navigate, shoot, and survive",
    ),
]


class CurriculumTrainer:
    """Manages curriculum learning across DOOM scenarios.

    Tracks training progress and handles checkpoint loading between
    curriculum steps.

    Usage:
        curriculum = CurriculumTrainer(scenarios=DEFAULT_CURRICULUM)
        for step in curriculum:
            print(f"Training on {step.scenario}: {step.timesteps} steps")
            checkpoint = train(step)
            curriculum.save_checkpoint(step.scenario, checkpoint)
    """

    def __init__(
        self,
        scenarios: Optional[List[CurriculumStep]] = None,
        checkpoint_dir: str = "models",
    ):
        """Initialize curriculum trainer.

        Args:
            scenarios: List of curriculum steps (uses default if None)
            checkpoint_dir: Directory to save/load checkpoints
        """
        self.scenarios = scenarios or DEFAULT_CURRICULUM
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._current_step = 0
        self._completed_steps: List[str] = []

    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self):
        for i, scenario in enumerate(self.scenarios):
            self._current_step = i
            yield scenario

    def get_current_scenario(self) -> CurriculumStep:
        """Get the current scenario in the curriculum.

        Returns:
            Current CurriculumStep
        """
        return self.scenarios[self._current_step]

    def get_checkpoint_path(self, scenario: str) -> Path:
        """Get the checkpoint file path for a scenario.

        Args:
            scenario: Scenario name

        Returns:
            Path to the checkpoint file
        """
        return self.checkpoint_dir / f"ann_{scenario}"

    def get_previous_checkpoint(self) -> Optional[Path]:
        """Get the checkpoint from the previous curriculum step.

        Returns:
            Path to previous checkpoint, or None if first step
        """
        if not self._completed_steps:
            return None
        last_scenario = self._completed_steps[-1]
        checkpoint = self.get_checkpoint_path(last_scenario)
        return checkpoint if checkpoint.exists() else None

    def mark_completed(self, scenario: str) -> None:
        """Mark a scenario as completed.

        Args:
            scenario: Completed scenario name
        """
        if scenario not in self._completed_steps:
            self._completed_steps.append(scenario)

    def next_step(self) -> bool:
        """Advance to the next curriculum step.

        Returns:
            True if there is a next step, False if curriculum is complete
        """
        if self._current_step < len(self.scenarios) - 1:
            self._current_step += 1
            return True
        return False

    def progress_report(self) -> str:
        """Generate a progress report string.

        Returns:
            Formatted progress report
        """
        lines = ["=== Curriculum Progress ==="]
        for i, step in enumerate(self.scenarios):
            status = "DONE" if step.scenario in self._completed_steps else (
                "CURRENT" if i == self._current_step else "UPCOMING"
            )
            lines.append(f"  [{status}] {step.scenario}: {step.timesteps:,} steps")
            lines.append(f"         {step.description}")
        return "\n".join(lines)
