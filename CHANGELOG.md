# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with complete SNN pipeline
- ViZDoom Gymnasium environment wrapper (`env/vizdoom_env.py`)
- DoomNet ANN architecture (`models/ann/ppo_network.py`)
- Spiking CNN with LIF neurons (`models/snn/`)
- PPO training pipeline via Stable-Baselines3
- SNN fine-tuning with surrogate gradients
- ANN-to-SNN conversion pipeline
- Curriculum learning utility
- Evaluation and visualization utilities
- Scenario configs: Simple, DeadlyCorridor, Deathmatch
- Comprehensive documentation in `docs/`

### Changed
- None

### Fixed
- None

### Removed
- None

---

## Version 0.1.0 (2026-10-22)

### Added
- Project skeleton with modular structure
- Core environment implementation:
  - `DoomEnv` — Gymnasium wrapper for ViZDoom
  - `DoomStackedEnv` — Frame stacking wrapper
  - 12-action and 17-action sets for DOOM
  - Delta-based reward shaping
- Model implementations:
  - `DoomNet` — 3-layer CNN policy network (1.2M params)
  - `LIFNode` — Leaky Integrate-and-Fire neuron
  - `LIFConv2d` / `LIFLinear` — Spiking layer wrappers
  - `SpikingCNN` — Full spiking policy network
  - `SNNPolicy` — Policy wrapper with action selection
- Training code:
  - `ProgressCallback` — Training progress logging
  - `EvaluationCallback` — Periodic model evaluation
  - `SNNPPOTrainer` — SNN fine-tuning with surrogate gradients
- Evaluation utilities:
  - `evaluate_ann_agent()` — PPO agent evaluation
  - `evaluate_snn_agent()` — SNN agent evaluation with spike stats
  - `FrameRecorder` — Gameplay frame recording
  - `plot_firing_rates()` / `plot_training_progress()` — Visualization
- Utility modules:
  - `RewardConfig` — Configurable reward shaping
  - `FrameProcessor` — Frame preprocessing pipeline
  - `CurriculumTrainer` — Scenario progression manager
- Configuration files:
  - 3 ViZDoom scenario configs
  - 4 YAML training config files
- CLI entry point (`main.py`) with 6 modes
- Comprehensive documentation:
  - `README.md` — Project overview and quick start
  - `docs/ARCHITECTURE.md` — System architecture and data flow
  - `docs/API.md` — Complete API reference
  - `docs/SETUP.md` — Installation and troubleshooting
  - `docs/TRAINING.md` — Training, evaluation, and tuning guide
  - `docs/CONTRIBUTING.md` — Contribution guidelines
  - `CHANGELOG.md` — Version history
  - `research/notes.md` — Research tracking template
- Test suite:
  - DoomNet forward pass test
  - LIF neuron dynamics test
  - SpikingCNN shape verification
  - Weight conversion test
  - All unit tests passing

### Security
- None
