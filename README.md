# PeraMorphic-DOOM-SNN

<div align="center">

**Autonomous DOOM gameplay simulation using Spiking Neural Networks (SNNs), targeting FPGA neuromorphic accelerator deployment via PeraMorphIQ.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![ViZDoom](https://img.shields.io/badge/ViZDoom-1.2.4+-green.svg)](https://vizdoom.github.io/)
[![snnTorch](https://img.shields.io/badge/snnTorch-0.8%2B-ff69b4.svg)](https://snntorch.readthedocs.io/)

</div>

## Overview

This project builds a Python pipeline that trains a Spiking Neural Network (SNN) to play DOOM autonomously. The SNN will eventually be deployed on an FPGA-based neuromorphic accelerator (PeraMorphIQ).

### The Pipeline

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. Train ANN    │────►│  2. Convert      │────►│  3. Fine-tune    │────►│  4. Export to    │
│  PPO Baseline    │     │  ANN → SNN       │     │  SNN via         │     │  FPGA            │
│  (ViZDoom + SB3) │     │  (Weight Copy)   │     │  Surrogate Grads │     │  (Model Spec)  │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

1. **ANN Baseline**: Train a standard neural network (DoomNet) with PPO on ViZDoom
2. **SNN Conversion**: Convert the trained ANN to a rate-coded SNN by replacing activations with LIF neurons
3. **SNN Fine-tuning**: Train the SNN with surrogate gradients to recover performance
4. **Hardware Export**: Generate FPGA-compatible model specifications

## Quick Start

```bash
# 1. Install dependencies (see docs/SETUP.md)
pip install -r requirements.txt

# 2. Verify setup with random agent
python main.py --mode test-random --scenario simple

# 3. Train ANN baseline
python main.py --mode train-ann --scenario simple

# 4. Evaluate trained agent
python main.py --mode evaluate --model models/ann_simple --episodes 100

# 5. Convert to SNN
python main.py --mode convert --checkpoint models/ann_simple

# 6. Fine-tune SNN
python main.py --mode train-snn --checkpoint models/snn_converted
```

## Documentation

| Document | Description |
|----------|-------------|
| [Setup Guide](docs/SETUP.md) | Installation instructions, environment setup, troubleshooting |
| [Architecture](docs/ARCHITECTURE.md) | System architecture, data flow, design decisions |
| [API Reference](docs/API.md) | Complete API reference for all public classes and functions |
| [Training Guide](docs/TRAINING.md) | How to train, evaluate, and optimize agents |
| [Contributing](docs/CONTRIBUTING.md) | How to contribute code, tests, and documentation |
| [Code of Conduct](docs/CODE_OF_CONDUCT.md) | Community guidelines |
| [Changelog](CHANGELOG.md) | Version history and notable changes |

## Project Structure

```
PeraMorphic-DOOM-SNN/
├── config/                       # Configuration files
│   ├── doom_cfg/                 # ViZDoom scenario configs (.cfg)
│   │   ├── simple.cfg            # Simple movement + shooting
│   │   ├── deadly_corridor.cfg   # Navigate corridor, kill enemies
│   │   └── deathmatch.cfg        # Full combat scenario
│   └── train/                    # Training hyperparameters (YAML)
│       ├── ppo_simple.yaml
│       ├── ppo_deadly_corridor.yaml
│       ├── ppo_deathmatch.yaml
│       └── snn_finetune.yaml
├── env/                          # Environment implementation
│   └── vizdoom_env.py            # Gymnasium wrapper for ViZDoom
├── models/                       # Neural network architectures
│   ├── ann/                      # Non-spiking ANN (PPO baseline)
│   │   └── ppo_network.py        # DoomNet architecture (~1.2M params)
│   └── snn/                      # Spiking neural networks
│       ├── lif_neuron.py         # LIF neuron + spiking layers
│       ├── snn_cnn.py            # Spiking CNN (SNN equivalent of DoomNet)
│       └── snn_policy.py         # SNN policy wrapper
├── training/                     # Training pipelines
│   ├── train_ann.py              # PPO training with Stable-Baselines3
│   ├── convert_and_simulate.py   # ANN-to-SNN conversion pipeline
│   └── train_snn.py              # SNN fine-tuning with surrogate gradients
├── eval/                         # Evaluation utilities
│   ├── evaluate.py               # Episode evaluation + metrics
│   └── visualize.py              # Frame recording + spike visualization
├── utils/                        # Utility modules
│   ├── reward.py                 # Reward function builder
│   ├── preprocess.py             # Frame preprocessing pipeline
│   └── curriculum.py             # Scenario progression logic
├── research/                     # Research tracking
│   └── notes.md                  # Team research log
├── docs/                         # Documentation
│   ├── SETUP.md
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── TRAINING.md
│   ├── CONTRIBUTING.md
│   └── CODE_OF_CONDUCT.md
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
├── setup.sh                      # Automated setup script
├── CHANGELOG.md
├── CONTRIBUTING.md
└── README.md
```

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Ubuntu 20.04 / Windows 10+ | Ubuntu 22.04 LTS |
| **Python** | 3.10 | 3.10 - 3.11 |
| **PyTorch** | 2.0+ | 2.1+ |
| **ViZDoom** | 1.2.4 | Latest stable |
| **GPU** | Any (CPU support) | NVIDIA GPU 4GB+ VRAM |
| **RAM** | 8 GB | 16 GB |

See [Setup Guide](docs/SETUP.md) for detailed installation instructions.

## ViZDoom Installation

### Linux (Recommended)

```bash
pip install vizdoom
```

### Windows (WSL2)

ViZDoom on Windows works best through WSL2 (Windows Subsystem for Linux). See [Setup Guide](docs/SETUP.md#method-2-windows-wsl2-recommended-for-windows-users) for details.

### Windows (Native)

ViZDoom has pre-built Windows wheels, but they may lag behind the latest version. See [Setup Guide](docs/SETUP.md#method-3-windows-native-experimental) for installation.

## Training Scenarios

The project uses curriculum learning to progressively train on harder scenarios:

| Stage | Scenario | Timesteps | Focus |
|-------|----------|-----------|-------|
| 1 | **Simple** | 5M | Learn basic movement + shooting |
| 2 | **DeadlyCorridor** | 10M | Navigate corridor, kill enemies |
| 3 | **Deathmatch** | 20M | Full combat: navigate, shoot, survive |

Each stage builds on the previous one. See [Training Guide](docs/TRAINING.md) for detailed training instructions.

## SNN Framework

Uses **snnTorch** for spiking neural network implementation:

| Feature | Detail |
|---------|--------|
| **Neuron Model** | Leaky Integrate-and-Fire (LIF) |
| **Encoding** | Rate coding (spike count ≈ action probability) |
| **Training** | Surrogate gradients via Straight-Through Estimator |
| **ANN→SNN** | Direct weight copying + LIF activation replacement |

## Architecture

The core architecture is the **DoomNet** — a 3-layer CNN proven for ViZDoom:

```
Input (8, 84, 84)
    │
    ▼
Conv2d → ReLU → Conv2d → ReLU → Conv2d → ReLU
    │
    ▼
Linear → ReLU → Linear (policy) + Linear (value)
```

Both ANN and SNN use this same structure. The SNN replaces ReLU with LIF neurons.

See [Architecture Guide](docs/ARCHITECTURE.md) for the full system architecture.

## FPGA Deployment

Upon completion, a `models/fpga_spec.json` will be generated containing:

- Full network architecture (layers, dimensions)
- Spiking LIF neuron parameters (tau, thresholds, reset values)
- Model weights (numpy `.npz` format)
- Spike statistics for hardware estimation

## Getting Started

New to the project? Start here:

1. **[Setup Guide](docs/SETUP.md)** — Install dependencies and verify your environment
2. **[Quick Start](#quick-start)** — Run the full pipeline
3. **[Training Guide](docs/TRAINING.md)** — Train and evaluate agents
4. **[Architecture](docs/ARCHITECTURE.md)** — Understand how everything works
5. **[API Reference](docs/API.md)** — Explore the code

## License

MIT License — see [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [Contributing Guide](docs/CONTRIBUTING.md) for guidelines.
