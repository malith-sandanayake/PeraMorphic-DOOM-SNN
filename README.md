# PeraMorphic-DOOM-SNN

Autonomous DOOM gameplay simulation using Spiking Neural Networks (SNNs), targeting FPGA neuromorphic accelerator deployment via PeraMorphIQ.

## Project Overview

This project builds a Python-based SNN pipeline capable of playing DOOM autonomously. The approach follows a staged strategy:

1. **ANN Baseline** — Train a standard neural network (PPO) on ViZDoom using Stable-Baselines3
2. **SNN Conversion** — Convert the trained ANN to a rate-coded SNN via snnTorch
3. **SNN Fine-tuning** — Train the SNN with surrogate gradients to recover performance
4. **Hardware Export** — Generate FPGA-compatible model specifications

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run a random agent to verify setup
python main.py --mode test-random --scenario simple

# 3. Train ANN baseline
python main.py --mode train-ann --scenario simple

# 4. Evaluate trained agent
python main.py --mode evaluate --model models/ann_simple

# 5. Convert to SNN
python main.py --mode convert --checkpoint models/ann_simple

# 6. Fine-tune SNN
python main.py --mode train-snn --checkpoint models/snn_converted
```

## Project Structure

```
config/
├── doom_cfg/           # ViZDoom .cfg scenario files
└── train/              # YAML training hyperparameters
env/
└── vizdoom_env.py      # Gymnasium + ViZDoom wrapper
models/
├── ann/                # Non-spiking ANN (PPO baseline)
│   └── ppo_network.py
└── snn/                # Spiking neural networks
    ├── lif_neuron.py   # LIF neuron implementation
    ├── snn_cnn.py      # Spiking CNN architecture
    └── snn_policy.py   # SNN policy network
training/
├── train_ann.py        # PPO training pipeline
├── convert_and_simulate.py  # ANN→SNN conversion
└── train_snn.py        # SNN fine-tuning
eval/
├── evaluate.py         # Episode evaluation & metrics
└── visualize.py        # Frame recording & visualization
utils/
├── reward.py           # Reward function builder
├── preprocess.py       # Frame preprocessing
└── curriculum.py       # Scenario progression logic
research/
└── notes.md            # Team research log
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- ViZDoom (see setup notes below)
- CUDA-capable GPU recommended for training

## ViZDoom Installation

### Linux
```bash
pip install vizdoom
```

### Windows
ViZDoom requires the ZDoom engine binaries. On Windows, it's recommended to:
1. Use **WSL2** (Ubuntu on Windows) for easier installation
2. Or install ViZDoom binaries manually from [ViZDoom releases](https://github.com/mwydmuch/ViZDoom/releases)

See `setup.sh` for automated setup.

## Scenario Progression

The project uses curriculum learning:

| Stage | Scenario | Timesteps | Focus |
|-------|----------|-----------|-------|
| 1 | Simple | 5M | Learn basic movement + shooting |
| 2 | DeadlyCorridor | 10M | Navigate corridor, kill enemies |
| 3 | Deathmatch | 20M | Full combat, multiple enemies |

## SNN Framework

Uses **snnTorch** for spiking neural network implementation:
- PyTorch-native API (easy for beginners)
- Built-in LIF neurons and surrogate gradients
- Rate-coding support for ANN-to-SNN conversion

## FPGA Deployment

Upon completion, a `models/fpga_spec.json` will be generated containing:
- Full network architecture (layers, dimensions)
- Spiked LIF neuron parameters (tau, thresholds)
- Model weights
- Spike statistics for hardware estimation

## License

MIT
