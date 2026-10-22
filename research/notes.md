# Research Notes

Team research log, experiment tracking, and reference materials for the PeraMorphic-DOOM-SNN project.

## Table of Contents

- [Setup Notes](#setup-notes)
- [Key Papers](#key-papers)
- [Experiments](#experiments)
- [SNN Literature](#snn-literature)
- [DOOM AI Literature](#doom-ai-literature)
- [FPGA Considerations](#fpga-considerations)

## Setup Notes

Tasks completed during environment setup:

- [ ] ViZDoom tutorial: Run `ViZDoom/src/python/examples/basic.py`
- [ ] Stable-Baselines3 tutorial: Train PPO on CartPole
- [ ] snnTorch tutorial: Build spiking MNIST classifier
- [ ] Install CUDA toolkit and verify GPU accessibility
- [ ] Test ViZDoom headless rendering on target platform

## Key Papers

### Foundational

| Paper | Year | Key Idea | Project Relevance |
|-------|------|----------|-------------------|
| **Mnih et al., "Playing Atari with Deep RL"** (DQN) | 2013 | Deep Q-Network with experience replay | Foundation of RL-from-pixels approach |
| **Schulman et al., "PPO"** | 2017 | Proximal Policy Optimization | Chosen RL algorithm — stable, sample-efficient |
| **Budzianowski & Koutnar, "ViZDoom"** | 2016 | ViZDoom research platform | Our game environment |

### SNN-Specific

| Paper | Year | Key Idea | Project Relevance |
|-------|------|----------|-------------------|
| **Maass, "Networks of Spiking Neurons"** | 1997 | LIF neuron model | Our neuron model choice |
| **Zenke & Ganguli, "Surrogate Gradient"** | 2018 | STE for SNN training | Surrogate gradient method |
| **Neftci et al., "Surrogate Gradient Learning"** | 2019 | Practical SNN training | Training methodology |
| **Hunsberger & Eliasmith, "Spiking Deep Nets"** | 2015 | ANN-to-SNN conversion | Our conversion approach |
| **Tieleman & Hinton, "RMSProp"** | 2012 | Adaptive optimizer | Potential optimizer alternative |

## Experiments

Use this template for each experiment:

```
### YYYY-MM-DD — Experiment Name

**Goal:** What are we trying to verify or measure?

**Setup:**
- Model: (DoomNet, SpikingCNN, custom variant)
- Scenario: (simple, deadly_corridor, deathmatch)
- Hyperparameters: (LR, batch size, timesteps, etc.)
- Hardware: (GPU model, CPU only)

**Results:**
- Metric 1: value
- Metric 2: value
- (Include tables or links to plots)

**Conclusion:**
- What did we learn?
- What should we try next?
```

### Experiment Template

```
### YYYY-MM-DD — [Name]

**Goal:**

**Setup:**
- Model:
- Scenario:
- Hyperparameters:
- Hardware:

**Results:**
- Metric 1:
- Metric 2:

**Conclusion:**

**Next Steps:**
```

## SNN Literature

Notes on SNN-specific research relevant to our project:

### Rate vs Temporal Coding

- **Rate coding**: Average spike count over T timesteps represents output value. Simple, direct ANN→SNN mapping. Used in this project.
- **Temporal coding**: Spike timing encodes information. More efficient (fewer spikes) but harder to train.
- **Latency coding**: First-spike timing represents value. Could be explored in Phase 4.

### Surrogate Gradient Methods

| Method | Paper | Pros | Cons |
|--------|-------|------|------|
| Straight-Through Estimator | Bengio et al. 2013 | Simple, effective | Gradient approximation |
| Leaky Sigmoid | Shrestha & Orchard 2018 | Smooth gradient | Needs alpha tuning |
| Finite Difference | Bertschinger & Natschlager 2004 | Exact | Computationally expensive |

### Performance Benchmarks

| Dataset | Method | Accuracy | Sparsity |
|---------|--------|----------|----------|
| MNIST | Spiking ResNet | ~98.5% | ~5-10% |
| CIFAR-10 | ANN→SNN conversion | ~95% | ~10-15% |
| Spiking MNIST | Temporal coding | ~95% | ~2-5% |

## DOOM AI Literature

### ViZDoom Benchmark Results

| Method | Simple | DeadlyCorridor | DefendMyPosition |
|--------|--------|----------------|------------------|
| DQN (original) | ~8.0 | ~3.5 | ~2.0 |
| PPO (SB3) | ~10.0 | ~5.0 | ~4.0 |
| Rainbow DQN | ~10.5 | ~6.0 | ~4.5 |
| SOTA (human benchmark) | ~15.0 | ~10.0 | ~8.0 |

### Reward Design Patterns

Standard ViZDoom reward function:
```
reward = +1.0 * Δhealth + 0.1 * Δarmor + 10.0 * Δkills - 100.0 * died
```

Alternative reward designs to explore:
- Intrinsic motivation (curiosity-based)
- Shaped rewards (proximity to enemies, health pickups)
- Curriculum-based rewards (gradually increase difficulty)

## FPGA Considerations

### Requirements to Determine

- [ ] LIF neuron precision (fixed-point bit depth)
- [ ] Spike encoding format (dense vs sparse)
- [ ] Interconnect topology (mesh, torus, custom)
- [ ] Memory bandwidth constraints
- [ ] Power budget
- [ ] Target FPGA device (Xilinx/Intel)

### Estimated Resource Budget

| Component | Parameters | Approximate FPGA Resources |
|-----------|-----------|---------------------------|
| Conv1 (8→32) | 8×32×8×8 = 2,048 weights | ~2K multipliers |
| Conv2 (32→64) | 32×64×4×4 = 32,768 weights | ~32K multipliers |
| Conv3 (64→64) | 64×64×3×3 = 36,864 weights | ~36K multipliers |
| FC1 (3136→512) | 3136×512 = 1,605,632 weights | ~1.6M multipliers |
| LIF neurons | ~1.7K per layer | ~2K neuron cores |

### LIF Neuron FPGA Implementation

```
V(t) = leak × V(t-1) + current(t)
```

- **Membrane potential**: Fixed-point (16-bit)
- **Leak factor**: Fixed-point (8-bit)
- **Current accumulation**: Fixed-point (16-bit)
- **Threshold comparison**: 1-bit
- **Spike output**: 1-bit event stream

## Links and Resources

- [ViZDoom GitHub](https://github.com/mwydmuch/ViZDoom)
- [snnTorch Documentation](https://snntorch.readthedocs.io/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [ViZDoom Scenarios](https://github.com/mwydmuch/ViZDoom/tree/master/scenarios)
- [PeraMorphIQ Website](https://peramorphiq.ce.pdn.ac.lk)
