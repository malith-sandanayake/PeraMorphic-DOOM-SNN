# Architecture

System architecture, data flow, and design decisions for the PeraMorphic-DOOM-SNN pipeline.

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ViZDoom Engine                     PPO Training Loop               │
│  ┌──────────────┐                   ┌──────────────────────────┐    │
│  │              │  frames (obs)   │  │                        │    │
│  │  DoomEnv     │────────────────►│  │  DoomNet (ANN)         │    │
│  │  (Gym API)   │                 │  │  Conv1→Conv2→Conv3→FC  │    │
│  │              │  reward, done   │  │                        │    │
│  └──────┬───────┘                 │  └───────────┬────────────┘    │
│         │                          └─────────────┼────────────────┘
│         │                                        │
│         │                              (weights updated)
│         │                                        │
│         ▼                                        ▼
│  ┌──────────────┐                   ┌──────────────────────────┐    │
│  │  DoomEnv     │  frames (obs)   │  │ SpikingCNN (SNN)       │    │
│  │  (Gym API)   │────────────────►│  │ Conv1+LIF→...→FC+LIF   │    │
│  │              │                 │  │                         │    │
│  │              │  reward, done   │  └───────────┬────────────┘    │
│  └──────────────┘                 │              │                │
│                                   └──────────────┼────────────────┘
│                                                  │ (surrogate grads)
│         ┌────────────────────────────────────────┘
│         │
│         ▼
│  ┌──────────────────────────────────────────────────────┐
│  │  EVALUATION                                          │
│  │  • Ann agent → kills, survival, reward               │
│  │  • SNN agent → spikes, firing rate, reward           │
│  │  • Comparison reports → eval/results_*.json          │
│  └──────────────────────────────────────────────────────┘
│
└─────────────────────────────────────────────────────────────────────┘
```

## Component Overview

Each component is described below with its responsibilities, inputs/outputs, and design rationale.

### 1. Environment (`env/`)

The environment is the bridge between the DOOM game engine and the RL algorithm.

**`vizdoom_env.py`**

| Aspect | Detail |
|--------|--------|
| **Role** | Wraps ViZDoom's native C++ interface in a Gymnasium-compatible environment |
| **Input** | ViZDoom game state (screens, depth maps, player stats) |
| **Output** | Gymnasium observation dict, reward float, termination flags, info dict |
| **Key Class** | `DoomEnv(gym.Env)` |

**Why Gymnasium**: Standardizes the environment interface so any Gym-compatible RL library (SB3, RLlib) can be used without modification.

**Observation Space**:
- **Screen buffer**: RGB 64×48, converted to 84×84 via bicubic interpolation
- **4-frame stack**: Temporal context stored in `_frame_stack` (4×H×W×3)
- **Depth buffer** (optional): Grayscale depth map, concatenated as 4th channel

**Action Space**:
- **Simple**: 12 discrete actions (movement + turning + shooting + combinations)
- **Deathmatch**: 17 actions (adds backward-variant combinations)

**Reward Function** (internal):
```
reward = 1.0 * Δhealth + 0.1 * Δarmor + 10.0 * Δkills - 100.0 * died
```
This is computed as a delta from the previous frame, providing dense learning signal.

**Design decisions**:
- Frame skip = 1 (no downsampling of action frequency)
- Headless mode for training; window visible only for manual testing
- Random seed propagated to ViZDoom for reproducibility

---

### 2. Frame Preprocessing (`utils/preprocess.py`)

**`FrameProcessor`** is the central preprocessing pipeline:

```
Raw Frame (64×48×3 RGB)
        │
        ▼
┌───────────────┐
│  Resize       │  64×48 → 84×84 (bilinear)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Stack 4      │  Append to ring buffer, keep last 4
│  Frames       │  Shape: (84, 84, 12)  [H, W, C×4]
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Normalize    │  / 255.0 → [0, 1] float32
└───────┬───────┘
        │
        ▼
  Output: (84, 84, 12) float32
```

**Why 4-frame stacking**: SNNs naturally handle temporal sequences, but frame stacking provides explicit motion information that helps both ANN and SNN agents learn effective policies faster.

**Why RGB over grayscale**: Color information helps distinguish enemy types, health pickups, and weapons — critical features for DOOM gameplay that are lost in grayscale.

---

### 3. ANN Model (`models/ann/`)

**`DoomNet`** — The non-spiking policy network trained with PPO:

```
Input: (batch, 8, 84, 84)  [8 channels = 4 RGB frames stacked]
        │
        ▼
┌─────────────────────────────────┐
│  Feature Extractor (Sequential) │
│  ┌───────────────────────────┐  │
│  │ Conv2d(8→32, k=8, s=4)   │  │ → (32, 19, 19)
│  │ ReLU                       │  │
│  ├───────────────────────────┤  │
│  │ Conv2d(32→64, k=4, s=2)  │  │ → (64, 9, 9)
│  │ ReLU                       │  │
│  ├───────────────────────────┤  │
│  │ Conv2d(64→64, k=3, s=1)  │  │ → (64, 7, 7)
│  │ ReLU                       │  │
│  └───────────────────────────┘  │
└───────────┬─────────────────────┘
            │ flatten → (3136)
            ▼
┌─────────────────────────────────┐
│  Fully Connected                │
│  Linear(3136→512) + ReLU        │
└───────┬───────────────┬─────────┘
        │               │
        ▼               ▼
┌──────────────┐  ┌────────────┐
│ Policy Head  │  │ Value Head │
│ (12 logits)  │  │ (1 scalar) │
└──────────────┘  └────────────┘
```

**Parameter count**: ~1.2M total
- Conv1: 8×32×8×8 + 32 = 16,416
- Conv2: 32×64×4×4 + 64 = 32,832
- Conv3: 64×64×3×3 + 64 = 36,928
- FC1: 3136×512 + 512 = 1,606,144
- Heads: 512×12 + 12 + 512×1 + 1 = 6,685

**Why DoomNet**: This is the proven architecture from the original ViZDoom paper. It provides a strong baseline that is well-understood and reproducible.

---

### 4. SNN Model (`models/snn/`)

**`SpikingCNN`** — The spiking equivalent of DoomNet:

| ANN Layer | SNN Equivalent | Key Difference |
|-----------|---------------|----------------|
| `Conv2d + ReLU` | `Conv2d + LIF` | Output is spike count over T timesteps, not instantaneous activation |
| `Linear + ReLU` | `Linear + LIF` | Same |
| `Linear` (policy) | `Linear` (no activation) | Output is raw spike count; firing rate = spikes/T ≈ action probability |
| `Linear` (value) | `Linear` (no activation) | Same |

**Rate Coding**: The fraction of timesteps a neuron fires represents its output value:

```
action_probability[a] = total_spikes[a] / T
```

For T=16 timesteps, if a neuron fires 12 times, its firing rate is 0.75, meaning action a has 75% probability.

**`LIFNode`** — The core neuron model:

```
Membrane Potential Dynamics:
  V(t) = leak × V(t-1) + current(t)

Spike Detection:
  spike(t) = 1 if V(t) ≥ V_thresh else 0

After Spike:
  V(t) = V_reset   (typically 0.0)

Surrogate Gradient (for backprop):
  d(spike)/d(V) ≈ sigmoid_alpha(V - V_thresh)
  (gradient flows through "soft" approximation of threshold)
```

**Why rate coding over temporal coding**:
- Simplest mapping from ANN to SNN (direct weight copying)
- Well-understood learning dynamics
- Compatible with PPO policy gradient (firing rate ≈ probability)
- Easier for beginners to understand and debug

**Why LIF over other neuron models**:
- Simplest spiking neuron model (2 parameters: tau, leak)
- Most commonly used in SNN literature
- Directly corresponds to ANN activations (integrate = sum, spike = activation)
- Hardware-friendly for FPGA deployment (PeraMorphIQ target)

---

### 5. Training Pipelines (`training/`)

#### 5a. PPO Training (`train_ann.py`)

**Algorithm**: Proximal Policy Optimization (PPO-Clip)

**Hyperparameters** (tuned for ViZDoom):
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 3×10⁻⁴ | Standard for PPO, works well across environments |
| n_steps | 2048 | Balance between on-policy freshness and compute |
| batch_size | 64 | Fits in GPU memory for DoomNet |
| n_epochs | 10 | More epochs = better data reuse per batch |
| gamma | 0.99 | Long-term reward emphasis |
| gae_lambda | 0.95 | Balanced bias-variance for advantage estimation |
| clip_range | 0.2 | Standard PPO clipping for stable updates |
| ent_coef | 0.01 | Exploration bonus to prevent premature convergence |

**Training loop**:
```
for each iteration:
    for each parallel env (4):
        collect n_steps frames using current policy
    stack observations: (batch, 4 frames × 2 channels, 84, 84)
    compute advantages via GAE
    update network for n_epochs on collected batch
    record metrics to TensorBoard
```

**Parallelism**: 4 `SubprocVecEnv` instances run DOOM episodes in parallel, reducing correlation between samples and speeding up training.

#### 5b. SNN Fine-tuning (`train_snn.py`)

**Core challenge**: Spiking is non-differentiable. The `spike = V ≥ V_thresh` operation has zero gradient everywhere except at the threshold (a delta function).

**Solution**: Surrogate Gradient via Straight-Through Estimator (STE):

```python
# During forward pass (real spike):
spike = (voltage >= threshold).float()

# During backward pass (surrogate gradient):
grad_via_surr = sigmoid_alpha(voltage - threshold) * grad_from_above
```

**Training loop**:
```
for each iteration:
    collect experiences with SNN agent
    for t in range(T):                      # simulate T timesteps
        voltage, spike = LIF_step(current)  # spiking dynamics
    compute PPO loss from spike rates       # rate = spikes/T
    backward()                              # surrogate gradient
    optimizer.step()
```

---

### 6. Conversion Pipeline (`convert_and_simulate.py`)

The ANN-to-SNN conversion process:

```
Trained ANN Checkpoint
        │
        │ PPO.load("models/ann_simple")
        │
        ▼
┌─────────────────────────────────┐
│  Extract architecture details   │
│  - n_actions, input_channels    │
│  - layer dimensions             │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│  Create SpikingCNN with         │
│  same dimensions                │
└───────────┬─────────────────────┘
            │
            │ copy conv weights
            │ copy fc weights
            │ copy head weights
            ▼
┌─────────────────────────────────┐
│  SpikingCNN with ANN weights    │
│  (but LIF instead of ReLU)      │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│  Evaluate → measure performance │
│  drop from ANN baseline         │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│  Fine-tune with surrogate       │
│  gradients to recover loss      │
└─────────────────────────────────┘
```

**Expected performance drop after conversion**: 20-40% kill rate, due to:
1. ReLU → LIF changes activation dynamics
2. Spike threshold creates a dead zone (no output below threshold)
3. Discrete spikes have higher variance than continuous activations

**Recovery via fine-tuning**: With T=16 and learning rate 1×10⁻⁴, the SNN typically recovers to 80-95% of ANN performance.

---

### 7. Curriculum Learning (`utils/curriculum.py`)

Progressive difficulty increases learning success rate:

```
Phase 1: Simple (5M steps)
    └── Learn: move, look, shoot, avoid enemies
    └── Target: 5+ kills/episode
    
         ↓ (load checkpoint)
    
Phase 2: DeadlyCorridor (10M steps)
    └── Learn: navigate complex geometry, manage ammo
    └── Target: reach end of corridor with kills
    
         ↓ (load checkpoint)
    
Phase 3: Deathmatch (20M steps)
    └── Learn: multi-enemy combat, positioning, weapon switching
    └── Target: sustained combat survival
```

**Why curriculum**: Each scenario builds skills the previous one teaches. Jumping directly to Deathmatch would overwhelm a learning agent — the action space and enemy density are too high.

---

## Data Flow Summary

```
┌────────────┐    raw frames     ┌─────────────┐
│  ViZDoom   │─────────────────►│  DoomEnv    │
│  Engine    │  + player stats  │  (Gym API)  │
└────────────┘                  └──────┬──────┘
                                       │
                                       │ stacked, normalized
                                       ▼
                               ┌───────────────┐
                               │  DoomNet      │──► PPO update
                               │  or           │
                               │  SpikingCNN   │──► SNN update
                               └───────────────┘
                                       │
                                       │ action
                                       ▼
                               ┌───────────────┐
                               │  DoomEnv      │──► new frame
                               └───────────────┘
```

## Design Trade-offs

| Decision | Trade-off | Rationale |
|----------|-----------|-----------|
| PyTorch over JAX | Ecosystem size vs speed | PyTorch has better SNN support (snnTorch) and ViZDoom compatibility |
| Rate coding over temporal | Simplicity vs efficiency | Rate coding enables direct ANN→SNN conversion; temporal coding requires re-architecture |
| PPO over DQN | Computation vs stability | PPO handles large discrete action spaces (12-17 actions) more stably |
| 84×84 over 256×256 | Speed vs information | 84×84 is the standard for RL; DOOM's simple graphics don't need high resolution |
| CPU training over GPU | Accessibility vs speed | All team members can train on laptops; GPU only for faster iterations |
