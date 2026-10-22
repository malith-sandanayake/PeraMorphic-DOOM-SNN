# API Reference

Complete reference for all public classes, functions, and attributes.

## Environment API

### `DoomEnv`

**Module**: `env.vizdoom_env`

Gymnasium environment wrapping ViZDoom for DOOM AI research.

```python
from env.vizdoom_env import DoomEnv

env = DoomEnv(
    scenario="simple",          # "simple", "deadly_corridor", "deathmatch"
    screen_size=(64, 48),       # (width, height)
    frame_skip=1,               # frames per RL step
    use_depth=False,            # include depth buffer
    use_hud=True,               # render HUD elements
    seed=42,                    # reproducibility
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenario` | `str` | `"simple"` | ViZDoom scenario name |
| `screen_size` | `tuple[int, int]` | `(64, 48)` | Screen resolution (W, H) |
| `frame_skip` | `int` | `1` | How many frames to process per RL step |
| `use_depth` | `bool` | `False` | Include depth buffer in observations |
| `use_hud` | `bool` | `True` | Enable HUD rendering |
| `seed` | `int or None` | `None` | Random seed for reproducibility |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `action_space` | `gym.spaces.Discrete` | Discrete action space (12 or 17 actions) |
| `observation_space` | `gym.spaces.Box` | Box space for frame observations |

#### Methods

##### `reset(seed=None, options=None) -> (obs, info)`

Reset environment to initial state.

**Returns**:
- `obs`: numpy array `(stacked_H, W, C)` — stacked frames
- `info`: dict with `kills`, `deaths`, `health`, `armor`, `episode_steps`

**Example**:
```python
obs, info = env.reset(seed=42)
```

##### `step(action) -> (obs, reward, terminated, truncated, info)`

Execute one environment step.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `action` | `int` | Action index (0 to num_actions-1) |

**Returns**:
| Value | Type | Description |
|-------|------|-------------|
| `obs` | `np.ndarray` | Stacked frame observation |
| `reward` | `float` | Shaped reward |
| `terminated` | `bool` | Episode ended (death or goal) |
| `truncated` | `bool` | Episode timeout |
| `info` | `dict` | Episode statistics |

**Example**:
```python
obs, reward, terminated, truncated, info = env.step(5)
if terminated or truncated:
    obs, info = env.reset()
```

##### `render() -> np.ndarray or None`

Get current screen buffer.

**Returns**: RGB array `(H, W, 3)` or `None` if window hidden

##### `get_player_stats() -> DoomPlayerStats`

Get current player statistics from ViZDoom.

**Returns**: `DoomPlayerStats` with health, kills, ammo, position, etc.

##### `close()`

Shut down ViZDoom game instance.

##### `seed(seed=None) -> list`

Set random seed.

---

### `DoomStackedEnv`

**Module**: `env.vizdoom_env`

Wrapper that stacks 4 consecutive frames for temporal context.

```python
from env.vizdoom_env import DoomEnv, DoomStackedEnv

env = DoomStackedEnv(DoomEnv(scenario="simple"), stack_size=4)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env` | `DoomEnv` | — | Base environment to wrap |
| `stack_size` | `int` | `4` | Number of frames to stack |

---

## Model API

### `DoomNet`

**Module**: `models.ann.ppo_network`

DoomNet CNN policy network for PPO in ViZDoom.

```python
from models.ann.ppo_network import DoomNet

model = DoomNet(
    input_channels=8,   # 4 RGB frames × 2 channels (screen + depth)
    n_actions=12,       # number of discrete actions
    hidden_dim=512,     # FC layer size
)
```

#### Architecture

```
Input: (B, C, H, W)          # e.g. (batch, 8, 84, 84)
        │
        ▼
Conv2d(C→32, k=8, s=4) + ReLU    # → (B, 32, 19, 19)
Conv2d(32→64, k=4, s=2) + ReLU   # → (B, 64, 9, 9)
Conv2d(64→64, k=3, s=1) + ReLU   # → (B, 64, 7, 7)
        │
        ▼
flatten → (B, 3136)
Linear(3136→512) + ReLU
        │
        ├──► Linear(512→n_actions)  # Policy logits
        └──► Linear(512→1)          # Value estimate
```

#### Methods

##### `forward(x) -> (policy_logits, values)`

Forward pass.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Input tensor `(B, C, H, W)` |

**Returns**: `(policy_logits, values)` — policy logits `(B, n_actions)` and values `(B, 1)`

##### `get_action_and_value(x, action=None) -> tuple`

Forward pass with optional action value computation.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Input tensor `(B, C, H, W)` |
| `action` | `torch.Tensor or None` | Optional action tensor `(B,)` |

**Returns**:
- If `action=None`: `(policy_logits, values)`
- If `action` provided: `(log_probs, values, entropy)`

---

### `LIFNode`

**Module**: `models.snn.lif_neuron`

Leaky Integrate-and-Fire neuron with surrogate gradient training.

```python
from models.snn.lif_neuron import LIFNode

neuron = LIFNode(
    tau=16.0,           # Membrane time constant
    leak=1.0,           # Leak factor (1.0 = no decay)
    V_thresh=1.0,       # Firing threshold
    V_reset=0.0,        # Reset potential
    learn_tau=False,    # Learnable tau
    learn_thresh=False, # Learnable threshold
)
```

#### Forward dynamics

```
V(t) = leak × V(t-1) + current(t)
spike(t) = 1 if V(t) ≥ V_thresh else 0
V(t) = V_reset if spike else V(t)
```

#### Methods

##### `forward(current, voltage=None, spike=None) -> (voltage, (spike, hidden))`

Process one timestep.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `current` | `torch.Tensor` | Input current (same shape as voltage) |
| `voltage` | `torch.Tensor or None` | Previous membrane potential (auto-init if None) |
| `spike` | `torch.Tensor or None` | Previous spike output (auto-init if None) |

**Returns**: `(voltage, (spike, hidden_dict))` — updated voltage and tuple of spike + hidden state

##### `init_hidden(x) -> (voltage, hidden_dict)`

Initialize hidden state for a new sequence.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Input tensor to determine shape and device |

**Returns**: `(voltage, {"voltage": voltage})` — initial hidden state

---

### `LIFConv2d`

**Module**: `models.snn.lif_neuron`

Convolutional layer + LIF neuron (spiking equivalent of Conv2d + ReLU).

```python
from models.snn.lif_neuron import LIFConv2d

conv_snn = LIFConv2d(
    in_channels=8,
    out_channels=32,
    kernel_size=8,
    stride=4,
    tau=16.0,
)
```

#### Methods

##### `forward(x) -> spikes`

Process sequence of frames.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Input `(B, T, C, H, W)` — sequence of frames |

**Returns**: `torch.Tensor` — spike counts `(B, T, C_out, H_out, W_out)`

---

### `LIFLinear`

**Module**: `models.snn.lif_neuron`

Linear layer + LIF neuron (spiking equivalent of Linear + ReLU).

```python
from models.snn.lif_neuron import LIFLinear

linear_snn = LIFLinear(
    in_features=512,
    out_features=12,
    tau=16.0,
)
```

#### Methods

##### `forward(x) -> spikes`

Process sequence through linear + LIF.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Input `(B, T, in_features)` |

**Returns**: `torch.Tensor` — spike counts `(B, T, out_features)`

---

### `SpikingCNN`

**Module**: `models.snn.snn_cnn`

Spiking CNN policy network for PPO-based DOOM agents.

```python
from models.snn.snn_cnn import SpikingCNN

snn = SpikingCNN(
    n_actions=12,
    T=16,               # simulation time steps per env step
    tau=16.0,           # LIF neuron time constant
    leak=1.0,           # LIF leak factor
    V_thresh=1.0,       # Firing threshold
    input_channels=8,   # 4 RGB frames × 2 channels
)
```

#### Architecture

```
Input: (B, T, C, H, W)            # e.g. (2, 16, 8, 84, 84)
        │  (T timesteps)
        ▼
for t in range(T):
    Conv2d + ReLU + LIF → spike1
    Conv2d + ReLU + LIF → spike2
    Conv2d + ReLU + LIF → spike3
    flatten + FC1 + ReLU + LIF → spike4
    policy_logits = FC1(spike4)     # (B, n_actions)
    value_logits = FC2(spike4)      # (B, 1)
    accumulate over T timesteps
        │
        ▼
Policy: rate = accumulated / T    # (B, n_actions)
Value: rate = accumulated / T     # (B,)
```

#### Methods

##### `forward(x) -> (policy_rate, value, hidden)`

Full forward pass with T timestep simulation.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Input `(B, T, C, H, W)` |

**Returns**:
- `policy_rate`: `(B, n_actions)` — normalized spike count (= firing rate)
- `value`: `(B,)` — value estimate
- `hidden`: `dict` — hidden state dict (for BPTT)

##### `get_action_and_value(x, action=None) -> tuple`

Forward pass with action value computation.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Input `(B, T, C, H, W)` |
| `action` | `torch.Tensor or None` | Optional action tensor `(B,)` |

**Returns**: `(log_probs, values, entropy, hidden)` if action provided; `(policy_rate, values, hidden)` if not

##### `convert_from_ann(ann_model) -> None`

Load weights from trained ANN DoomNet model.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `ann_model` | `DoomNet` | Trained ANN model from `models.ann.ppo_network` |

Copies all convolutional, linear, and head weights from ANN to SNN.

---

### `SNNPolicy`

**Module**: `models.snn.snn_policy`

SNN policy wrapper for action selection and PPO training.

```python
from models.snn.snn_policy import SNNPolicy

policy = SNNPolicy(
    n_actions=12,
    T=16,
    tau=16.0,
    leak=1.0,
    V_thresh=1.0,
    input_channels=8,
)
```

#### Methods

##### `forward(x, action=None) -> dict`

Full forward pass returning action, log_prob, value, entropy.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Input `(B, T, C, H, W)` |
| `action` | `torch.Tensor or None` | Optional action tensor |

**Returns**: `dict` with `action`, `log_prob`, `value`, `entropy`, `hidden` (if action provided)

##### `act(observation, deterministic=False) -> (action, value)`

Select action for a single observation.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `observation` | `torch.Tensor` | 4D `(T, C, H*stack, W)` or 5D `(B, T, C, H, W)` |
| `deterministic` | `bool` | If True, pick highest-rate action |

**Returns**: `(action, value)` — action index and value estimate, both scalar tensors

##### `convert_from_ann(ann_model) -> None`

Convert ANN weights to SNN weights.

---

## Training API

### `ProgressCallback`

**Module**: `training.train_ann`

Callback that prints training progress every N timesteps.

```python
from training.train_ann import ProgressCallback

cb = ProgressCallback(print_every=1_000_000)
model.learn(5_000_000, callback=cb)
```

### `EvaluationCallback`

**Module**: `training.train_ann`

Callback that periodically evaluates the agent during training.

```python
from training.train_ann import EvaluationCallback

cb = EvaluationCallback(
    eval_env=eval_env,
    n_eval_episodes=10,
    callback_after_step=1_000_000,
)
```

### `SNNPPOTrainer`

**Module**: `training.train_snn`

Trains SNN policy using PPO with surrogate gradients.

```python
from training.train_snn import SNNPPOTrainer

trainer = SNNPPOTrainer(
    snn_policy=snn_policy,
    env=env,
    device="cuda" if torch.cuda.is_available() else "cpu",
    config={
        "learning_rate": 1e-4,
        "total_timesteps": 2_000_000,
        "n_steps": 2048,
        "batch_size": 32,
        "n_epochs": 10,
        "gamma": 0.99,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "T": 16,
    },
)

trainer.train(total_timesteps=2_000_000)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `snn_policy` | `SNNPolicy` | — | SNN policy network to train |
| `env` | `DoomEnv` | — | Training environment |
| `device` | `str` | `"cpu"` | Device for model and tensors |
| `config` | `dict` | default config dict | Training hyperparameters |

#### Methods

##### `train(total_timesteps, log_interval=100_000)`

Full training loop.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `total_timesteps` | `int` | Total environment steps to train |
| `log_interval` | `int` | Steps between progress logging |

---

## Evaluation API

### `evaluate_ann_agent`

**Module**: `eval.evaluate`

Evaluate a trained PPO model across multiple episodes.

```python
from eval.evaluate import evaluate_ann_agent

results = evaluate_ann_agent(
    model=ppo_model,
    scenario="simple",
    n_episodes=100,
    screen_size=(84, 84),
    use_depth=False,
    deterministic=True,
)
```

**Returns**: `dict` with `mean_reward`, `std_reward`, `mean_kills`, `std_kills`, `mean_steps`, `std_steps`, `max_kills`, `max_steps`

### `evaluate_snn_agent`

**Module**: `eval.evaluate`

Evaluate a trained SNN policy across multiple episodes.

```python
from eval.evaluate import evaluate_snn_agent

results = evaluate_snn_agent(
    snn_policy=snn_policy,
    scenario="simple",
    n_episodes=100,
    screen_size=(84, 84),
    T=16,
    use_depth=False,
    deterministic=True,
)
```

**Returns**: `dict` — same as `evaluate_ann_agent` plus `avg_firing_rate`

### `save_results`

**Module**: `eval.evaluate`

Save evaluation results to JSON.

```python
from eval.evaluate import save_results

save_results(results, "eval/results_anna_simple.json")
```

### `print_results`

**Module**: `eval.evaluate`

Print formatted evaluation results.

```python
from eval.evaluate import print_results

print_results(results, "SNN (T=16, tau=16)")
```

---

## Visualization API

### `FrameRecorder`

**Module**: `eval.visualize`

Records DOOM gameplay frames to video.

```python
from eval.visualize import FrameRecorder

recorder = FrameRecorder(
    output_path="eval/videos/simple.mp4",
    fps=35,
    screen_size=(64, 48),
)
recorder.start()
for frame in observation_stream:
    recorder.record(frame)
recorder.stop()
```

#### Methods

| Method | Description |
|--------|-------------|
| `start()` | Clear frame buffer and start recording |
| `record(frame)` | Record single RGB frame `(H, W, 3)` |
| `stop() -> Optional[str]` | Stop recording, save frames to PNG, return path |
| `record_episode(env, model=None, n_frames=1000) -> Optional[str]` | Record complete episode with optional model |

### `plot_firing_rates`

**Module**: `eval.visualize`

Plot neuron firing rates across network layers.

```python
from eval.visualize import plot_firing_rates

plot_firing_rates(
    firing_rates=[0.15, 0.22, 0.18, 0.31, 0.45, 0.38],
    output_path="eval/firing_rates.png",
)
```

### `plot_training_progress`

**Module**: `eval.visualize`

Plot training reward progress over timesteps.

```python
from eval.visualize import plot_training_progress

plot_training_progress(
    timesteps=[0, 1e6, 2e6, 3e6, 4e6, 5e6],
    rewards=[-5.2, -2.1, 1.3, 3.7, 4.5, 5.1],
    output_path="eval/training_progress.png",
)
```

---

## Utility API

### `RewardConfig`

**Module**: `utils.reward`

Configurable reward shaping with weighted components.

```python
from utils.reward import RewardConfig

config = RewardConfig(
    health_gain_weight=1.0,
    kill_reward=10.0,
    death_penalty=-100.0,
    survival_reward=0.01,
)

reward = config.compute_reward(
    prev_health=50,
    current_health=75,
    prev_armor=0,
    current_armor=25,
    prev_kills=2,
    current_kills=3,
    prev_weapon=1,
    current_weapon=1,
    died=False,
    episode_step=100,
)
# reward = 1.0*25 + 0.1*25 + 10.0*1 + 0.01 = 37.51
```

### `simple_reward`

**Module**: `utils.reward`

Minimal reward for quick testing.

```python
from utils.reward import simple_reward

reward = simple_reward(
    prev_health=100,
    current_health=100,
    prev_kills=0,
    current_kills=1,
    died=False,
)
# reward = 10.0 (one kill)
```

### `FrameProcessor`

**Module**: `utils.preprocess`

Configurable frame preprocessing pipeline.

```python
from utils.preprocess import FrameProcessor

processor = FrameProcessor(
    target_size=(84, 84),
    stack_size=4,
    normalize=True,
    grayscale=False,
)

# Process single frame
stacked = processor.process(raw_frame)  # returns (H, W, C*4) normalized

# Process batch
batch_tensor = processor.preprocess_batch([frame1, frame2, ...])  # (N, C, H, W)

# Reset buffer between episodes
processor.reset()
```

### `resize_frame`

**Module**: `utils.preprocess`

Resize a DOOM frame to target size.

```python
from utils.preprocess import resize_frame

resized = resize_frame(frame, target_size=(84, 84), interpolation=cv2.INTER_AREA)
```

### `stack_frames`

**Module**: `utils.preprocess`

Stack multiple frames along the channel axis.

```python
from utils.preprocess import stack_frames

stacked = stack_frames(frames, stack_size=4, channels_last=True)
```

### `CurriculumTrainer`

**Module**: `utils.curriculum`

Manages progressive training across DOOM scenarios.

```python
from utils.curriculum import CurriculumTrainer, CurriculumStep, DEFAULT_CURRICULUM

# Use default curriculum
curriculum = CurriculumTrainer()

# Custom curriculum
custom = CurriculumTrainer(
    scenarios=[
        CurriculumStep("simple", 5_000_000, "Learn basics"),
        CurriculumStep("deadly_corridor", 10_000_000, "Navigate and kill"),
    ],
    checkpoint_dir="models",
)

# Iterate through scenarios
for step in curriculum:
    print(f"{step.scenario}: {step.timesteps:,} steps")
    # ... train model ...
    curriculum.mark_completed(step.scenario)

# Get progress report
print(curriculum.progress_report())
```

#### Methods

| Method | Description |
|--------|-------------|
| `get_current_scenario() -> CurriculumStep` | Get current scenario |
| `get_checkpoint_path(scenario) -> Path` | Get checkpoint file path |
| `get_previous_checkpoint() -> Optional[Path]` | Get prior scenario checkpoint |
| `mark_completed(scenario)` | Mark scenario as finished |
| `next_step() -> bool` | Advance to next scenario |
| `progress_report() -> str` | Get formatted progress string |
