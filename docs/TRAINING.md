# Training Guide

How to train, evaluate, and optimize agents in the PeraMorphic-DOOM-SNN pipeline.

## Quick Start

```bash
# 1. Verify environment setup
python main.py --mode test-random --scenario simple --episodes 5

# 2. Train ANN baseline (Simple scenario)
python main.py --mode train-ann --scenario simple

# 3. Evaluate the trained agent
python main.py --mode evaluate --model models/ann_simple --episodes 100

# 4. Convert to SNN
python main.py --mode convert --checkpoint models/ann_simple

# 5. Fine-tune SNN
python main.py --mode train-snn --checkpoint models/snn_converted

# 6. Evaluate SNN
python main.py --mode evaluate --checkpoint models/snn_finetuned --episodes 100
```

## Training ANN Baseline

### Basic Training

```bash
# Train on Simple scenario with defaults
python main.py --mode train-ann --scenario simple

# Train on DeadlyCorridor with custom timesteps
python main.py --mode train-ann --scenario deadly_corridor --total-timesteps 10000000

# Train on Deathmatch with TensorBoard
python main.py --mode train-ann --scenario deathmatch --tensorboard
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--scenario` | `"simple"` | ViZDoom scenario (`simple`, `deadly_corridor`, `deathmatch`) |
| `--total-timesteps` | scenario default | Override total training timesteps |
| `--n-envs` | `4` | Number of parallel environments |
| `--learning-rate` | `3e-4` | PPO learning rate |
| `--screen-size` | `[84, 84]` | Target screen size `(width height)` |
| `--use-depth` | `False` | Include depth buffer in observations |
| `--tensorboard` | `False` | Enable TensorBoard logging |

### Scenario-Specific Settings

Each scenario has default configurations in `config/train/`:

| Scenario | Default Timesteps | Default n_envs | n_actions |
|----------|-------------------|----------------|-----------|
| `simple` | 5,000,000 | 4 | 12 |
| `deadly_corridor` | 10,000,000 | 4 | 12 |
| `deathmatch` | 20,000,000 | 4 | 12 |

### Training Process

1. **Environment initialization**: Creates 4 parallel ViZDoom instances
2. **Episode collection**: Each step collects frames, states, and rewards
3. **Advantage estimation**: Generalized Advantage Estimation (GAE) computes advantages
4. **Policy update**: PPO clips the policy ratio for stable updates
5. **Checkpointing**: Best model saved automatically based on evaluation reward

### Monitoring Training

**Console output**:
```
  [Step 1,000,000] Last ep reward: -2.3
  [Step 2,000,000] Last ep reward: 1.5
  [Step 3,000,000] Last ep reward: 4.8
```

**TensorBoard** (if `--tensorboard`):
```bash
tensorboard --logdir ./logs/simple/
# Open http://localhost:6006
```

Monitor: episode reward, episode length, policy entropy, value loss, gradient norms.

## Curriculum Learning

Progress through scenarios from easy to hard:

```bash
# Step 1: Simple
python main.py --mode train-ann --scenario simple --total-timesteps 5000000

# Step 2: DeadlyCorridor (loads Simple checkpoint)
python main.py --mode train-ann --scenario deadly_corridor \
    --total-timesteps 10000000 \
    --checkpoint models/ann_simple

# Step 3: Deathmatch (loads DeadlyCorridor checkpoint)
python main.py --mode train-ann --scenario deathmatch \
    --total-timesteps 20000000 \
    --checkpoint models/ann_deadly_corridor
```

**Why curriculum**: Each scenario builds skills from the previous one:
- **Simple**: Learns movement, shooting, basic survival
- **DeadlyCorridor**: Learns navigation, corridor clearing
- **Deathmatch**: Learns multi-enemy combat, positioning, weapon management

## Evaluating Agents

### Evaluate ANN Agent

```bash
# Standard evaluation (deterministic actions)
python main.py --mode evaluate --model models/ann_simple --episodes 100

# Stochastic evaluation (adds exploration noise)
python main.py --mode evaluate --model models/ann_simple --episodes 100 --deterministic false
```

### Evaluate SNN Agent

```bash
# Evaluate converted SNN
python main.py --mode evaluate --checkpoint models/snn_converted --episodes 100

# Evaluate fine-tuned SNN
python main.py --mode evaluate --checkpoint models/snn_finetuned --episodes 100
```

### Output Format

Results are printed to console and saved to `eval/results_<model_name>.json`:

```
==================================================
  ANN PPO Evaluation Results
==================================================
  Mean Reward:  5.23 ± 3.41
  Mean Kills:   3.12 ± 1.87
  Mean Steps:   450 ± 120
  Max Kills:    8
  Max Steps:    678
==================================================
```

### Programmatic Evaluation

```python
from stable_baselines3 import PPO
from eval.evaluate import evaluate_ann_agent, print_results

model = PPO.load("models/ann_simple")
results = evaluate_ann_agent(model, scenario="simple", n_episodes=100)
print_results(results, "My Agent")
```

## Converting ANN to SNN

```bash
# Convert with defaults (T=16, tau=16.0)
python main.py --mode convert --checkpoint models/ann_simple

# Convert with custom simulation steps
python main.py --mode convert --checkpoint models/ann_simple --simulation-steps 32

# Convert with custom tau
python main.py --mode convert --checkpoint models/ann_simple --snn-tau 8.0
```

### Conversion Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | `models/ann_simple` | Path to trained ANN checkpoint |
| `--simulation-steps` / `-T` | `16` | Number of timesteps per env step |
| `--snn-tau` | `16.0` | LIF neuron membrane time constant |
| `--screen-size` | `[84, 84]` | Observation resolution |
| `--use-depth` | `False` | Include depth buffer |

## Fine-tuning SNN

```bash
# Fine-tune with defaults
python main.py --mode train-snn --checkpoint models/snn_converted

# Custom learning rate and timesteps
python main.py --mode train-snn --checkpoint models/snn_converted \
    --snn-learning-rate 5e-5 \
    --total-timesteps 5000000
```

### SNN Training Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--snn-learning-rate` | `1e-4` | `1e-5` - `1e-3` | SNN fine-tuning learning rate |
| `--simulation-steps` / `-T` | `16` | `8` - `64` | Timesteps per environment step |
| `--snn-tau` | `16.0` | `4.0` - `32.0` | LIF neuron time constant |

### Tuning Guide

If the SNN performs poorly after conversion:

1. **Increase T (simulation steps)**: More timesteps = finer-grained action selection
   ```bash
   --simulation-steps 32
   ```

2. **Decrease learning rate**: Lower LR for more stable fine-tuning
   ```bash
   --snn-learning-rate 5e-5
   ```

3. **Increase tau**: Slower membrane dynamics = more stable behavior
   ```bash
   --snn-tau 32.0
   ```

4. **More timesteps**: More training data for the SNN to learn from
   ```bash
   --total-timesteps 5000000
   ```

## Experiment Tracking

### Using TensorBoard

```bash
# Start training with TensorBoard logging
python main.py --mode train-ann --scenario simple --tensorboard

# View in browser
tensorboard --logdir ./logs/simple/
```

Key metrics to monitor:
- **episode_reward_mean**: Should trend upward
- **episode_length_mean**: Should increase (agent stays alive longer)
- **policy_entropy**: Should decrease (agent becomes more decisive)
- **value_loss**: Should decrease (value estimates become more accurate)

### Using Weights & Biases (Optional)

```bash
pip install wandb
wandb login
```

```python
# In train_ann.py, add to PPO init:
import wandb
wandb.init(project="doom-snn", entity="your-team")

# SB3 has built-in WandbCallback:
from stable_baselines3.common.callbacks import WandbCallback

model = PPO(..., callback=WandbCallback(
    gradient_save_freq=1000,
    verbose=1,
))
```

## Expected Performance

### ANN Baseline (PPO)

| Scenario | Timesteps | Expected Kills/Episode | Survival Time |
|----------|-----------|----------------------|---------------|
| Simple | 5M | 5-8 kills | 300+ steps |
| DeadlyCorridor | 10M | 3-6 kills | Reach goal |
| Deathmatch | 20M | 2-5 kills | 200+ steps |

### SNN (Converted + Fine-tuned)

| Scenario | Timesteps | Expected Kill Rate vs ANN |
|----------|-----------|--------------------------|
| Simple | 2M fine-tune | 80-95% of ANN |
| DeadlyCorridor | 5M fine-tune | 70-90% of ANN |
| Deathmatch | 5M fine-tune | 60-85% of ANN |

### Computational Requirements

| Task | GPU RAM | CPU RAM | Estimated Time (RTX 3090) |
|------|---------|---------|---------------------------|
| Random agent test | 2 GB | 4 GB | ~30 seconds |
| Simple (5M steps) | 4 GB | 8 GB | ~2 hours |
| DeadlyCorridor (10M) | 4 GB | 8 GB | ~4 hours |
| Deathmatch (20M) | 6 GB | 12 GB | ~8 hours |
| SNN fine-tune (2M) | 4 GB | 8 GB | ~1 hour |
