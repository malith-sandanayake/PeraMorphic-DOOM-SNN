# Contributing Guide

Guidelines for contributing to the PeraMorphic-DOOM-SNN project.

## Code Style

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) with 4-space indentation
- Maximum line length: 88 characters (Black formatter default)
- Use type hints for all function signatures
- Use docstrings for all public classes and functions

### Docstring Style

Use Google-style docstrings:

```python
def train_agent(
    model: DoomNet,
    env: DoomEnv,
    total_timesteps: int = 5_000_000,
) -> PPO:
    """Train a PPO agent on ViZDoom.

    Args:
        model: The neural network to train
        env: The Gymnasium environment
        total_timesteps: Total environment steps for training

    Returns:
        Trained PPO model

    Raises:
        ValueError: If total_timesteps is negative
    """
```

### Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| Modules | `snake_case` | `vizdoom_env.py` |
| Classes | `PascalCase` | `DoomEnv`, `LIFNode` |
| Functions | `snake_case` | `compute_reward()`, `resize_frame()` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_ACTIONS`, `CONFIG_DIR` |
| Variables | `snake_case` | `episode_reward`, `frame_stack` |

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples**:
```
feat(env): add DoomStackedEnv wrapper
fix(snn): correct LIF voltage reset logic
docs(training): add SNN fine-tuning guide
test(models): add SpikingCNN shape tests
```

## Development Workflow

### 1. Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/your-org/PeraMorphic-DOOM-SNN.git
cd PeraMorphic-DOOM-SNN

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### 2. Create a Feature Branch

```bash
git checkout -b feat/your-feature-name
```

### 3. Write Code and Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/models/test_snn_cnn.py -v

# Check code style
flake8 env/ models/ training/
mypy models/ env/
```

### 4. Commit Changes

```bash
git add <files>
git commit -m "feat(snn): add recurrent SNN support"
```

### 5. Push and Create Pull Request

```bash
git push origin feat/your-feature-name

# Then create a PR on GitHub
```

## Adding a New Scenario

1. Create a new `.cfg` file in `config/doom_cfg/`:

```ini
# config/doom_cfg/my_scenario.cfg
map=/doom2.wad
videoFormat=RGB24
screenResolution=SCREEN_RESOLUTION.RES_64X48
# ... rest of config
```

2. Add to `env/vizdoom_env.py`:

```python
# Add to DEFAULT_ACTIONS or create SCENARIO_ACTIONS dict
SCENARIO_ACTIONS = {
    "simple": DEFAULT_ACTIONS,
    "deathmatch": DEATHMATCH_ACTIONS,
    "my_scenario": MY_SCENARIO_ACTIONS,  # New
}
```

3. Update `training/train_ann.py`:

```python
# Add to get_scenario_info
configs = {
    "simple": {...},
    "my_scenario": {
        "n_actions": 12,
        "timesteps": 5_000_000,
        "config": "config/train/ppo_my_scenario.yaml",
    },
}
```

4. Create `config/train/ppo_my_scenario.yaml`.

5. Update documentation in `docs/SETUP.md` and `docs/TRAINING.md`.

## Adding a New Model Architecture

1. Create the model class in `models/ann/` or `models/snn/`

2. Include unit tests in `tests/models/`

3. Update `docs/API.md` with the new class

4. Add usage example to `docs/TRAINING.md` if relevant

5. Update `README.md` architecture section if it's a major addition

## Testing

### Test Structure

```
tests/
├── test_env/
│   └── test_vizdoom_env.py
├── test_models/
│   ├── test_doom_net.py
│   ├── test_lif_neuron.py
│   └── test_snn_cnn.py
├── test_utils/
│   └── test_preprocess.py
└── test_training/
    └── test_convert.py
```

### Writing Tests

```python
# tests/test_models/test_lif_neuron.py
import torch
import pytest
from models.snn.lif_neuron import LIFNode

def test_lif_neuron_forward():
    """LIF neuron should spike when voltage exceeds threshold."""
    neuron = LIFNode(tau=16.0, leak=1.0, V_thresh=1.0)
    current = torch.ones(1, 10) * 0.3
    voltage, hidden = neuron.init_hidden(current)

    spikes = []
    for t in range(10):
        v, (sp, hidden) = neuron(current[:, t:t+1], **hidden)
        spikes.append(sp.sum().item())

    assert spikes[3] >= 1.0, "Should spike at t=3"
    assert all(s <= 1.0 for s in spikes), "Max spike count should be 1.0"

def test_lif_neuron_gradient():
    """LIF neuron should support gradient flow via surrogate gradient."""
    neuron = LIFNode(tau=16.0, leak=1.0, V_thresh=1.0)
    current = torch.ones(1, 10) * 0.5
    current.requires_grad_(True)
    voltage, hidden = neuron.init_hidden(current)

    total_spikes = 0
    for t in range(10):
        v, (sp, hidden) = neuron(current[:, t:t+1], **hidden)
        total_spikes += sp.sum()

    loss = total_spikes
    loss.backward()
    assert current.grad is not None, "Gradient should flow through LIF"
```

## Documentation Checklist

Before committing documentation changes:

- [ ] All new public APIs documented in `docs/API.md`
- [ ] Setup instructions tested on a clean environment
- [ ] Training examples runnable as-is
- [ ] Links checked and valid
- [ ] Code blocks formatted with correct language tags
- [ ] No broken internal links

## Review Process

1. **Self-review**: Check your own code against the style guide
2. **CI checks**: Ensure all tests pass and linters are happy
3. **Peer review**: At least one other team member reviews
4. **Acceptance criteria**: PR meets the described requirements
5. **Merge**: Squash and merge to main
