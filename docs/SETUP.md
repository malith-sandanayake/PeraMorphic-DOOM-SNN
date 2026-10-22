# Setup Guide

Complete installation and configuration instructions.

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 20.04 / Windows 10+ / macOS | Ubuntu 22.04 LTS |
| Python | 3.10 | 3.10 - 3.11 |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB | 10 GB |
| GPU | Any (CPU only) | NVIDIA GPU with 4GB+ VRAM |

## Installation Methods

Choose one of the methods below based on your platform.

### Method 1: Linux (Native) — Recommended

```bash
# 1. Clone the repository
git clone https://github.com/your-org/PeraMorphic-DOOM-SNN.git
cd PeraMorphic-DOOM-SNN

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch (CPU or GPU version)
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install ViZDoom
pip install vizdoom

# 5. Install remaining dependencies
pip install -r requirements.txt

# 6. Run setup script (downloads scenario configs)
bash setup.sh

# 7. Verify installation
python -c "import vizdoom; import stable_baselines3; import snntorch; print('All imports OK')"
```

### Method 2: Windows (WSL2) — Recommended for Windows Users

ViZDoom on Windows requires the ZDoom engine, which is easiest to install via WSL2.

```bash
# 1. Install WSL2 (if not already installed)
wsl --install

# 2. Open Ubuntu from Windows Store or Run (Win+R → wsl)

# 3. Inside WSL2, follow Method 1 steps above
```

### Method 3: Windows (Native) — Experimental

ViZDoom has pre-built Windows wheels, but they may lag behind the latest version.

```bash
# 1. Install Python 3.10+ from python.org
# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install ViZDoom (pre-built Windows wheel)
pip install vizdoom

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Run setup script
bash setup.sh   # Works in Git Bash
```

> **Note**: Windows native installation may have compatibility issues with certain PyTorch versions. If you encounter errors, use WSL2 instead.

### Method 4: Docker — For Reproducible Environments

```bash
# Build the Docker image
docker build -t doom-snn .

# Run training inside container
docker run -it --gpus all \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    doom-snn \
    python main.py --mode train-ann --scenario simple
```

## Verifying Installation

Run the test suite to verify everything works:

```bash
# Quick import test
python -c "
import vizdoom
import stable_baselines3
import snntorch
import torch
import gymnasium
print('ViZDoom:', vizdoom.__version__)
print('Stable-Baselines3:', stable_baselines3.__version__)
print('snnTorch:', snntorch.__version__)
print('PyTorch:', torch.__version__)
print('Gymnasium:', gymnasium.__version__)
print('All packages installed successfully!')
"

# Run random agent test (requires ViZDoom to be fully functional)
python main.py --mode test-random --scenario simple --episodes 5
```

Expected output for a successful test:
```
Testing random agent on scenario 'simple'...
  Episode 5/5 | Reward: -2.00 | Kills: 0 | Steps: 100

=== Random Agent Results ===
  Episodes: 5
  Avg Reward: -1.50
  Avg Kills: 0.00
  Avg Steps: 95

Environment setup verified successfully!
```

## Troubleshooting

### "ViZDoom init failed: No such file or directory"

**Cause**: ViZDoom cannot find the WAD files (game maps).

**Fix**: Ensure you have a DOOM installation. The WAD files are typically:
- **Linux**: `/usr/share/doom2.wad` or `/usr/share/rimworld/ doom2.wad`
- **Windows (WSL2)**: Copy `DOOM2.WAD` from your Windows DOOM install to `/usr/share/`

```bash
# Find DOOM2.WAD on your system
find / -name "DOOM2.WAD" 2>/dev/null

# Copy to ViZDoom share directory
sudo cp /path/to/DOOM2.WAD /usr/share/
```

### "viZDoom initialization failed: Cannot open shared library libSDL2"

**Cause**: Missing system dependencies.

**Fix**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libsdl2-dev libboost-all-dev liblua5.3-dev libsdl2-ttf-dev

# Fedora
sudo dnf install -y SDL2-devel boost-devel lua-devel SDL2_ttf-devel
```

### "ImportError: libtorch_cuda.so: cannot open shared object file"

**Cause**: CUDA toolkit version mismatch with PyTorch.

**Fix**:
```bash
# Check your CUDA version
nvcc --version

# Reinstall PyTorch matching your CUDA version
# CUDA 12.1:
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8:
# pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### "CUDA out of memory"

**Cause**: GPU memory is insufficient for the current batch size.

**Fix**: Reduce batch size in config:
```yaml
# config/train/ppo_simple.yaml
batch_size: 32  # Reduce from 64 to 32
```

Or use CPU training (slower but reliable):
```python
python main.py --mode train-ann --scenario simple
# Model will automatically use CPU if no GPU is available
```

### "Gymnasium version conflict"

**Cause**: Another package requires an incompatible Gymnasium version.

**Fix**:
```bash
pip install "gymnasium>=0.29.0,<1.0"
pip install "stable-baselines3>=2.0.0"
```

### "ModuleNotFoundError: No module named 'cv2'"

**Cause**: OpenCV is not installed.

**Fix**:
```bash
pip install opencv-python
# Or for headless server:
pip install opencv-python-headless
```

### "ViZDoom config error: map=/doom2.wad not found"

**Cause**: ViZDoom scenario configs expect DOOM2.WAD in a specific location.

**Fix**: Set the `WAD_FILE` environment variable:
```bash
export WAD_FILE=/path/to/DOOM2.WAD
```

Or modify the scenario config files in `config/doom_cfg/`:
```
map=/full/path/to/doom2.wad
```

## Optional Dependencies

The following are optional but recommended:

| Package | Use | Install |
|---------|-----|---------|
| `wandb` | Experiment tracking | `pip install wandb` |
| `ffmpeg` | Video recording | `sudo apt install ffmpeg` (Linux) / `brew install ffmpeg` (macOS) |
| `jupyter` | Interactive notebooks | `pip install jupyter` |
