#!/bin/bash
# Setup script for PeraMorphic-DOOM-SNN
# Works in Git Bash (MinGW) on Windows or on Linux/macOS

set -e

echo "=== PeraMorphic-DOOM-SNN Setup ==="

# Check Python version
echo "Checking Python version..."
python3 --version || python --version
echo ""

# Install core dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt || pip3 install -r requirements.txt
echo ""

# Download ViZDoom scenario configurations
echo "Downloading ViZDoom scenario configs..."
SCENARIOS_DIR="config/doom_cfg"
mkdir -p "$SCENARIOS_DIR"

# Download a few key scenario configs from ViZDoom GitHub
wget -q -O "$SCENARIOS_DIR/simple.cfg" \
    "https://raw.githubusercontent.com/mwydmuch/ViZDoom/master/scenarios/simple.cfg" \
    || echo "Warning: Could not download simple.cfg - you'll need ViZDoom scenarios installed separately"

wget -q -O "$SCENARIOS_DIR/deadly_corridor.cfg" \
    "https://raw.githubusercontent.com/mwydmuch/ViZDoom/master/scenarios/deadly_corridor.cfg" \
    || echo "Warning: Could not download deadly_corridor.cfg"

wget -q -O "$SCENARIOS_DIR/deadly_terrace.cfg" \
    "https://raw.githubusercontent.com/mwydmuch/ViZDoom/master/scenarios/deadly_terrace.cfg" \
    || echo "Warning: Could not download deadly_terrace.cfg"

wget -q -O "$SCENARIOS_DIR/defend_my_position.cfg" \
    "https://raw.githubusercontent.com/mwydmuch/ViZDoom/master/scenarios/defend_my_position.cfg" \
    || echo "Warning: Could not download defend_my_position.cfg"

wget -q -O "$SCENARIOS_DIR/deathmatch.cfg" \
    "https://raw.githubusercontent.com/mwydmuch/ViZDoom/master/scenarios/deathmatch.cfg" \
    || echo "Warning: Could not download deathmatch.cfg"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Verify installations: python -c 'import vizdoom; import stable_baselines3; import snntorch; print(\"All imports OK\")'"
echo "2. Run a test: python main.py --mode test-random --scenario simple"
echo "3. Start training: python main.py --mode train-ann --scenario simple"
