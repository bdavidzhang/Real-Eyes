#!/bin/bash
# dev_setup.sh — Full development environment setup for VGGT-SLAM 2.0
# Creates a Python 3.11 venv, installs all deps, and configures Modal.
#
# Usage: chmod +x dev_setup.sh && ./dev_setup.sh

set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ── Colors ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
info() { echo -e "${YELLOW}→ $*${NC}"; }
err()  { echo -e "${RED}✗ $*${NC}"; exit 1; }

echo "=================================================="
echo "  VGGT-SLAM 2.0 Development Setup"
echo "=================================================="

# ── 1. Xcode license ──
info "Accepting Xcode license..."
sudo xcodebuild -license accept 2>/dev/null && ok "Xcode license accepted" || ok "Xcode license already accepted"

# ── 2. Python 3.11 via Homebrew ──
PYTHON311=/opt/homebrew/opt/python@3.11/bin/python3.11
if [ ! -f "$PYTHON311" ]; then
    info "Installing Python 3.11 via Homebrew..."
    brew install python@3.11
fi
"$PYTHON311" --version | grep -q "3.11" || err "Python 3.11 not found at $PYTHON311"
ok "Python $("$PYTHON311" --version)"

# ── 3. Create fresh venv ──
info "Creating Python 3.11 venv at .venv..."
rm -rf .venv
"$PYTHON311" -m venv .venv
source .venv/bin/activate
ok "Venv activated: $(python --version)"

# ── 4. Core requirements ──
info "Installing requirements.txt..."
pip install -r requirements.txt --quiet
ok "Core requirements installed"

# ── 5. Third-party repos ──
mkdir -p third_party

clone_if_missing() {
    local dir="$1" url="$2" name="$3"
    if [ -d "third_party/$dir" ]; then
        ok "$name already cloned"
    else
        info "Cloning $name..."
        git clone "$url" "third_party/$dir"
        ok "$name cloned"
    fi
}

clone_if_missing salad            "https://github.com/Dominic101/salad.git"                   "Salad"
clone_if_missing vggt             "https://github.com/MIT-SPARK/VGGT_SPARK.git"               "VGGT_SPARK"
clone_if_missing perception_models "https://github.com/facebookresearch/perception_models.git" "Perception Encoder"
clone_if_missing sam3             "https://github.com/facebookresearch/sam3.git"              "SAM3"

info "Installing third-party packages..."
pip install -e ./third_party/salad --quiet
pip install -e ./third_party/vggt --quiet
pip install -e ./third_party/sam3 --quiet
# perception_models has decord==0.6.0 which doesn't support Python 3.11;
# use --no-deps since decord is only needed for video benchmark scripts we don't use.
pip install -e ./third_party/perception_models --no-deps --quiet
ok "Third-party packages installed"

# ── 6. Main package ──
info "Installing vggt_slam in editable mode..."
pip install -e . --quiet
ok "vggt_slam installed"

# ── 7. Server / streaming deps ──
info "Installing server dependencies..."
pip install modal flask flask-socketio flask-cors python-dotenv 'google-generativeai' --quiet
ok "Server deps installed"

# ── 8. Modal auth ──
echo ""
info "Setting up Modal authentication..."
if modal token list 2>/dev/null | grep -q "jalenlu"; then
    ok "Modal already authenticated"
else
    echo "  Opening browser for Modal login..."
    modal token new
fi

# ── 9. Sanity check ──
echo ""
info "Running import check..."
python -c "
import vggt_slam, torch, cv2, flask_socketio, modal
print('  vggt_slam  ✓')
print('  torch', torch.__version__, ' ✓')
print('  cv2        ✓')
print('  flask_socketio ✓')
print('  modal      ✓')
"
ok "All imports OK"

# ── Done ──
echo ""
echo "=================================================="
echo -e "${GREEN}  Setup complete!${NC}"
echo "=================================================="
echo ""
echo "  Activate environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  Run batch SLAM (local):"
echo "    python main.py --image_folder office_loop --max_loops 1 --vis_map"
echo ""
echo "  Run streaming server:"
echo "    python -m server.app --video office_loop.mp4 --fast"
echo ""
echo "  Run on Modal GPU:"
echo "    modal run modal_app.py --image-folder ./office_loop --submap-size 16 --max-loops 1"
echo ""
