#!/usr/bin/env bash
# =============================================================================
# setup.sh — Air Piano V1 installer for Raspberry Pi 3B+
# =============================================================================
# This script:
#   1. Detects your architecture and Python version
#   2. Installs system-level dependencies
#   3. Creates a Python virtual environment
#   4. Installs the correct MediaPipe wheel for your platform
#   5. Installs remaining Python packages
#   6. Verifies the installation
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# After setup:
#   source piano_env/bin/activate
#   python main.py --audio-test
#   python main.py
# =============================================================================

set -e   # Exit on first error

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓  $1${NC}"; }
warn() { echo -e "${YELLOW}  ⚠  $1${NC}"; }
err()  { echo -e "${RED}  ✗  $1${NC}"; }
info() { echo -e "${CYAN}     $1${NC}"; }

echo ""
echo "============================================="
echo "     AIR PIANO V1  —  Setup Script"
echo "============================================="
echo ""

# ── 1. Detect environment ─────────────────────────────────────────────────────
ARCH=$(uname -m)
OS=$(uname -s)
PY_BIN=""

info "Detected architecture: $ARCH"
info "Detected OS: $OS"

# Require Linux
if [ "$OS" != "Linux" ]; then
    err "This script is for Linux (Raspberry Pi OS). Got: $OS"
    exit 1
fi

# ── 2. Find Python ────────────────────────────────────────────────────────────
for candidate in python3.11 python3.10 python3.9 python3.8 python3.7 python3; do
    if command -v "$candidate" &>/dev/null; then
        PY_BIN="$candidate"
        break
    fi
done

if [ -z "$PY_BIN" ]; then
    err "No Python 3 interpreter found."
    echo "  Install with: sudo apt install python3"
    exit 1
fi

PY_VERSION=$($PY_BIN --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
info "Python: $PY_BIN ($PY_VERSION)"

# Check minimum Python version
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 7 ]; }; then
    err "Python 3.7+ required. Found: $PY_VERSION"
    exit 1
fi

# ── 3. Determine MediaPipe strategy ──────────────────────────────────────────
echo ""
info "Determining MediaPipe compatibility..."

MP_PACKAGE=""
MP_VERSION=""

if [ "$ARCH" = "armv7l" ] || [ "$ARCH" = "armhf" ]; then
    # 32-bit ARMv7 — Raspberry Pi 3B+
    warn "32-bit ARMv7 detected (Pi 3B+). Standard mediapipe has no 32-bit wheel."
    if [ "$PY_MINOR" -le 9 ] && [ "$PY_MINOR" -ge 7 ]; then
        info "Will try mediapipe==0.8.5 (last known armv7l wheel, Python 3.7-3.9)"
        MP_PACKAGE="mediapipe==0.8.5"
    else
        warn "Python $PY_VERSION may not have a mediapipe armv7l wheel."
        info "Will attempt mediapipe==0.8.5 anyway; if it fails, tracking will be disabled."
        MP_PACKAGE="mediapipe==0.8.5"
    fi
elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    # 64-bit ARM — Raspberry Pi 4/5
    info "64-bit ARM detected (Pi 4/5 or Pi 3B+ with 64-bit OS)"
    if [ "$PY_MINOR" -ge 11 ]; then
        MP_PACKAGE="mediapipe"
    elif [ "$PY_MINOR" -ge 9 ]; then
        MP_PACKAGE="mediapipe>=0.9.0"
    else
        MP_PACKAGE="mediapipe==0.8.5"
    fi
else
    # x86_64 (development machine)
    info "x86_64 detected (development/testing machine)"
    MP_PACKAGE="mediapipe"
fi

info "MediaPipe package selection: $MP_PACKAGE"
echo ""

# ── 4. System dependencies ────────────────────────────────────────────────────
echo "Installing system dependencies..."
sudo apt-get update -q

sudo apt-get install -y \
    python3-dev \
    python3-venv \
    python3-pip \
    libatlas-base-dev \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    v4l-utils \
    libsdl2-dev \
    libsdl2-mixer-dev \
    libportaudio2 \
    libasound2-dev \
    alsa-utils \
    build-essential \
    wget \
    curl

ok "System dependencies installed"

# ── 5. Create virtual environment ─────────────────────────────────────────────
VENV_DIR="piano_env"
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at $VENV_DIR"
    read -p "  Re-create it? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment: $VENV_DIR ..."
    $PY_BIN -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

source "$VENV_DIR/bin/activate"
info "Activated: $(which python)"

# ── 6. Upgrade pip ───────────────────────────────────────────────────────────
python -m pip install --upgrade pip setuptools wheel -q
ok "pip upgraded"

# ── 7. Install core Python packages ──────────────────────────────────────────
echo ""
echo "Installing Python packages..."

# numpy first (mediapipe depends on it)
pip install "numpy>=1.21.0,<2.0.0" -q
ok "numpy installed"

# PyYAML for config parsing
pip install "PyYAML>=5.4.0" -q
ok "PyYAML installed"

# pygame for audio
pip install "pygame>=2.1.0,<3.0.0" -q
ok "pygame installed"

# OpenCV — try pip first, fall back to apt package
if pip install "opencv-python==4.8.1.78" -q 2>/dev/null; then
    ok "opencv-python installed from pip"
elif pip install "opencv-python" -q 2>/dev/null; then
    ok "opencv-python installed (latest)"
else
    warn "pip install of opencv-python failed."
    warn "Trying system opencv..."
    sudo apt-get install -y python3-opencv
    # Link system opencv into the venv via pth
    SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
    echo "/usr/lib/python3/dist-packages" > "$SITE_PKGS/system_opencv.pth"
    ok "opencv-python linked from system packages"
fi

# ── 8. Install MediaPipe ──────────────────────────────────────────────────────
echo ""
echo "Installing MediaPipe ($MP_PACKAGE)..."
if pip install "$MP_PACKAGE" -q 2>/dev/null; then
    ok "MediaPipe installed: $MP_PACKAGE"
else
    warn "MediaPipe install failed for $MP_PACKAGE"
    warn "Hand tracking will be DISABLED."
    warn "The Air Piano will still run (camera + audio + piano UI)."
    warn "To fix: check https://github.com/nicholasjconn/RPi-MediaPipe"
    warn "Or try: pip install mediapipe-rpi3  (community wheel)"
fi

# ── 9. Verify installation ────────────────────────────────────────────────────
echo ""
echo "Verifying installation..."

python -c "import cv2; print(f'  opencv: {cv2.__version__}')" && ok "OpenCV OK" || err "OpenCV FAILED"
python -c "import numpy; print(f'  numpy: {numpy.__version__}')" && ok "numpy OK" || err "numpy FAILED"
python -c "import pygame; print(f'  pygame: {pygame.__version__}')" && ok "pygame OK" || err "pygame FAILED"
python -c "import yaml; print('  PyYAML OK')" && ok "PyYAML OK" || err "PyYAML FAILED"

MP_STATUS="DISABLED"
if python -c "import mediapipe; print(f'  mediapipe: {mediapipe.__version__}')" 2>/dev/null; then
    ok "MediaPipe OK"
    MP_STATUS="OK"
else
    warn "MediaPipe NOT available — hand tracking disabled"
fi

# ── 10. Audio device check ────────────────────────────────────────────────────
echo ""
echo "Audio device check..."
if aplay -l &>/dev/null 2>&1; then
    ok "Audio device(s) found:"
    aplay -l 2>/dev/null | grep "^card" | head -5 | while read line; do
        info "  $line"
    done
else
    warn "No audio devices found via aplay."
    warn "Connect speaker/headphones. Check: aplay -l"
fi

# ── 11. Camera check ──────────────────────────────────────────────────────────
echo ""
echo "Camera check..."
if ls /dev/video* &>/dev/null 2>&1; then
    ok "Camera device(s) found:"
    ls /dev/video* | while read dev; do info "  $dev"; done
else
    warn "No /dev/video* devices found."
    warn "Connect USB webcam and run: ls /dev/video*"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "           SETUP COMPLETE"
echo "============================================="
echo ""
echo "  MediaPipe : $MP_STATUS"
echo ""
echo "  To run the Air Piano:"
echo ""
echo "    source $VENV_DIR/bin/activate"
echo "    cd $(pwd)"
echo "    python main.py --audio-test    # test audio"
echo "    python main.py --camera-test   # test camera"
echo "    python main.py --hand-test     # test hand tracking"
echo "    python main.py --piano-test    # test piano geometry"
echo "    python main.py                 # full Air Piano"
echo ""
if [ "$MP_STATUS" = "DISABLED" ]; then
    echo "  ⚠  Hand tracking unavailable."
    echo "     The piano will display but finger detection won't work."
    echo "     See README.md for manual mediapipe installation steps."
    echo ""
fi
echo "  Controls:"
echo "    Q / ESC  — Quit"
echo "    R        — Recalibrate"
echo "    M        — Toggle landmarks"
echo "    P        — Pause / Resume"
echo "    + / -    — Volume"
echo ""
