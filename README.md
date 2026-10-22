# 🎹 Air Piano V1 — Raspberry Pi 3B+

A real-time, locally-running Air Piano controlled by hand gestures.  
No cloud. No browser. No physical touch.  
Runs on **Raspberry Pi 3B+ with 32-bit Raspberry Pi OS**.

---

## Hardware

| Component | Specification |
|-----------|---------------|
| **SBC** | Raspberry Pi 3 Model B+ |
| **OS** | Raspberry Pi OS (32-bit, Bullseye or Bookworm) |
| **Camera** | USB webcam (640×480) **or** Android phone via IP Webcam |
| **Display** | HDMI monitor |
| **Audio** | 3.5mm speaker / headphones, or HDMI audio out |

---

## Software Requirements

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.7 – 3.9 | 3.9 recommended for Pi 3B+ |
| opencv-python | 4.8.x | or system `python3-opencv` |
| mediapipe | 0.8.5 | **last 32-bit armv7l wheel** |
| numpy | 1.21 – 1.26 | required by mediapipe 0.8.x |
| pygame | 2.x | audio + display |
| PyYAML | 5.4+ | config parsing |

> ⚠️ **MediaPipe 32-bit Warning**: The official `mediapipe` PyPI package  
> dropped 32-bit (armv7l) wheels after version 0.8.5. Do not run  
> `pip install mediapipe` without specifying the version — it will fail.  
> `setup.sh` handles this automatically.

---

## Installation

### Step 1 — Clone the project

```bash
git clone <your-repo-url> ~/air_piano
cd ~/air_piano
```

### Step 2 — Run the setup script

```bash
chmod +x setup.sh
./setup.sh
```

The script will:
- Detect your architecture (ARMv7 32-bit vs ARM64)
- Install system dependencies via `apt`
- Create a Python virtual environment (`piano_env/`)
- Install `mediapipe==0.8.5` (the correct 32-bit wheel)
- Verify the installation

Expected output:
```
  ✓  System dependencies installed
  ✓  numpy OK
  ✓  pygame OK
  ✓  OpenCV OK
  ✓  MediaPipe OK
```

### Step 3 — Activate the environment

```bash
source piano_env/bin/activate
```

### Step 4 — Test each subsystem

```bash
# Test camera
python main.py --camera-test

# Test hand tracking (make sure mediapipe installed)
python main.py --hand-test

# Test piano geometry (move mouse over keys)
python main.py --piano-test

# Test audio (plays all 13 notes)
python main.py --audio-test

# Full Air Piano
python main.py
```

---

## Camera Setup

### Option A — USB Webcam (recommended)

1. Plug in your USB webcam
2. Verify it is detected:

```bash
ls /dev/video*
# Should show: /dev/video0
```

3. Test it with:

```bash
python main.py --camera-test
```

The default config uses `/dev/video0` (device index 0).  
If you have multiple cameras, edit `config.yaml`:

```yaml
camera:
  mode: usb
  device: 1       # change to 1, 2, etc.
```

### Option B — Android Phone (IP Webcam)

1. Install **IP Webcam** from Google Play Store (free)
2. Open the app → scroll to bottom → tap **Start Server**
3. Note the IP address shown (e.g. `192.168.1.5`)
4. Edit `config.yaml`:

```yaml
camera:
  mode: ip
  ip_url: "http://192.168.1.5:8080/video"
```

5. Make sure your Pi and phone are on the **same WiFi network**
6. Test:

```bash
python main.py --camera-test
```

---

## Audio Setup

The Air Piano generates audio using pure numpy synthesis — no samples needed.

### Check audio devices

```bash
aplay -l
```

Expected output:
```
card 0: Headphones [bcm2835 Headphones], device 0: bcm2835 Headphones [bcm2835 Headphones]
```

### 3.5mm headphones/speakers

Plug into the 3.5mm jack on the Pi.  
If no sound, set the audio output:

```bash
# Force 3.5mm output (not HDMI)
sudo raspi-config
# → System Options → Audio → 3.5mm
```

Or:

```bash
amixer cset numid=3 1   # 1 = 3.5mm, 2 = HDMI
```

### HDMI audio

Your HDMI monitor must support audio.

```bash
amixer cset numid=3 2   # force HDMI audio
```

---

## Running

```bash
source piano_env/bin/activate
cd ~/air_piano
python main.py
```

### Startup output

```
========================================
           AIR PIANO  V1
========================================
  Platform  : armv7l / Linux
  Python    : 3.9.2
  Camera    : 640x480  OK
  Tracking  : OK
  Audio     : 22050 Hz  OK
========================================
```

---

## Controls

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit cleanly |
| `R` | Recalibrate piano region |
| `M` | Toggle hand landmark display |
| `P` | Pause / Resume |
| `+` / `=` | Volume up |
| `-` | Volume down |

---

## Piano Layout

The keyboard spans the bottom **~38%** of the camera frame:

```
| C | D | E | F | G | A | B | C |    ← White keys (C5–C6)
   C#  D#     F#  G#  A#             ← Black keys
```

All 13 chromatic notes: C5 C#5 D5 D#5 E5 F5 F#5 G5 G#5 A5 A#5 B5 C6

**Playing technique**: Move your fingertips into the piano region at the  
bottom of the screen. Keep fingers slightly angled downward toward the keys.

---

## Test Modes

```bash
# Camera feed only — verify camera works
python main.py --camera-test

# Hand skeleton — verify mediapipe detects your hands
python main.py --hand-test

# Piano geometry — hover mouse over keys to test collision
python main.py --piano-test

# Audio — plays all 13 notes in sequence
python main.py --audio-test

# Debug overlay — shows finger IDs, coords, key, state
python main.py --debug
```

---

## Performance Tuning

The Air Piano targets **12–20 FPS** on Pi 3B+.

If FPS is too low:

### Reduce detection frequency
Edit `config.yaml`:
```yaml
vision:
  detection_confidence: 0.4    # lower = faster (less accurate)
  tracking_confidence: 0.4
```

### Reduce resolution
```yaml
camera:
  width: 480
  height: 360
```

### Reduce max hands (if playing solo)
```yaml
vision:
  max_hands: 1
```

### Check CPU usage while running
```bash
# In another terminal:
top
# or
htop
```

Expected CPU on Pi 3B+: 60–85% on one core.

---

## Troubleshooting

### "Camera /dev/video0 was not found"
```bash
ls /dev/video*          # check devices
v4l2-ctl --list-devices # detailed device info
```
Make sure webcam is plugged in. Try `device: 1` in config if multiple cameras.

### "Hand tracking: UNAVAILABLE"
```bash
source piano_env/bin/activate
pip install mediapipe==0.8.5
python -c "import mediapipe; print(mediapipe.__version__)"
```
On 32-bit Pi 3B+ only `0.8.5` has armv7l wheels. If that fails:
```bash
pip install mediapipe-rpi3  # community build
```

### "No audio" / silence
```bash
aplay -l                     # list audio devices
speaker-test -c 1 -t sine    # test speaker
amixer cset numid=3 1        # force 3.5mm output
```

### Notes retrigger rapidly (music machine-gun effect)
This is a bug — report it. The state machine in `piano/note_state.py`  
should prevent retriggers. Check `debounce_ms` in `config.yaml`.

### FPS very low (< 8)
Lower confidence thresholds and/or resolution (see Performance Tuning above).  
On Pi 3B+ with USB webcam at 640×480, expect 12–18 FPS.

### Piano feels backwards (left↔right mirrored)
The mirror mode should already handle this. Verify in `config.yaml`:
```yaml
camera:
  flip: true   # must be true for mirror mode
```

---

## Architecture

```
air_piano/
├── main.py              Entry point, main loop, test modes
├── config.yaml          All settings (no hardcoded values)
├── requirements.txt     Pinned Pi-compatible dependencies
├── setup.sh             Automated setup + mediapipe detection
│
├── vision/
│   ├── camera.py        USB + IP camera abstraction
│   ├── hand_tracker.py  MediaPipe shim + landmark extraction
│   └── calibration.py   Piano region definition
│
├── piano/
│   ├── keyboard.py      Key geometry (dynamic, frame-relative)
│   ├── key_detector.py  Fingertip→key collision (black-key-first)
│   └── note_state.py    Per-finger state machine (no retrigger)
│
├── audio/
│   ├── engine.py        Polyphonic pygame audio engine
│   └── synth.py         Additive sine synthesis (numpy)
│
├── ui/
│   ├── renderer.py      OpenCV keyboard + hand drawing
│   └── hud.py           Status overlay (FPS, notes, camera)
│
└── utils/
    └── fps.py           Rolling FPS counter
```

---

## V2 Roadmap (not implemented in V1)

- Musical scales (Major, Minor, Pentatonic, Blues, ...)
- Root key selection
- Octave switching
- Sustain pedal mode
- Recording and playback
- MIDI output
- Multi-language note labels

---

## License

MIT — see LICENSE file.
