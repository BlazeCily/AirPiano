"""
main.py — Air Piano entry point

Raspberry Pi 3B+ Air Piano — V1
================================
Complete pipeline:
  Camera → Hand Tracking → Fingertip Detection → Piano Key Collision
  → Note State → Audio → Visual Feedback

Usage:
    python main.py                  # Full Air Piano
    python main.py --camera-test    # Camera feed only
    python main.py --hand-test      # Camera + hand landmarks
    python main.py --piano-test     # Piano geometry test (no hands needed)
    python main.py --audio-test     # Plays all 13 notes in sequence
    python main.py --debug          # Show per-finger debug overlay

Controls:
    Q / ESC  → Quit
    R        → Recalibrate piano region
    M        → Toggle hand landmark display
    P        → Pause / Resume

Keyboard shortcuts:
    + / =    → Volume up
    -        → Volume down
"""

import argparse
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── Windows terminal UTF-8 fix ────────────────────────────────────────────────
# Prevents UnicodeEncodeError on Windows CP1252 terminals
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow imports from the air_piano package root regardless of cwd
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Config ────────────────────────────────────────────────────────────────────
def _load_config(path: str = "config.yaml") -> dict:
    """Load YAML config. Falls back to minimal defaults if PyYAML missing."""
    config_path = Path(_HERE) / path
    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    except ImportError:
        print("WARNING: PyYAML not installed. Using built-in defaults.")
        return {}
    except FileNotFoundError:
        print(f"WARNING: {config_path} not found. Using defaults.")
        return {}


def _get(cfg: dict, *keys, default=None):
    """Safe nested config lookup."""
    d = cfg
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d if d is not None else default


# ── Logging setup ─────────────────────────────────────────────────────────────
def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)
    # Suppress mediapipe's own verbose logging unless in debug mode
    if not debug:
        logging.getLogger("mediapipe").setLevel(logging.WARNING)
        logging.getLogger("absl").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ── Startup banner ────────────────────────────────────────────────────────────
def _print_banner(cfg: dict, tracker_ok: bool, audio_ok: bool, camera_ok: bool) -> None:
    w = _get(cfg, "camera", "width", default=640)
    h = _get(cfg, "camera", "height", default=480)
    sr = _get(cfg, "audio", "sample_rate", default=22050)

    print("")
    print("=" * 40)
    print("           AIR PIANO  V1")
    print("=" * 40)
    print(f"  Platform  : {platform.machine()} / {platform.system()}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  Camera    : {w}x{h}  {'OK' if camera_ok else 'DISCONNECTED'}")
    print(f"  Tracking  : {'OK' if tracker_ok else 'UNAVAILABLE (mediapipe missing?)'}")
    print(f"  Audio     : {sr} Hz  {'OK' if audio_ok else 'UNAVAILABLE'}")
    print("=" * 40)
    print("")


# ── Test modes ────────────────────────────────────────────────────────────────

def run_camera_test(cfg: dict) -> None:
    """Show camera feed only. Press Q to quit."""
    import cv2
    from vision.camera import Camera

    print("\n[CAMERA TEST] Press Q to quit\n")
    cam = Camera(_get(cfg, "camera", default={}))
    if not cam.open():
        print("ERROR: Could not open camera.")
        return

    from utils.fps import FPSCounter
    fps_counter = FPSCounter()

    while True:
        ok, frame = cam.read()
        if not ok:
            print("Camera disconnected.")
            break

        fps_counter.tick()
        fps = fps_counter.get()
        cv2.putText(frame, f"Camera: OK  FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        cv2.putText(frame, "Q to quit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.namedWindow("Air Piano — Camera Test", cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty("Air Piano — Camera Test", cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("Air Piano — Camera Test", frame)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            break

    cam.release()
    cv2.destroyAllWindows()


def run_hand_test(cfg: dict) -> None:
    """Show camera + hand landmarks. Press Q to quit."""
    import cv2
    from vision.camera import Camera
    from vision.hand_tracker import HandTracker
    from utils.fps import FPSCounter

    print("\n[HAND TEST] Press Q to quit\n")
    cam = Camera(_get(cfg, "camera", default={}))
    tracker = HandTracker(_get(cfg, "vision", default={}))

    if not cam.open():
        print("ERROR: Could not open camera.")
        return

    ok = tracker.initialize()
    if not ok:
        print("WARNING: Hand tracking unavailable. Showing camera only.")

    fps_counter = FPSCounter()

    while True:
        ok_frame, frame = cam.read()
        if not ok_frame:
            break

        hands = tracker.process(frame) if tracker.is_available else []
        tracker.draw_landmarks(frame, hands)
        fps_counter.tick()

        cv2.putText(frame, f"Hands: {len(hands)}  FPS: {fps_counter.get_int()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        cv2.putText(frame, "Q to quit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.namedWindow("Air Piano — Hand Test", cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty("Air Piano — Hand Test", cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("Air Piano — Hand Test", frame)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            break

    tracker.release()
    cam.release()
    cv2.destroyAllWindows()


def run_piano_test(cfg: dict) -> None:
    """Show piano key geometry with click-to-test collision. Press Q to quit."""
    import cv2
    import numpy as np
    from piano.keyboard import PianoKeyboard
    from piano.key_detector import KeyDetector
    from vision.calibration import Calibration

    print("\n[PIANO TEST] Move mouse over keys to test collision. Q to quit\n")

    w = _get(cfg, "camera", "width", default=640)
    h = _get(cfg, "camera", "height", default=480)
    piano_cfg = _get(cfg, "piano", default={})
    disp_cfg  = _get(cfg, "display", default={})

    calibration = Calibration(piano_cfg, h, w)
    keyboard = PianoKeyboard(piano_cfg)
    keyboard.build(w, calibration.top_y, calibration.bottom_y)
    detector = KeyDetector(keyboard)

    from ui.renderer import Renderer
    renderer = Renderer(disp_cfg)

    # Mouse tracking
    mouse_pos = [0, 0]
    def on_mouse(event, x, y, flags, param):
        mouse_pos[0], mouse_pos[1] = x, y
    cv2.namedWindow("Air Piano — Piano Test")
    cv2.setMouseCallback("Air Piano — Piano Test", on_mouse)

    while True:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)

        mx, my = mouse_pos
        hovered_key = detector.get_key_at(mx, my)
        active = {hovered_key} if hovered_key else set()

        renderer.draw(frame, keyboard, [], active, {}, show_landmarks=False)

        # Mouse crosshair
        cv2.line(frame, (mx - 12, my), (mx + 12, my), (0, 255, 255), 1)
        cv2.line(frame, (mx, my - 12), (mx, my + 12), (0, 255, 255), 1)

        status = f"Hover: {hovered_key or '—'}  ({mx}, {my})"
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 220, 0), 1)
        cv2.putText(frame, "Q to quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1)

        cv2.imshow("Air Piano — Piano Test", frame)
        if (cv2.waitKey(16) & 0xFF) in (ord('q'), 27):
            break

    cv2.destroyAllWindows()


def run_audio_test(cfg: dict) -> None:
    """Play all 13 chromatic notes in sequence."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    from audio.engine import AudioEngine
    from audio.synth import get_all_note_names

    print("\n[AUDIO TEST] Playing all notes C5 -> C6\n")
    engine = AudioEngine(_get(cfg, "audio", default={}))
    if not engine.initialize():
        print("ERROR: Audio engine failed to initialize.")
        print("  Check: speaker/headphones connected?")
        print("  Try: aplay -l  (to list audio devices)")
        return

    notes = get_all_note_names()
    for note in notes:
        print(f"  Playing: {note}")
        engine.play_note(note)
        time.sleep(0.6)
        engine.stop_note(note)
        time.sleep(0.1)

    print("  Done!")
    time.sleep(0.5)
    engine.shutdown()


# ── Main Air Piano loop ────────────────────────────────────────────────────────

def run_air_piano(cfg: dict, debug: bool = False) -> None:
    """
    Full Air Piano — the complete pipeline.
    """
    import cv2
    import numpy as np
    from vision.camera import Camera
    from vision.hand_tracker import HandTracker, FINGERTIP_IDS
    from vision.calibration import Calibration
    from piano.keyboard import PianoKeyboard
    from piano.key_detector import KeyDetector
    from piano.note_state import NoteStateManager, NoteEvent
    from audio.engine import AudioEngine
    from ui.renderer import Renderer
    from ui.hud import HUD
    from utils.fps import FPSCounter

    piano_cfg = _get(cfg, "piano", default={})
    camera_cfg = _get(cfg, "camera", default={})
    vision_cfg = _get(cfg, "vision", default={})
    audio_cfg  = _get(cfg, "audio", default={})
    disp_cfg   = _get(cfg, "display", default={})
    # inter_cfg reserved for future use (debounce_ms read directly above)

    debounce_ms = int(_get(cfg, "interaction", "debounce_ms", default=150))
    volume      = float(_get(cfg, "audio", "volume", default=0.8))
    VOLUME_STEP = 0.05

    # ── Initialize subsystems ────────────────────────────────────────────
    cam      = Camera(camera_cfg)
    tracker  = HandTracker(vision_cfg)
    engine   = AudioEngine(audio_cfg)

    camera_ok  = cam.open()
    tracker_ok = tracker.initialize()
    audio_ok   = engine.initialize()

    _print_banner(cfg, tracker_ok, audio_ok, camera_ok)

    if not camera_ok:
        print("Cannot start without camera. Exiting.")
        engine.shutdown()
        tracker.release()
        return

    # Determine actual frame size (camera may not honour requested size)
    frame_w = cam.width
    frame_h = cam.height

    calibration = Calibration(piano_cfg, frame_h, frame_w)
    keyboard    = PianoKeyboard(piano_cfg)
    keyboard.build(frame_w, calibration.top_y, calibration.bottom_y)

    detector    = KeyDetector(keyboard)
    state_mgr   = NoteStateManager(debounce_ms=debounce_ms)
    renderer    = Renderer(disp_cfg)
    hud         = HUD(disp_cfg)
    fps_counter = FPSCounter(window=30)

    # State flags
    show_landmarks = bool(_get(cfg, "display", "show_landmarks", default=True))
    paused   = False
    mode_text: Optional[str] = None
    mediapipe_ver = tracker.mediapipe_version

    cv2.namedWindow("Air Piano", cv2.WINDOW_NORMAL)
    try:
        # Bring window to front on launch
        cv2.setWindowProperty("Air Piano", cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    print("Starting Air Piano. Press Q or ESC to quit.\n")

    # ── Main loop ────────────────────────────────────────────────────────
    while True:
        # ── Key handling ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):          # Q / ESC
            break
        elif key == ord('p'):              # P — pause / resume
            paused = not paused
            mode_text = "PAUSED" if paused else None
            if paused:
                events = state_mgr.release_all()
                for ev in events:
                    if ev.event == NoteEvent.NOTE_OFF:
                        engine.stop_note(ev.note)
        elif key == ord('m'):              # M — toggle landmarks
            show_landmarks = not show_landmarks
        elif key == ord('r'):              # R — recalibrate
            calibration.run_calibration(cam.read)
            keyboard.build(frame_w, calibration.top_y, calibration.bottom_y)
            mode_text = None
        elif key in (ord('+'), ord('=')):  # + — volume up
            volume = min(1.0, volume + VOLUME_STEP)
            engine.set_volume(volume)
        elif key == ord('-'):              # - — volume down
            volume = max(0.0, volume - VOLUME_STEP)
            engine.set_volume(volume)

        # ── Frame capture ─────────────────────────────────────────────────
        ok, frame = cam.read()
        if not ok or frame is None:
            # Camera dropped — show error frame
            err_frame = _make_error_frame(frame_w, frame_h)
            cv2.imshow("Air Piano", err_frame)
            continue

        if paused:
            _draw_paused(frame)
            hud.draw(frame, fps_counter.get(), 0, cam.is_connected(),
                     audio_ok, tracker_ok, engine.active_notes,
                     mode_text="PAUSED")
            cv2.imshow("Air Piano", frame)
            fps_counter.tick()
            continue

        # ── Hand tracking ─────────────────────────────────────────────────
        hands: List = []
        if tracker_ok:
            try:
                hands = tracker.process(frame)
            except Exception as exc:
                logger.warning("Hand tracking error: %s", exc)

        # ── Fingertip → Key mapping ───────────────────────────────────────
        finger_keys = []            # [(finger_id, key_or_None), ...]
        active_fingers_in_piano = {}  # finger_id → (px, py) for visual highlight
        debug_info = {} if debug else None

        for hand_idx, hand in enumerate(hands):
            for fid, nx, ny, nz in hand.fingertips():
                px = int(nx * frame_w)
                py = int(ny * frame_h)
                finger_id = f"h{hand_idx}_lm{fid}"

                # Collision detect
                key_name = detector.get_key_at(px, py) if calibration.is_in_piano_region(px, py) else None
                finger_keys.append((finger_id, key_name))

                if key_name is not None:
                    active_fingers_in_piano[finger_id] = (px, py)

                # Debug info
                if debug_info is not None:
                    hand_label = f"{hand.handedness} {_FINGER_NAMES.get(fid, str(fid))}"
                    debug_info[finger_id] = {
                        "label": hand_label,
                        "px": px, "py": py,
                        "key": key_name or "—",
                        "state": "ACTIVE" if key_name else "—",
                    }

        # ── Note state machine ────────────────────────────────────────────
        events = state_mgr.update(finger_keys)
        for ev in events:
            if ev.event == NoteEvent.NOTE_ON:
                engine.play_note(ev.note)
            elif ev.event == NoteEvent.NOTE_OFF:
                engine.stop_note(ev.note)

        active_notes = engine.active_notes

        # ── Render ───────────────────────────────────────────────────────
        renderer.draw(
            frame, keyboard, hands, active_notes,
            active_fingers_in_piano,
            show_landmarks=show_landmarks,
            debug_info=debug_info,
        )
        hud.draw(
            frame,
            fps=fps_counter.get(),
            n_hands=len(hands),
            camera_ok=cam.is_connected(),
            audio_ok=audio_ok,
            tracker_ok=tracker_ok,
            active_notes=active_notes,
            mode_text=mode_text,
        )
        hud.draw_controls_hint(frame)

        fps_counter.tick()
        cv2.imshow("Air Piano", frame)

    # ── Shutdown ─────────────────────────────────────────────────────────
    print("\nShutting down …")
    events = state_mgr.release_all()
    for ev in events:
        if ev.event == NoteEvent.NOTE_OFF:
            engine.stop_note(ev.note)
    time.sleep(0.1)
    engine.shutdown()
    tracker.release()
    cam.release()
    cv2.destroyAllWindows()
    print("Goodbye!")


# ── Helpers ───────────────────────────────────────────────────────────────────

_FINGER_NAMES = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}


def _make_error_frame(w: int, h: int):
    """Create a dark error frame when the camera is unavailable."""
    import cv2
    import numpy as np
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (20, 20, 50)
    cv2.putText(frame, "Camera: DISCONNECTED", (w // 2 - 160, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)
    cv2.putText(frame, "Check USB cable / IP address  |  Q to quit",
                (w // 2 - 190, h // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return frame


def _draw_paused(frame) -> None:
    """Dim the frame when paused."""
    import cv2
    import numpy as np
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)


# ── Argument parser ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Air Piano — Raspberry Pi 3B+ gesture instrument"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--camera-test", action="store_true",
                      help="Show camera feed only")
    mode.add_argument("--hand-test", action="store_true",
                      help="Show camera + hand landmarks")
    mode.add_argument("--piano-test", action="store_true",
                      help="Test piano key collision geometry (mouse hover)")
    mode.add_argument("--audio-test", action="store_true",
                      help="Play all 13 notes in sequence")
    p.add_argument("--debug", action="store_true",
                   help="Show per-finger debug overlay")
    p.add_argument("--config", default="config.yaml",
                   help="Path to config file (default: config.yaml)")
    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    _setup_logging(args.debug)

    cfg = _load_config(args.config)

    if args.camera_test:
        run_camera_test(cfg)
    elif args.hand_test:
        run_hand_test(cfg)
    elif args.piano_test:
        run_piano_test(cfg)
    elif args.audio_test:
        run_audio_test(cfg)
    else:
        run_air_piano(cfg, debug=args.debug)
