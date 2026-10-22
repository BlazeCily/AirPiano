"""
ui/renderer.py — Frame renderer

Draws all visual elements onto the OpenCV frame:
  1. Piano keyboard (white and black keys, idle + active states)
  2. Hand skeleton and landmarks
  3. Fingertip highlights
  4. Active key highlights
  5. Piano region entry indicator

All drawing uses OpenCV (cv2) only — no extra GUI libraries.
Colors are read from config for easy customization.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from piano.keyboard import PianoKeyboard, KeyRect
from vision.hand_tracker import HandLandmarks, FINGERTIP_IDS, HAND_CONNECTIONS

logger = logging.getLogger(__name__)


def _bgr(color_cfg) -> Tuple[int, int, int]:
    """Convert a config color (list of 3 ints) to a BGR tuple."""
    if isinstance(color_cfg, (list, tuple)) and len(color_cfg) == 3:
        r, g, b = color_cfg
        return (int(r), int(g), int(b))
    return (128, 128, 128)


class Renderer:
    """
    Stateless renderer — call draw() each frame.

    All methods draw directly onto the provided frame in-place.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: The 'display' section of config.yaml.
        """
        cfg = config
        self._c_white_key    = _bgr(cfg.get("white_key_color",    [240, 240, 240]))
        self._c_black_key    = _bgr(cfg.get("black_key_color",    [30, 30, 30]))
        self._c_active_white = _bgr(cfg.get("active_white_color", [100, 220, 255]))
        self._c_active_black = _bgr(cfg.get("active_black_color", [80, 160, 220]))
        self._c_border       = _bgr(cfg.get("piano_border_color", [80, 80, 80]))
        self._c_landmark     = _bgr(cfg.get("landmark_color",     [0, 255, 0]))
        self._c_fingertip    = _bgr(cfg.get("fingertip_color",    [0, 200, 255]))
        self._c_connection   = _bgr(cfg.get("connection_color",   [0, 180, 0]))

    # ── Main draw entrypoint ──────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        keyboard: PianoKeyboard,
        hands: List[HandLandmarks],
        active_notes: Set[str],
        active_fingers_in_piano: Dict[str, Tuple[int, int]],  # finger_id → (px, py)
        show_landmarks: bool = True,
        debug_info: Optional[Dict] = None,
    ) -> None:
        """
        Render everything onto the frame in-place.

        Args:
            frame:                    BGR numpy array to draw on.
            keyboard:                 Built PianoKeyboard.
            hands:                    List of HandLandmarks from tracker.
            active_notes:             Set of currently playing note names.
            active_fingers_in_piano:  Dict of finger_id → pixel coords for fingers
                                      that are inside the piano region.
            show_landmarks:           Whether to draw hand skeleton.
            debug_info:               Optional dict with debug text per finger.
        """
        if keyboard.is_built:
            self._draw_keyboard(frame, keyboard, active_notes)

        if show_landmarks:
            self._draw_hands(frame, hands)

        self._draw_active_fingertips(frame, active_fingers_in_piano)

        if debug_info:
            self._draw_debug(frame, debug_info)

    # ── Keyboard drawing ──────────────────────────────────────────────────

    def _draw_keyboard(
        self,
        frame: np.ndarray,
        keyboard: PianoKeyboard,
        active_notes: Set[str],
    ) -> None:
        # ── 1. White keys ─────────────────────────────────────────────────
        for key in keyboard.white_keys:
            color = self._c_active_white if key.name in active_notes else self._c_white_key
            cv2.rectangle(frame, (key.x, key.y), (key.x + key.w, key.y + key.h),
                          color, -1)
            # Border
            cv2.rectangle(frame, (key.x, key.y), (key.x + key.w, key.y + key.h),
                          self._c_border, 1)
            # Key label (small, at bottom of key)
            self._draw_key_label(frame, key, active_notes)

        # ── 2. Black keys (drawn on top of white keys) ────────────────────
        for key in keyboard.black_keys:
            color = self._c_active_black if key.name in active_notes else self._c_black_key
            cv2.rectangle(frame, (key.x, key.y), (key.x + key.w, key.y + key.h),
                          color, -1)
            cv2.rectangle(frame, (key.x, key.y), (key.x + key.w, key.y + key.h),
                          (60, 60, 60), 1)
            # Black key label
            self._draw_key_label(frame, key, active_notes)

        # ── 3. Piano top border line ──────────────────────────────────────
        h, w = frame.shape[:2]
        cv2.line(frame, (0, keyboard.top_y), (w, keyboard.top_y),
                 (100, 100, 100), 2)

    def _draw_key_label(
        self,
        frame: np.ndarray,
        key: KeyRect,
        active_notes: Set[str],
    ) -> None:
        """Draw a small note name label on the key."""
        label = key.name
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.35
        thickness = 1

        if key.is_black:
            # Label below the black key, in white text
            tx = key.x + 2
            ty = key.y + key.h + 14
            color = (220, 220, 220)
        else:
            # Label near bottom of white key
            tx = key.x + 4
            ty = key.y + key.h - 8
            color = (50, 50, 50) if key.name not in active_notes else (20, 20, 80)

        cv2.putText(frame, label, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

    # ── Hand drawing ──────────────────────────────────────────────────────

    def _draw_hands(self, frame: np.ndarray, hands: List[HandLandmarks]) -> None:
        h, w = frame.shape[:2]
        for hand in hands:
            lms = hand.landmarks
            if len(lms) < 21:
                continue

            # Skeleton connections
            for (a, b) in HAND_CONNECTIONS:
                if a < len(lms) and b < len(lms):
                    ax, ay = int(lms[a][0] * w), int(lms[a][1] * h)
                    bx, by = int(lms[b][0] * w), int(lms[b][1] * h)
                    cv2.line(frame, (ax, ay), (bx, by), self._c_connection, 2, cv2.LINE_AA)

            # Landmarks
            for idx, (x, y, _) in enumerate(lms):
                px, py = int(x * w), int(y * h)
                if idx in FINGERTIP_IDS:
                    # Fingertip — larger, bright fill + white ring
                    cv2.circle(frame, (px, py), 11, self._c_fingertip, -1)
                    cv2.circle(frame, (px, py), 11, (255, 255, 255), 2)
                else:
                    # Ordinary joint
                    cv2.circle(frame, (px, py), 5, self._c_landmark, -1)
                    cv2.circle(frame, (px, py), 5, (200, 200, 200), 1)

    # ── Active fingertip highlights ───────────────────────────────────────

    def _draw_active_fingertips(
        self,
        frame: np.ndarray,
        active_fingers: Dict[str, Tuple[int, int]],
    ) -> None:
        """Draw a pulsing ring around fingertips that are inside the piano."""
        for finger_id, (px, py) in active_fingers.items():
            # Outer glow ring
            cv2.circle(frame, (px, py), 18, (255, 140, 0), 3, cv2.LINE_AA)
            # Inner bright dot
            cv2.circle(frame, (px, py), 7, (255, 220, 50), -1)

    # ── Debug overlay ─────────────────────────────────────────────────────

    def _draw_debug(self, frame: np.ndarray, debug_info: Dict) -> None:
        """
        Draw per-finger debug text.

        debug_info format:
            {
              "h0_lm8": {
                "label": "Left Index",
                "px": 314, "py": 402,
                "key": "E5",
                "state": "ACTIVE",
              },
              ...
            }
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 160
        for finger_id, info in debug_info.items():
            label  = info.get("label", finger_id)
            px     = info.get("px", 0)
            py     = info.get("py", 0)
            key    = info.get("key", "—")
            state  = info.get("state", "—")
            text   = f"{label}  x={px} y={py}  key={key}  [{state}]"
            cv2.putText(frame, text, (10, y), font, 0.42, (255, 200, 50), 1, cv2.LINE_AA)
            y += 18
