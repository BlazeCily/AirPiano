"""
piano/keyboard.py — Virtual piano keyboard geometry

Calculates the pixel rectangles for all white and black keys
dynamically from the frame dimensions. No hard-coded positions.

Keyboard: C5 D5 E5 F5 G5 A5 B5 C6  (8 white keys)
          C#5 D#5  F#5 G#5 A#5      (5 black keys)

Real piano layout (12-note octave):
  White keys: C D E F G A B C
  Black keys: between C–D, D–E, F–G, G–A, A–B
  NO black key between E–F and B–C

Black key positioning (as fraction of white key width from left edge):
  C# is at 0.6 from C's left edge
  D# is at 0.6 from D's left edge
  F# is at 0.6 from F's left edge
  G# is at 0.6 from G's left edge
  A# is at 0.6 from A's left edge
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KeyRect:
    """
    A single piano key with its screen rectangle.

    Attributes:
        name:      Note name, e.g. "C5", "C#5"
        x:         Left pixel of the key rectangle
        y:         Top pixel of the key rectangle
        w:         Width in pixels
        h:         Height in pixels
        is_black:  True for black keys
    """
    name: str
    x: int
    y: int
    w: int
    h: int
    is_black: bool

    def contains(self, px: int, py: int) -> bool:
        """Return True if pixel (px, py) falls inside this key."""
        return (
            self.x <= px < self.x + self.w
            and self.y <= py < self.y + self.h
        )

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) tuple."""
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class PianoKeyboard:
    """
    Virtual piano keyboard with realistic white/black key layout.

    The keyboard spans the full frame width and occupies the lower
    region of the frame defined by (top_y, bottom_y).

    Call build(frame_width, top_y, bottom_y) to (re)compute geometry.
    """

    # White key names in order (left → right)
    WHITE_KEY_NAMES = ["C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6"]

    # Black key data: (name, white_key_index_to_left, x_offset_fraction)
    # x_offset_fraction: how far from the LEFT edge of the white key to position
    # the black key's LEFT edge. Standard piano: ~0.6 of white key width.
    # None means no black key at that gap.
    BLACK_KEY_DATA = [
        # Gap between index 0 (C) and 1 (D) → C#
        ("C#5", 0, 0.55),
        # Gap between index 1 (D) and 2 (E) → D#
        ("D#5", 1, 0.55),
        # NO black key between E (2) and F (3)
        # Gap between index 3 (F) and 4 (G) → F#
        ("F#5", 3, 0.55),
        # Gap between index 4 (G) and 5 (A) → G#
        ("G#5", 4, 0.55),
        # Gap between index 5 (A) and 6 (B) → A#
        ("A#5", 5, 0.55),
        # NO black key between B (6) and C (7)
    ]

    def __init__(self, config: dict):
        """
        Args:
            config: The 'piano' section of config.yaml.
        """
        self._cfg = config
        self._white_keys: List[KeyRect] = []
        self._black_keys: List[KeyRect] = []
        self._all_keys: Dict[str, KeyRect] = {}
        self._frame_w = 0
        self._top_y = 0
        self._bot_y = 0
        self._built = False

    def build(self, frame_width: int, top_y: int, bottom_y: int) -> None:
        """
        Calculate key rectangles from frame dimensions.

        Args:
            frame_width: Frame width in pixels.
            top_y:       Y pixel where piano top starts.
            bottom_y:    Y pixel where piano bottom ends.
        """
        self._frame_w = frame_width
        self._top_y = top_y
        self._bot_y = bottom_y
        self._white_keys = []
        self._black_keys = []
        self._all_keys = {}

        piano_h = bottom_y - top_y
        n_white = len(self.WHITE_KEY_NAMES)

        # Each white key gets an equal share of the full frame width
        wk_w = frame_width // n_white
        # Last white key may be 1 pixel wider to fill any remainder
        remainder = frame_width - wk_w * n_white

        # Black key dimensions
        bk_w_ratio = float(self._cfg.get("black_key_width_ratio", 0.6))
        bk_h_ratio = float(self._cfg.get("black_key_height_ratio", 0.6))
        bk_w = max(8, int(wk_w * bk_w_ratio))
        bk_h = max(20, int(piano_h * bk_h_ratio))

        # ── White keys ────────────────────────────────────────────────────
        x_cursor = 0
        white_x_positions = []  # left-x of each white key
        for i, name in enumerate(self.WHITE_KEY_NAMES):
            w = wk_w + (remainder if i == n_white - 1 else 0)
            key = KeyRect(
                name=name,
                x=x_cursor,
                y=top_y,
                w=w,
                h=piano_h,
                is_black=False,
            )
            self._white_keys.append(key)
            self._all_keys[name] = key
            white_x_positions.append(x_cursor)
            x_cursor += wk_w  # all equal-width except possibly last

        # ── Black keys ────────────────────────────────────────────────────
        for name, white_idx, x_frac in self.BLACK_KEY_DATA:
            white_key_left = white_x_positions[white_idx]
            bk_x = int(white_key_left + wk_w * x_frac)
            # Center the black key so it straddles the gap
            bk_x = bk_x - bk_w // 2 + wk_w // 2
            # Clamp to frame
            bk_x = max(0, min(bk_x, frame_width - bk_w))

            key = KeyRect(
                name=name,
                x=bk_x,
                y=top_y,
                w=bk_w,
                h=bk_h,
                is_black=True,
            )
            self._black_keys.append(key)
            self._all_keys[name] = key

        self._built = True
        logger.info(
            "Keyboard built: %d white keys × %dpx, %d black keys × %dpx, "
            "piano region y=[%d, %d]",
            n_white, wk_w, len(self._black_keys), bk_w, top_y, bottom_y,
        )

    # ── Key lookup ────────────────────────────────────────────────────────

    def get_key(self, name: str) -> Optional[KeyRect]:
        return self._all_keys.get(name)

    def get_all_keys(self) -> Dict[str, KeyRect]:
        return dict(self._all_keys)

    @property
    def white_keys(self) -> List[KeyRect]:
        return list(self._white_keys)

    @property
    def black_keys(self) -> List[KeyRect]:
        return list(self._black_keys)

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def top_y(self) -> int:
        return self._top_y

    @property
    def bottom_y(self) -> int:
        return self._bot_y
