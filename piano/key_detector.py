"""
piano/key_detector.py — Fingertip → Key collision detection

Maps pixel coordinates to piano key names.

CRITICAL: Black keys are checked BEFORE white keys.
Black keys physically overlap the white key region, so a fingertip
over C# must resolve to "C#5", not "C5".

Usage:
    detector = KeyDetector(keyboard)
    key_name = detector.get_key_at(px, py)  # None if outside piano
"""

import logging
from typing import Optional

from piano.keyboard import PianoKeyboard, KeyRect

logger = logging.getLogger(__name__)


class KeyDetector:
    """
    Maps pixel (x, y) coordinates to a piano key name.

    Collision priority:
        1. Black keys   ← checked first (they sit on top visually)
        2. White keys
        3. None         ← fingertip is outside the piano region
    """

    def __init__(self, keyboard: PianoKeyboard):
        """
        Args:
            keyboard: Built PianoKeyboard instance.
        """
        self._kb = keyboard

    def get_key_at(self, px: int, py: int) -> Optional[str]:
        """
        Return the name of the key at pixel (px, py).

        Args:
            px: Horizontal pixel (already scaled from normalized landmark).
            py: Vertical pixel (already scaled from normalized landmark).

        Returns:
            Key name string (e.g. "C5", "C#5") or None.
        """
        # Check black keys first — they visually overlap white keys
        for key in self._kb.black_keys:
            if key.contains(px, py):
                return key.name

        # Then check white keys
        for key in self._kb.white_keys:
            if key.contains(px, py):
                return key.name

        return None

    def get_key_object_at(self, px: int, py: int) -> Optional[KeyRect]:
        """
        Return the KeyRect at (px, py), black-key-first priority.
        """
        for key in self._kb.black_keys:
            if key.contains(px, py):
                return key
        for key in self._kb.white_keys:
            if key.contains(px, py):
                return key
        return None

    def is_in_piano(self, px: int, py: int) -> bool:
        """
        Return True if (px, py) is anywhere within the piano region
        (including the area above black keys but inside the white key rectangle).
        """
        return self.get_key_at(px, py) is not None
