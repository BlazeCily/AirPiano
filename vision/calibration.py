"""
vision/calibration.py — Piano region calibration

Provides a simple calibration helper that lets the user define
where the piano region sits in the camera frame.

V1 design: the piano region is defined as a vertical fraction of
the frame (top_fraction … bottom_fraction), not a complex polygon.
This is enough to handle normal camera angles and positions.

Press R in the main app to enter calibration mode.
"""

import logging
import time
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Calibration:
    """
    Manages the piano region definition.

    The piano region is represented as:
        (top_y, bottom_y) — pixel rows in the current frame.

    These values are computed from fraction parameters in config.yaml
    and can be overridden by the user via calibration mode.
    """

    def __init__(self, config: dict, frame_height: int, frame_width: int):
        """
        Args:
            config: The 'piano' section of config.yaml.
            frame_height: Camera frame height in pixels.
            frame_width:  Camera frame width in pixels.
        """
        self._cfg = config
        self._frame_h = frame_height
        self._frame_w = frame_width

        top_frac = float(config.get("top_fraction", 0.62))
        bot_frac = float(config.get("bottom_fraction", 1.0))

        self._top_y = int(top_frac * frame_height)
        self._bot_y = int(bot_frac * frame_height)
        self._calibrated = False

    # ── Region access ─────────────────────────────────────────────────────

    @property
    def top_y(self) -> int:
        return self._top_y

    @property
    def bottom_y(self) -> int:
        return self._bot_y

    @property
    def piano_height(self) -> int:
        return self._bot_y - self._top_y

    def is_in_piano_region(self, px: int, py: int) -> bool:
        """
        Return True if pixel (px, py) is within the piano region.
        """
        return (
            0 <= px < self._frame_w
            and self._top_y <= py < self._bot_y
        )

    # ── Calibration mode ──────────────────────────────────────────────────

    def run_calibration(self, frame_getter) -> None:
        """
        Simple interactive calibration: show the camera + a guide
        overlay for a few seconds, then return.

        For V1 we keep calibration trivially simple:
          - Show the piano region overlay
          - Let user visually confirm their hand position
          - No complex CV needed

        Args:
            frame_getter: Callable that returns (ok, frame) like camera.read().
        """
        logger.info("Calibration mode started")
        start = time.time()
        duration = 4.0  # seconds of calibration display

        while time.time() - start < duration:
            ok, frame = frame_getter()
            if not ok or frame is None:
                break

            overlay = frame.copy()
            h, w = frame.shape[:2]

            # Shade the piano area
            roi = overlay[self._top_y:self._bot_y, 0:w]
            roi[:] = (roi * 0.4 + np.array([60, 120, 60], dtype=np.uint8) * 0.6).astype(np.uint8)

            # Border line
            cv2.line(overlay, (0, self._top_y), (w, self._top_y), (0, 255, 100), 3)

            # Text instructions
            remaining = max(0, int(duration - (time.time() - start)) + 1)
            cv2.putText(overlay, "CALIBRATION MODE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
            cv2.putText(overlay, "Place your hand over the green area", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, f"Starting in {remaining}s ...", (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Progress bar
            progress = (time.time() - start) / duration
            bar_w = int(w * progress)
            cv2.rectangle(overlay, (0, h - 8), (bar_w, h), (0, 220, 100), -1)

            cv2.imshow("Air Piano", overlay)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break

        self._calibrated = True
        logger.info(
            "Calibration complete: piano region y=[%d, %d]",
            self._top_y, self._bot_y,
        )

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated
