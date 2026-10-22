"""
ui/hud.py — Heads-Up Display overlay

Draws the status panel in the top-left corner:

  ┌─────────────────────┐
  │ AIR PIANO           │
  │ Camera: OK          │
  │ Hands: 2            │
  │ FPS: 16             │
  │ Playing: C5 E5 G5   │
  └─────────────────────┘

Kept minimal — does not cover the camera feed.
All text uses cv2.putText for Raspberry Pi compatibility
(no PIL, no freetype font loading needed).
"""

import logging
from typing import Optional, Set

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HUD:
    """
    Minimal status overlay drawn on the camera frame.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: The 'display' section of config.yaml.
        """
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._scale = float(config.get("font_scale", 0.55))
        self._color = tuple(config.get("hud_color", [255, 255, 255]))
        self._bg_alpha = 0.45  # semi-transparent background

    def draw(
        self,
        frame: np.ndarray,
        fps: float,
        n_hands: int,
        camera_ok: bool,
        audio_ok: bool,
        tracker_ok: bool,
        active_notes: Set[str],
        mode_text: Optional[str] = None,
        mediapipe_version: Optional[str] = None,
    ) -> None:
        """
        Draw the HUD onto frame in-place.

        Args:
            frame:              BGR frame to annotate.
            fps:                Current frames-per-second.
            n_hands:            Number of hands detected this frame.
            camera_ok:          Whether camera is connected.
            audio_ok:           Whether audio engine is ready.
            tracker_ok:         Whether hand tracker is available.
            active_notes:       Currently playing note names.
            mode_text:          Optional mode label (e.g. "PAUSED").
            mediapipe_version:  Displayed in first-frame startup info.
        """
        lines = self._build_lines(
            fps, n_hands, camera_ok, audio_ok, tracker_ok, active_notes
        )

        # Draw semi-transparent background rectangle
        panel_w = 210
        line_h = 22
        panel_h = len(lines) * line_h + 14
        x0, y0 = 8, 8

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - 4, y0 - 4),
                      (x0 + panel_w, y0 + panel_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, self._bg_alpha, frame,
                        1 - self._bg_alpha, 0, frame)

        # Draw text lines
        for i, (text, color) in enumerate(lines):
            ty = y0 + 12 + i * line_h
            cv2.putText(frame, text, (x0 + 4, ty),
                        self._font, self._scale, color, 1, cv2.LINE_AA)

        # Optional mode banner (e.g. PAUSED, CALIBRATION)
        if mode_text:
            self._draw_mode_banner(frame, mode_text)

    def _build_lines(
        self,
        fps: float,
        n_hands: int,
        camera_ok: bool,
        audio_ok: bool,
        tracker_ok: bool,
        active_notes: Set[str],
    ):
        """Build the list of (text, color) tuples."""
        WHITE  = (255, 255, 255)
        GREEN  = (100, 255, 100)
        RED    = (80, 80, 255)
        YELLOW = (80, 220, 255)
        CYAN   = (255, 220, 80)

        lines = []

        # Title
        lines.append(("AIR PIANO", CYAN))

        # Camera status
        cam_text  = "Camera: OK"    if camera_ok  else "Camera: DISCONNECTED"
        cam_color = GREEN           if camera_ok  else RED
        lines.append((cam_text, cam_color))

        # Hand tracker
        if tracker_ok:
            lines.append((f"Hands: {n_hands}", WHITE))
        else:
            lines.append(("Hands: NO TRACKER", RED))

        # Audio
        audio_text  = "Audio: OK"    if audio_ok  else "Audio: UNAVAILABLE"
        audio_color = GREEN          if audio_ok  else RED
        lines.append((audio_text, audio_color))

        # FPS
        fps_color = GREEN if fps >= 12 else (RED if fps < 8 else YELLOW)
        lines.append((f"FPS: {int(fps)}", fps_color))

        # Playing notes
        if active_notes:
            notes_str = " ".join(sorted(active_notes))
            lines.append((f"Playing: {notes_str}", YELLOW))
        else:
            lines.append(("Playing: —", (160, 160, 160)))

        return lines

    def _draw_mode_banner(self, frame: np.ndarray, text: str) -> None:
        """Draw a centred mode banner near the top of the frame."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        tx = (w - tw) // 2
        ty = 55
        # Shadow
        cv2.putText(frame, text, (tx + 2, ty + 2), font, scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # Text
        cv2.putText(frame, text, (tx, ty), font, scale,
                    (0, 255, 200), thickness, cv2.LINE_AA)

    def draw_controls_hint(self, frame: np.ndarray) -> None:
        """Draw a one-line controls hint at the very bottom."""
        h, w = frame.shape[:2]
        text = "Q/ESC: Quit  |  R: Recal  |  M: Marks  |  P: Pause"
        cv2.putText(frame, text, (8, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (160, 160, 160), 1, cv2.LINE_AA)
