"""
utils/fps.py — Rolling FPS counter

Tracks frame rate over a sliding window of timestamps.
No external dependencies.
"""

import time
from collections import deque


class FPSCounter:
    """
    Rolling-average FPS counter.

    Usage:
        fps = FPSCounter(window=30)
        while True:
            fps.tick()
            print(fps.get())
    """

    def __init__(self, window: int = 30):
        """
        Args:
            window: Number of frames to average over.
        """
        self._window = window
        self._timestamps: deque = deque(maxlen=window)
        self._start = time.monotonic()

    def tick(self) -> None:
        """Call once per rendered frame."""
        self._timestamps.append(time.monotonic())

    def get(self) -> float:
        """Return current FPS as a float. Returns 0 if fewer than 2 frames."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def get_int(self) -> int:
        """Return FPS rounded to nearest integer."""
        return int(round(self.get()))

    def reset(self) -> None:
        """Clear accumulated history."""
        self._timestamps.clear()
