"""
vision/camera.py — Camera abstraction

Supports two backends:
  1. USB webcam  — OpenCV VideoCapture(index)
  2. IP Webcam   — OpenCV VideoCapture(http://IP:port/video)

Public API:
    camera = Camera(config)
    camera.open()
    ok, frame = camera.read()
    camera.release()
    camera.is_connected()

The frame returned by read() is always:
  - BGR numpy array (standard OpenCV format)
  - Horizontally flipped if config['flip'] is True (mirror mode)
  - None on failure (caller must handle)

Error handling:
  - open() returns False if camera unavailable (app continues)
  - read() returns (False, None) on failure
  - is_connected() reflects current live state
"""

import logging
import platform
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Detect OS once at import time
_IS_WINDOWS = platform.system() == "Windows"
_IS_LINUX   = platform.system() == "Linux"


class Camera:
    """
    Unified camera capture for USB and IP Webcam sources.

    Mirror mode (horizontal flip) is applied here so all downstream
    code (hand tracking, collision) sees the same coordinate system
    as the display.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: The 'camera' section from config.yaml.
                mode: "usb" | "ip"
                device: int (USB index)
                width, height: desired resolution
                fps: desired capture framerate
                ip_url: HTTP stream URL for IP mode
                flip: bool — horizontal mirror
        """
        self._cfg = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._connected = False
        self._flip = bool(config.get("flip", True))
        self._mode = config.get("mode", "usb").lower()
        self._width = int(config.get("width", 640))
        self._height = int(config.get("height", 480))
        self._fps = int(config.get("fps", 20))
        self._device = config.get("device", 0)
        self._ip_url = config.get("ip_url", "")
        self._last_read_ok = False
        # Backend: auto-select based on OS, or use config override
        self._backend = self._resolve_backend(config.get("backend", "auto"))

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def open(self) -> bool:
        """
        Open the camera.

        Returns:
            True if the camera opened successfully and frames are readable.
        """
        source = self._resolve_source()
        logger.info(
            "Opening camera: %s (mode=%s, backend=%s)",
            source, self._mode,
            self._backend_name(),
        )

        try:
            if self._backend is not None and self._mode != "ip":
                self._cap = cv2.VideoCapture(source, self._backend)
            else:
                self._cap = cv2.VideoCapture(source)
        except Exception as exc:
            logger.error("cv2.VideoCapture(%s) raised: %s", source, exc)
            self._connected = False
            return False

        if not self._cap.isOpened():
            self._print_camera_error(source)
            self._connected = False
            return False

        # Configure resolution and FPS (best-effort; USB cameras may ignore)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        # Keep buffer small so we always get the latest frame
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Test read
        ok, frame = self._cap.read()
        if not ok or frame is None:
            logger.error("Camera opened but could not read a test frame")
            self._cap.release()
            self._cap = None
            self._connected = False
            return False

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera ready: %dx%d @ %.1f FPS (requested %dx%d @ %d)",
            actual_w, actual_h, actual_fps,
            self._width, self._height, self._fps,
        )
        self._connected = True
        return True

    def release(self) -> None:
        """Release the camera and free resources."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._connected = False
        logger.info("Camera released")

    # ── Frame capture ─────────────────────────────────────────────────────

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture the next frame.

        Returns:
            (True, frame)   — frame is a BGR numpy array
            (False, None)   — camera unavailable or read error
        """
        if self._cap is None or not self._cap.isOpened():
            self._connected = False
            return False, None

        ok, frame = self._cap.read()

        if not ok or frame is None or frame.size == 0:
            self._last_read_ok = False
            self._connected = False
            logger.warning("Camera read failed — connection lost?")
            return False, None

        self._last_read_ok = True
        self._connected = True

        # Mirror horizontally if requested (mirror mode)
        if self._flip:
            frame = cv2.flip(frame, 1)

        return True, frame

    # ── Status ────────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        """Return True if the camera is open and producing frames."""
        return self._connected and self._cap is not None and self._cap.isOpened()

    @property
    def width(self) -> int:
        if self._cap and self._cap.isOpened():
            return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return self._width

    @property
    def height(self) -> int:
        if self._cap and self._cap.isOpened():
            return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self._height

    @property
    def mode(self) -> str:
        return self._mode

    # ── Internal ──────────────────────────────────────────────────────────

    def _resolve_source(self):
        """Return the VideoCapture source (int or str)."""
        if self._mode == "ip":
            url = self._ip_url.strip()
            if not url:
                raise ValueError(
                    "camera.mode is 'ip' but camera.ip_url is empty in config.yaml"
                )
            return url
        else:
            # USB / built-in — device can be int index or "/dev/video0" path
            dev = self._device
            if isinstance(dev, str):
                dev = dev.strip()
                if dev.startswith("/dev/video"):
                    try:
                        return int(dev.replace("/dev/video", ""))
                    except ValueError:
                        return dev
            return int(dev)

    @staticmethod
    def _resolve_backend(backend_cfg: str) -> Optional[int]:
        """
        Map a backend name string to a cv2 backend constant.

        "auto"  → CAP_DSHOW on Windows, CAP_V4L2 on Linux, None elsewhere
        "dshow" → CAP_DSHOW  (Windows DirectShow — best for laptop cameras)
        "v4l2"  → CAP_V4L2   (Linux Video4Linux2 — best for Pi/USB cameras)
        "any"   → None       (let OpenCV decide)
        """
        b = str(backend_cfg).lower().strip()
        if b == "auto":
            if _IS_WINDOWS:
                logger.info("Camera backend: auto → CAP_DSHOW (Windows)")
                return cv2.CAP_DSHOW
            elif _IS_LINUX:
                logger.info("Camera backend: auto → CAP_V4L2 (Linux)")
                return cv2.CAP_V4L2
            else:
                return None
        elif b == "dshow":
            return cv2.CAP_DSHOW
        elif b in ("v4l2", "v4l"):
            return cv2.CAP_V4L2
        else:
            return None  # "any" or unrecognised → let OpenCV choose

    def _backend_name(self) -> str:
        """Return human-readable backend name for logging."""
        mapping = {
            cv2.CAP_DSHOW: "DSHOW",
            cv2.CAP_V4L2:  "V4L2",
        }
        return mapping.get(self._backend, "default")

    def _print_camera_error(self, source) -> None:
        """Print a user-friendly error message."""
        if self._mode == "usb":
            print(
                f"\nERROR: Camera device {source!r} was not found.\n"
                f"Try:\n"
                f"  • Connect a USB webcam\n"
                f"  • Run:  ls /dev/video*\n"
                f"  • Change camera.device in config.yaml\n"
            )
        else:
            print(
                f"\nERROR: Could not connect to IP Webcam at {source!r}.\n"
                f"Try:\n"
                f"  • Make sure the IP Webcam app is running on your phone\n"
                f"  • Check phone and Pi are on the same WiFi network\n"
                f"  • Verify camera.ip_url in config.yaml\n"
            )
        logger.error("Could not open camera source: %s", source)
