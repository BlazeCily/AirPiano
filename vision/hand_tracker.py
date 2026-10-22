"""
vision/hand_tracker.py — Hand tracking via MediaPipe

Supports TWO MediaPipe API generations transparently:

  API v1 (legacy)  — mediapipe <= 0.10.x
    Uses: mp.solutions.hands.Hands(...)
    Available on: Pi 3B+ 32-bit (mediapipe==0.8.5), older setups

  API v2 (Tasks)   — mediapipe >= 1.0.0
    Uses: mediapipe.tasks.python.vision.HandLandmarker
    Available on: Windows/Mac/Linux x86-64, Pi 4/5 64-bit
    Requires: hand_landmarker.task model file

The correct API is detected automatically at initialize() time.
No code changes needed when switching between environments.

MediaPipe compatibility matrix:
  ┌────────────────────────────────────────────────────────────┐
  │ Platform         │ Python   │ Package           │ API      │
  ├────────────────────────────────────────────────────────────┤
  │ Windows 64-bit   │ 3.9–3.13 │ mediapipe>=1.0    │ Tasks v2 │
  │ Pi 4/5 64-bit    │ 3.9–3.11 │ mediapipe>=0.9    │ legacy   │
  │ Pi 3B+ 32-bit    │ 3.7–3.9  │ mediapipe==0.8.5  │ legacy   │
  └────────────────────────────────────────────────────────────┘

Exposed data per detected hand (same for both APIs):
    hand.landmarks  — list of 21 (x, y, z) tuples, normalized 0–1
    hand.handedness — "Left" | "Right"

Fingertip landmark indices (MediaPipe spec):
    THUMB=4  INDEX=8  MIDDLE=12  RING=16  PINKY=20
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

FINGERTIP_IDS   = [4, 8, 12, 16, 20]
FINGERTIP_NAMES = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]

# Default model file location (Tasks API only)
_DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class HandLandmarks:
    """
    Normalized landmarks for one detected hand.
    landmarks: 21 tuples of (x, y, z), x/y in [0,1].
    handedness: 'Left' or 'Right'.
    """
    landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    handedness: str = "Unknown"

    def fingertips(self) -> List[Tuple[int, float, float, float]]:
        """Return [(fingertip_id, x, y, z), ...] for the 5 fingertips."""
        result = []
        for fid in FINGERTIP_IDS:
            if fid < len(self.landmarks):
                x, y, z = self.landmarks[fid]
                result.append((fid, x, y, z))
        return result


# ── API detection ─────────────────────────────────────────────────────────────

def _detect_mediapipe_api():
    """
    Detect which MediaPipe API generation is installed.

    Returns:
        "tasks"   — mediapipe >= 1.0 (Tasks API, needs model file)
        "legacy"  — mediapipe <= 0.10 (mp.solutions.hands)
        None      — mediapipe not installed
    """
    try:
        import mediapipe as mp
        ver = getattr(mp, "__version__", "0.0.0")
        logger.info("MediaPipe %s detected", ver)

        # Tasks API: mp.tasks exists and mp.solutions does NOT
        if hasattr(mp, "tasks") and not hasattr(mp, "solutions"):
            logger.info("MediaPipe API: Tasks (v2) — mediapipe >= 1.0")
            return "tasks", ver

        # Legacy API: mp.solutions.hands exists
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            logger.info("MediaPipe API: Legacy (v1) — mediapipe <= 0.10")
            return "legacy", ver

        # Ambiguous — try tasks first
        if hasattr(mp, "tasks"):
            return "tasks", ver

        logger.warning("MediaPipe installed but API unknown: %s", ver)
        return None, ver

    except ImportError:
        logger.error(
            "MediaPipe not installed.\n"
            "  Windows / 64-bit:  pip install mediapipe\n"
            "  Pi 3B+ 32-bit:     pip install mediapipe==0.8.5\n"
            "  Or run setup.sh"
        )
        return None, None


# ── HandTracker ───────────────────────────────────────────────────────────────

class HandTracker:
    """
    Real-time hand landmark tracker.
    Automatically uses the correct MediaPipe API for the installed version.

    Usage:
        tracker = HandTracker(config)
        tracker.initialize()
        hands = tracker.process(frame)   # List[HandLandmarks]
        tracker.release()
    """

    def __init__(self, config: dict):
        self._cfg = config
        self._api: Optional[str] = None        # "tasks" | "legacy"
        self._mp_version: Optional[str] = None
        self._available = False

        # Legacy API objects
        self._legacy_hands = None

        # Tasks API objects
        self._tasks_landmarker = None
        self._model_path: str = config.get("model_path", _DEFAULT_MODEL)

        self._last_results: List[HandLandmarks] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """
        Initialize hand tracking. Auto-selects API version.

        Returns:
            True if tracking is ready, False if unavailable.
        """
        api, ver = _detect_mediapipe_api()
        self._api = api
        self._mp_version = ver

        if api is None:
            self._available = False
            return False

        if api == "tasks":
            return self._init_tasks_api()
        else:
            return self._init_legacy_api()

    def _init_tasks_api(self) -> bool:
        """Initialize using mediapipe >= 1.0 Tasks API."""
        if not os.path.exists(self._model_path):
            logger.error(
                "Hand landmarker model not found: %s\n"
                "Download it with:\n"
                "  curl -O https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task\n"
                "  (place it in the air_piano/ root directory)",
                self._model_path,
            )
            print(
                f"\nERROR: MediaPipe model file not found.\n"
                f"Expected: {self._model_path}\n\n"
                f"Download it:\n"
                f"  Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'"
                f" -OutFile hand_landmarker.task\n"
            )
            self._available = False
            return False

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision

            max_hands  = int(self._cfg.get("max_hands", 2))
            det_conf   = float(self._cfg.get("detection_confidence", 0.5))
            track_conf = float(self._cfg.get("tracking_confidence", 0.5))

            base_opts = mp_tasks.BaseOptions(model_asset_path=self._model_path)
            opts = vision.HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=max_hands,
                min_hand_detection_confidence=det_conf,
                min_hand_presence_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
            self._tasks_landmarker = vision.HandLandmarker.create_from_options(opts)
            self._available = True
            logger.info(
                "HandTracker ready (Tasks API, mediapipe %s, max_hands=%d)",
                self._mp_version, max_hands,
            )
            return True
        except Exception as exc:
            logger.error("Tasks API init failed: %s", exc)
            self._available = False
            return False

    def _init_legacy_api(self) -> bool:
        """Initialize using mediapipe <= 0.10 legacy solutions API."""
        try:
            import mediapipe as mp

            max_hands  = int(self._cfg.get("max_hands", 2))
            det_conf   = float(self._cfg.get("detection_confidence", 0.5))
            track_conf = float(self._cfg.get("tracking_confidence", 0.5))

            self._legacy_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
            self._available = True
            logger.info(
                "HandTracker ready (Legacy API, mediapipe %s, max_hands=%d)",
                self._mp_version, max_hands,
            )
            return True
        except Exception as exc:
            logger.error("Legacy API init failed: %s", exc)
            self._available = False
            return False

    def release(self) -> None:
        """Release MediaPipe resources."""
        if self._legacy_hands is not None:
            try:
                self._legacy_hands.close()
            except Exception:
                pass
            self._legacy_hands = None

        if self._tasks_landmarker is not None:
            try:
                self._tasks_landmarker.close()
            except Exception:
                pass
            self._tasks_landmarker = None

        self._available = False
        logger.info("HandTracker released")

    # ── Processing ────────────────────────────────────────────────────────

    def process(self, frame) -> List[HandLandmarks]:
        """
        Run hand detection on a BGR OpenCV frame.

        Returns:
            List[HandLandmarks] — 0, 1, or 2 hands. Empty on failure.
        """
        if not self._available or frame is None:
            return []

        try:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self._api == "tasks":
                return self._process_tasks(rgb)
            else:
                return self._process_legacy(rgb)
        except Exception as exc:
            logger.warning("Hand detection error: %s", exc)
            return []

    def _process_tasks(self, rgb_frame) -> List[HandLandmarks]:
        """Process using Tasks API (mediapipe >= 1.0)."""
        import mediapipe as mp
        import numpy as np

        h, w = rgb_frame.shape[:2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        result = self._tasks_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            self._last_results = []
            return []

        detected = []
        handedness_list = result.handedness or []

        for i, hand_lm in enumerate(result.hand_landmarks):
            lms = [(lm.x, lm.y, lm.z) for lm in hand_lm]

            handedness = "Unknown"
            if i < len(handedness_list) and handedness_list[i]:
                try:
                    handedness = handedness_list[i][0].category_name
                except (IndexError, AttributeError):
                    pass

            detected.append(HandLandmarks(landmarks=lms, handedness=handedness))

        self._last_results = detected
        return detected

    def _process_legacy(self, rgb_frame) -> List[HandLandmarks]:
        """Process using Legacy solutions API (mediapipe <= 0.10)."""
        rgb_frame.flags.writeable = False
        results = self._legacy_hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            self._last_results = []
            return []

        detected = []
        handedness_list = results.multi_handedness or []

        for i, hand_lm in enumerate(results.multi_hand_landmarks):
            lms = [(lm.x, lm.y, lm.z) for lm in hand_lm.landmark]

            handedness = "Unknown"
            if i < len(handedness_list):
                try:
                    handedness = handedness_list[i].classification[0].label
                except (IndexError, AttributeError):
                    pass

            detected.append(HandLandmarks(landmarks=lms, handedness=handedness))

        self._last_results = detected
        return detected

    # ── Drawing helper ────────────────────────────────────────────────────

    def draw_landmarks(self, frame, hands: List[HandLandmarks]) -> None:
        """Draw skeleton and landmarks on frame in-place (pure OpenCV)."""
        import cv2
        h, w = frame.shape[:2]

        for hand in hands:
            lms = hand.landmarks
            if len(lms) < 21:
                continue

            for (a, b) in HAND_CONNECTIONS:
                if a < len(lms) and b < len(lms):
                    x1, y1 = int(lms[a][0] * w), int(lms[a][1] * h)
                    x2, y2 = int(lms[b][0] * w), int(lms[b][1] * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)

            for idx, (x, y, _) in enumerate(lms):
                px, py = int(x * w), int(y * h)
                if idx in FINGERTIP_IDS:
                    cv2.circle(frame, (px, py), 10, (0, 200, 255), -1)
                    cv2.circle(frame, (px, py), 10, (255, 255, 255), 2)
                else:
                    cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

    # ── Status ────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def mediapipe_version(self) -> Optional[str]:
        return self._mp_version

    @property
    def api_type(self) -> Optional[str]:
        """Returns 'tasks' or 'legacy' depending on which API is active."""
        return self._api

    def get_last_results(self) -> List[HandLandmarks]:
        return self._last_results
