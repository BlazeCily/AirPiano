"""
piano/note_state.py — Per-finger note state machine

This module prevents the most common bug in gesture instruments:
note re-triggering every frame while the finger holds still.

State machine per finger:
    ┌─────────────┐
    │  NO_KEY     │ ← finger not on any key / outside piano
    └──────┬──────┘
           │ enters key K
    ┌──────▼──────┐
    │  KEY_DOWN   │ ← finger is on key K → emit NOTE_ON(K)
    └──────┬──────┘
      ┌────┴────┐
      │         │
  stays on K  moves to J
      │         │
      │   ┌─────▼──────┐
      │   │  KEY_CHANGE │ → emit NOTE_OFF(K), NOTE_ON(J)
      │   └─────┬──────┘
      │         │
  ┌───▼─────────▼──────┐
  │      KEY_DOWN       │ ← holding, no re-trigger
  └─────────────────────┘
      │
    lifts
      │
    ┌─▼──────┐
    │ NO_KEY │ → emit NOTE_OFF(K)
    └────────┘

Finger identity:
    Derived from (hand_index, fingertip_landmark_id).
    Example: "h0_lm8" = hand 0, index finger tip (landmark 8).

This is stable within a frame even if hand order changes between
frames (because MediaPipe re-numbers hands), but in practice
MediaPipe hand ordering is fairly stable across frames.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── Events emitted by the state machine ──────────────────────────────────────

class NoteEvent(Enum):
    NOTE_ON  = auto()
    NOTE_OFF = auto()


@dataclass
class NoteAction:
    event: NoteEvent
    note: str
    finger_id: str


# ── Per-finger state ──────────────────────────────────────────────────────────

class FingerState(Enum):
    NO_KEY   = auto()   # finger not on any key
    KEY_DOWN = auto()   # finger is holding a key


@dataclass
class FingerRecord:
    """Tracks one finger's note state."""
    finger_id: str
    state: FingerState = FingerState.NO_KEY
    current_key: Optional[str] = None
    last_trigger_time: float = 0.0


# ── Note State Manager ────────────────────────────────────────────────────────

class NoteStateManager:
    """
    Manages note state for all tracked fingertips.

    Call update() once per frame with the current set of
    (finger_id, key_name_or_None) pairs.

    update() returns a list of NoteAction events that the caller
    should route to the AudioEngine.
    """

    def __init__(self, debounce_ms: int = 150):
        """
        Args:
            debounce_ms: Minimum ms before a finger can retrigger the
                         same key after releasing and re-pressing it.
        """
        self._debounce_s = debounce_ms / 1000.0
        self._fingers: Dict[str, FingerRecord] = {}
        self._seen_this_frame: Set[str] = set()

    def update(
        self,
        finger_keys: List[Tuple[str, Optional[str]]],
    ) -> List[NoteAction]:
        """
        Process the current frame's fingertip → key mapping.

        Args:
            finger_keys: List of (finger_id, key_name_or_None).
                         finger_id: e.g. "h0_lm8"
                         key_name:  e.g. "C5", "C#5", or None

        Returns:
            List of NoteAction events to act on this frame.
        """
        events: List[NoteAction] = []
        now = time.monotonic()
        self._seen_this_frame = {fid for fid, _ in finger_keys}

        for finger_id, key_name in finger_keys:
            events.extend(self._process_finger(finger_id, key_name, now))

        # Any finger not seen this frame → release its note
        for finger_id, rec in list(self._fingers.items()):
            if finger_id not in self._seen_this_frame:
                if rec.state == FingerState.KEY_DOWN and rec.current_key:
                    events.append(NoteAction(NoteEvent.NOTE_OFF, rec.current_key, finger_id))
                    logger.debug("NOTE_OFF %s (finger lost: %s)", rec.current_key, finger_id)
                del self._fingers[finger_id]

        return events

    def _process_finger(
        self,
        finger_id: str,
        key_name: Optional[str],
        now: float,
    ) -> List[NoteAction]:
        events: List[NoteAction] = []

        if finger_id not in self._fingers:
            self._fingers[finger_id] = FingerRecord(finger_id=finger_id)
        rec = self._fingers[finger_id]

        # ── Finger NOT on any key ─────────────────────────────────────────
        if key_name is None:
            if rec.state == FingerState.KEY_DOWN and rec.current_key:
                events.append(NoteAction(NoteEvent.NOTE_OFF, rec.current_key, finger_id))
                logger.debug("NOTE_OFF %s (%s lifted)", rec.current_key, finger_id)
            rec.state = FingerState.NO_KEY
            rec.current_key = None
            return events

        # ── Finger IS on a key ────────────────────────────────────────────
        if rec.state == FingerState.NO_KEY:
            # Fresh press
            elapsed_since_last = now - rec.last_trigger_time
            if elapsed_since_last >= self._debounce_s:
                events.append(NoteAction(NoteEvent.NOTE_ON, key_name, finger_id))
                logger.debug("NOTE_ON  %s (%s)", key_name, finger_id)
                rec.state = FingerState.KEY_DOWN
                rec.current_key = key_name
                rec.last_trigger_time = now

        elif rec.state == FingerState.KEY_DOWN:
            if key_name != rec.current_key:
                # Finger slid to a different key
                old_key = rec.current_key
                if old_key:
                    events.append(NoteAction(NoteEvent.NOTE_OFF, old_key, finger_id))
                    logger.debug("NOTE_OFF %s (slide: %s)", old_key, finger_id)
                events.append(NoteAction(NoteEvent.NOTE_ON, key_name, finger_id))
                logger.debug("NOTE_ON  %s (slide: %s)", key_name, finger_id)
                rec.current_key = key_name
                rec.last_trigger_time = now
            # Else: same key, same state → do nothing (no retrigger)

        return events

    def get_active_notes(self) -> Set[str]:
        """Return the set of note names currently held down."""
        return {
            rec.current_key
            for rec in self._fingers.values()
            if rec.state == FingerState.KEY_DOWN and rec.current_key
        }

    def get_finger_states(self) -> Dict[str, FingerRecord]:
        """Return a snapshot of all finger records (for debug HUD)."""
        return dict(self._fingers)

    def release_all(self) -> List[NoteAction]:
        """Force-release all held notes (call on app exit or pause)."""
        events = []
        for finger_id, rec in self._fingers.items():
            if rec.state == FingerState.KEY_DOWN and rec.current_key:
                events.append(NoteAction(NoteEvent.NOTE_OFF, rec.current_key, finger_id))
        self._fingers.clear()
        return events
