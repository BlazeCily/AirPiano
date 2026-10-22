"""
audio/engine.py — AudioEngine

Wraps pygame.mixer to provide:
  - Non-blocking polyphonic note playback
  - Pre-rendered note buffers (from synth.py)
  - play_note() / stop_note() API
  - Graceful degradation if pygame audio is unavailable

Design notes:
  - All note buffers are generated ONCE at initialize()
  - Each note gets a dedicated pygame Channel for independent control
  - stop_note() fades out rather than hard-cutting (avoids click artifacts)
  - The engine never spawns threads; it is driven by the main loop
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import pygame
    import numpy as np
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False
    logger.warning("pygame not available — audio disabled")

from audio.synth import generate_all_notes, get_all_note_names


class AudioEngine:
    """
    Polyphonic audio engine backed by pygame.mixer.

    Supports up to `channels` simultaneously playing notes.
    Each note has a dedicated channel so stop_note() is precise.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: The 'audio' section of config.yaml.
        """
        self._cfg = config
        self._ready = False
        self._sounds: Dict[str, "pygame.mixer.Sound"] = {}
        self._channels: Dict[str, "pygame.mixer.Channel"] = {}
        self._active_notes: set = set()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """
        Initialize pygame mixer and pre-render all note sounds.

        Returns:
            True if audio is ready, False if it failed (app continues).
        """
        if not _PYGAME_AVAILABLE:
            logger.error("pygame not installed — audio unavailable")
            return False

        sample_rate = int(self._cfg.get("sample_rate", 22050))
        buffer_size = int(self._cfg.get("buffer_size", 512))
        n_channels = int(self._cfg.get("channels", 8))
        volume = float(self._cfg.get("volume", 0.8))
        duration = float(self._cfg.get("note_duration", 2.0))
        attack_ms = float(self._cfg.get("attack_ms", 10))
        decay_ms = float(self._cfg.get("decay_ms", 800))

        try:
            # pygame.mixer.pre_init must be called before pygame.init
            pygame.mixer.pre_init(
                frequency=sample_rate,
                size=-16,          # signed 16-bit
                channels=2,        # stereo — matches (N,2) array from synth.py
                buffer=buffer_size,
            )
            pygame.mixer.init()
            pygame.mixer.set_num_channels(max(n_channels, 13))  # at least one per note

            logger.info(
                "pygame.mixer initialized: %d Hz, buffer=%d, channels=%d",
                sample_rate, buffer_size, n_channels,
            )
        except Exception as exc:
            logger.error("pygame.mixer init failed: %s", exc)
            return False

        # ── Pre-render all note waveforms ────────────────────────────────
        logger.info("Pre-rendering note waveforms …")
        try:
            note_arrays = generate_all_notes(
                sample_rate=sample_rate,
                duration=duration,
                volume=volume,
                attack_ms=attack_ms,
                decay_ms=decay_ms,
            )
        except Exception as exc:
            logger.error("Note synthesis failed: %s", exc)
            pygame.mixer.quit()
            return False

        # Convert numpy arrays → pygame.Sound objects
        for note_name, arr in note_arrays.items():
            try:
                sound = pygame.sndarray.make_sound(arr)
                self._sounds[note_name] = sound
            except Exception as exc:
                logger.warning("Could not create Sound for %s: %s", note_name, exc)

        # Pre-allocate a dedicated channel per note
        all_notes = get_all_note_names()
        for i, note_name in enumerate(all_notes):
            if note_name in self._sounds:
                ch = pygame.mixer.Channel(i)
                self._channels[note_name] = ch

        self._ready = True
        logger.info("AudioEngine ready — %d notes available", len(self._sounds))
        return True

    def shutdown(self) -> None:
        """Stop all audio and shut down pygame.mixer."""
        if not _PYGAME_AVAILABLE:
            return
        try:
            pygame.mixer.stop()
            pygame.mixer.quit()
        except Exception:
            pass
        self._ready = False
        self._active_notes.clear()
        logger.info("AudioEngine shut down")

    # ── Playback API ─────────────────────────────────────────────────────

    def play_note(self, note_name: str) -> bool:
        """
        Start playing a note.

        If the note is already playing it is restarted from the beginning.
        Non-blocking — returns immediately.

        Args:
            note_name: e.g. "C5", "C#5"

        Returns:
            True if playback started successfully.
        """
        if not self._ready:
            return False
        if note_name not in self._sounds:
            logger.warning("play_note: unknown note %r", note_name)
            return False

        sound = self._sounds[note_name]
        channel = self._channels.get(note_name)

        try:
            if channel is not None:
                channel.stop()
                channel.play(sound)
            else:
                sound.play()
            self._active_notes.add(note_name)
            return True
        except Exception as exc:
            logger.error("play_note(%s) failed: %s", note_name, exc)
            return False

    def stop_note(self, note_name: str, fadeout_ms: int = 80) -> None:
        """
        Stop a playing note with a short fadeout to avoid clicks.

        Args:
            note_name:   Note to stop.
            fadeout_ms:  Fadeout duration in ms (default 80 ms).
        """
        if not self._ready:
            return
        channel = self._channels.get(note_name)
        if channel is not None:
            try:
                channel.fadeout(fadeout_ms)
            except Exception:
                pass
        self._active_notes.discard(note_name)

    def stop_all(self, fadeout_ms: int = 150) -> None:
        """Stop all currently playing notes."""
        if not self._ready:
            return
        for note_name in list(self._active_notes):
            self.stop_note(note_name, fadeout_ms)

    # ── Status ───────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def active_notes(self) -> set:
        """Set of note names currently playing."""
        return set(self._active_notes)

    def is_playing(self, note_name: str) -> bool:
        return note_name in self._active_notes

    def set_volume(self, volume: float) -> None:
        """Set master volume 0.0–1.0."""
        volume = max(0.0, min(1.0, volume))
        for sound in self._sounds.values():
            try:
                sound.set_volume(volume)
            except Exception:
                pass
