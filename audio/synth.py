"""
audio/synth.py — Additive sine-wave synthesizer

Generates piano-like tones purely in numpy.
No samples, no external audio files.

Synthesis model:
  fundamental + 2nd + 3rd + 4th harmonics
  with amplitude decay (exponential envelope)

All notes are pre-rendered as numpy arrays at startup
so there is zero CPU cost during playback.
"""

import numpy as np
from typing import Dict

# ── Note frequency table ────────────────────────────────────────────────────
# Chromatic notes C5–C6 (MIDI 72–84)
NOTE_FREQUENCIES: Dict[str, float] = {
    "C5":  523.251,
    "C#5": 554.365,
    "D5":  587.330,
    "D#5": 622.254,
    "E5":  659.255,
    "F5":  698.456,
    "F#5": 739.989,
    "G5":  783.991,
    "G#5": 830.609,
    "A5":  880.000,
    "A#5": 932.328,
    "B5":  987.767,
    "C6":  1046.502,
}


def generate_note(
    note_name: str,
    sample_rate: int = 22050,
    duration: float = 2.0,
    volume: float = 0.8,
    attack_ms: float = 10.0,
    decay_ms: float = 800.0,
) -> np.ndarray:
    """
    Generate a single piano note as a 16-bit PCM numpy array (mono).

    The tone uses additive synthesis with 4 harmonics and an
    exponential decay envelope that simulates a struck piano string.

    Args:
        note_name:   One of the keys in NOTE_FREQUENCIES.
        sample_rate: Audio sample rate (default 22050 Hz).
        duration:    Pre-render length in seconds.
        volume:      Peak amplitude 0.0–1.0.
        attack_ms:   Linear attack ramp in milliseconds.
        decay_ms:    Time constant for exponential decay in milliseconds.

    Returns:
        np.ndarray of dtype int16, shape (N,).

    Raises:
        KeyError: If note_name is not in NOTE_FREQUENCIES.
    """
    if note_name not in NOTE_FREQUENCIES:
        raise KeyError(f"Unknown note: {note_name!r}. Valid: {list(NOTE_FREQUENCIES)}")

    freq = NOTE_FREQUENCIES[note_name]
    n_samples = int(sample_rate * duration)
    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float32)

    # ── Additive synthesis ────────────────────────────────────────────────
    # Harmonic amplitudes chosen to mimic a piano timbre:
    #   fundamental strong, 2nd slightly weaker, upper harmonics roll off.
    harmonics = [
        (1, 1.00),   # fundamental
        (2, 0.55),   # 2nd harmonic
        (3, 0.25),   # 3rd harmonic
        (4, 0.10),   # 4th harmonic
    ]
    wave = np.zeros(n_samples, dtype=np.float32)
    for harmonic, amp in harmonics:
        wave += amp * np.sin(2.0 * np.pi * freq * harmonic * t)

    # Normalize so peak ≈ 1.0
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave /= peak

    # ── Envelope ─────────────────────────────────────────────────────────
    envelope = np.ones(n_samples, dtype=np.float32)

    # Linear attack
    attack_samples = max(1, int(sample_rate * attack_ms / 1000.0))
    attack_samples = min(attack_samples, n_samples)
    envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)

    # Exponential decay after attack
    decay_tau = decay_ms / 1000.0           # time constant in seconds
    decay_t = np.arange(n_samples - attack_samples, dtype=np.float32) / sample_rate
    decay_env = np.exp(-decay_t / decay_tau)
    envelope[attack_samples:] = decay_env

    wave *= envelope
    wave *= volume

    # ── Convert to 16-bit PCM (stereo — pygame.sndarray requires 2D) ─────
    # pygame.mixer is initialised with channels=2 (stereo).
    # make_sound() expects shape (N, 2); duplicate mono → stereo.
    wave_int16 = (wave * 32767).astype(np.int16)
    stereo = np.column_stack((wave_int16, wave_int16))  # shape (N, 2)
    return stereo


def generate_all_notes(
    sample_rate: int = 22050,
    duration: float = 2.0,
    volume: float = 0.8,
    attack_ms: float = 10.0,
    decay_ms: float = 800.0,
) -> Dict[str, np.ndarray]:
    """
    Pre-render all notes in NOTE_FREQUENCIES.

    Returns:
        Dict mapping note name → int16 numpy array.
    """
    return {
        name: generate_note(
            name,
            sample_rate=sample_rate,
            duration=duration,
            volume=volume,
            attack_ms=attack_ms,
            decay_ms=decay_ms,
        )
        for name in NOTE_FREQUENCIES
    }


def get_all_note_names():
    """Return ordered list of all note names."""
    return list(NOTE_FREQUENCIES.keys())
