"""Procedural audio synthesizer and sound effect manager for PATHFINDER.

Synthesizes 8-bit / 16-bit retro-futuristic acoustic chimes, movement ticks,
checkpoint alert pulses, and victory fanfares entirely in-memory using Python's
built-in standard library (math, array). Requires zero external sound assets (.wav/.mp3),
guaranteeing 100% portability, zero 404 missing asset errors, and silent graceful
degradation in headless environments (e.g. CI / testing).
"""

import array
import math
from typing import Any, Dict, Optional
import pygame


class SoundManager:
    """Manages procedural audio effects with volume control and muted safeguards."""

    SAMPLE_RATE = 44100

    def __init__(self, config_settings: Optional[Dict[str, Any]] = None) -> None:
        """Initialize procedural sound bank.

        Args:
            config_settings: Dictionary reference containing 'sound_enabled' and 'volume'.
        """
        self.settings: Dict[str, Any] = config_settings if config_settings is not None else {}
        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        self.is_available: bool = False

        self._init_audio()

    def _init_audio(self) -> None:
        """Initialize Pygame mixer and synthesize procedural sound clips."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.SAMPLE_RATE, size=-16, channels=1, buffer=512)
            self._synthesize_all()
            self.is_available = True
        except Exception as e:
            # Headless or missing audio device: silently degrade
            self.is_available = False
            self.sounds.clear()

    def _generate_tone(
        self,
        frequencies: list[float],
        durations: list[float],
        wave_type: str = "sine",
        decay: bool = True,
        amplitude: int = 14000,
    ) -> Optional[pygame.mixer.Sound]:
        """Synthesize a sequence of frequencies into a Pygame Sound buffer."""
        try:
            raw_samples = array.array("h")
            for freq, dur in zip(frequencies, durations):
                n_samples = int(dur * self.SAMPLE_RATE)
                for i in range(n_samples):
                    t = float(i) / self.SAMPLE_RATE
                    envelope = (1.0 - (i / float(n_samples))) if decay else 1.0

                    if wave_type == "sine":
                        val = math.sin(2.0 * math.pi * freq * t)
                    elif wave_type == "square":
                        val = 0.8 if math.sin(2.0 * math.pi * freq * t) >= 0 else -0.8
                    elif wave_type == "triangle":
                        val = (2.0 / math.pi) * math.asin(math.sin(2.0 * math.pi * freq * t))
                    else:
                        val = math.sin(2.0 * math.pi * freq * t)

                    sample_val = int(amplitude * envelope * val)
                    sample_val = max(-32768, min(32767, sample_val))
                    raw_samples.append(sample_val)

            return pygame.mixer.Sound(buffer=raw_samples)
        except Exception:
            return None

    def _synthesize_all(self) -> None:
        """Synthesize all procedural sound effects."""
        # 1. Step: subtle soft tick (600 Hz decaying in 25ms)
        snd_step = self._generate_tone([600.0], [0.025], wave_type="sine", decay=True, amplitude=6000)
        if snd_step:
            self.sounds["step"] = snd_step

        # 2. Correct: ascending two-tone affirmative chime (523 Hz -> 659 Hz, C5 -> E5)
        snd_correct = self._generate_tone([523.25, 659.25], [0.09, 0.16], wave_type="sine", decay=True, amplitude=14000)
        if snd_correct:
            self.sounds["correct"] = snd_correct

        # 3. Wrong: downward dissonance buzzer (240 Hz -> 180 Hz)
        snd_wrong = self._generate_tone([240.0, 180.0], [0.12, 0.18], wave_type="triangle", decay=True, amplitude=12000)
        if snd_wrong:
            self.sounds["wrong"] = snd_wrong

        # 4. Checkpoint approach / trigger alert (330 Hz pulse)
        snd_cp = self._generate_tone([329.63, 440.0], [0.08, 0.12], wave_type="sine", decay=True, amplitude=13000)
        if snd_cp:
            self.sounds["checkpoint"] = snd_cp

        # 5. Hint used: shimmering triad arpeggio (G5 -> B5 -> D6)
        snd_hint = self._generate_tone([783.99, 987.77, 1174.66], [0.07, 0.07, 0.14], wave_type="sine", decay=True, amplitude=12000)
        if snd_hint:
            self.sounds["hint"] = snd_hint

        # 6. Victory: triumphant 4-tone fanfaric arpeggio (C5 -> E5 -> G5 -> C6)
        snd_victory = self._generate_tone([523.25, 659.25, 783.99, 1046.50], [0.10, 0.10, 0.12, 0.35], wave_type="sine", decay=True, amplitude=15000)
        if snd_victory:
            self.sounds["victory"] = snd_victory

        # 7. UI Click: crisp mechanical tick
        snd_click = self._generate_tone([1200.0], [0.015], wave_type="square", decay=True, amplitude=4000)
        if snd_click:
            self.sounds["click"] = snd_click

    def play(self, sound_name: str) -> None:
        """Play named procedural sound effect according to user audio preferences.

        Args:
            sound_name: Key of sound effect ('step', 'correct', 'wrong', 'checkpoint', 'hint', 'victory', 'click').
        """
        if not self.is_available:
            return

        sound_enabled = self.settings.get("sound_enabled", True)
        if not sound_enabled:
            return

        volume_pct = self.settings.get("volume", 75)
        volume_factor = max(0.0, min(1.0, float(volume_pct) / 100.0))
        if volume_factor <= 0.0:
            return

        sound = self.sounds.get(sound_name)
        if sound is not None:
            try:
                sound.set_volume(volume_factor)
                sound.play()
            except Exception:
                pass
