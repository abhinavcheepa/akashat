"""Delta-time based non-blocking countdown timer for PATHFINDER.

Guarantees smooth per-frame updates without halting or blocking the Pygame loop.
"""

from typing import Optional


class Timer:
    """Non-blocking countdown timer driven strictly by frame delta time."""

    def __init__(self, duration: float = 20.0) -> None:
        """Initialize timer with default duration."""
        self.duration: float = max(0.1, duration)
        self.time_remaining: float = self.duration
        self.is_running: bool = False
        self.is_expired: bool = False

    def start(self, duration: Optional[float] = None) -> None:
        """Start or restart the countdown timer."""
        if duration is not None:
            self.duration = max(0.1, duration)
        self.time_remaining = self.duration
        self.is_running = True
        self.is_expired = False

    def update(self, dt: float) -> bool:
        """Advance the timer by dt seconds.

        Args:
            dt: Delta time in seconds since previous frame.

        Returns:
            True if the timer expired during this tick, False otherwise.
        """
        if not self.is_running or self.is_expired:
            return False

        self.time_remaining -= dt
        if self.time_remaining <= 0.0:
            self.time_remaining = 0.0
            self.is_running = False
            self.is_expired = True
            return True

        return False

    def pause(self) -> None:
        """Pause the running countdown."""
        self.is_running = False

    def resume(self) -> None:
        """Resume the paused countdown if not expired."""
        if not self.is_expired and self.time_remaining > 0.0:
            self.is_running = True

    def stop(self) -> None:
        """Halt the timer completely."""
        self.is_running = False

    def reset(self) -> None:
        """Reset timer state back to initialized duration."""
        self.time_remaining = self.duration
        self.is_running = False
        self.is_expired = False

    @property
    def progress(self) -> float:
        """Return ratio of remaining time from 1.0 (full) down to 0.0 (expired)."""
        if self.duration <= 0.0:
            return 0.0
        return max(0.0, min(1.0, self.time_remaining / self.duration))

    @property
    def formatted_time(self) -> str:
        """Return human-readable remaining time string."""
        return f"{self.time_remaining:04.1f}s"
