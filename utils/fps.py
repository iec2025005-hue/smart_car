"""
utils/fps.py — Lightweight FPS (frames-per-second) counter.

Usage::

    fps = FPSCounter()
    while running:
        fps.tick()
        print(f"{fps.get():.1f} FPS")
"""

from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Sliding-window FPS counter.

    Maintains a deque of the last *window* timestamps and computes
    the average frame rate over that window.  This gives a stable,
    human-readable number rather than a jittery per-frame calculation.

    Parameters
    ----------
    window : int
        Number of recent frames to average over (default 30).
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        """Record the current timestamp."""
        self._timestamps.append(time.monotonic())

    def get(self) -> float:
        """Return the current average FPS (0.0 if fewer than 2 ticks)."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        self._timestamps.clear()


# ── standalone test ──────────────────────────

if __name__ == "__main__":
    print("utils.fps standalone test")
    counter = FPSCounter(window=5)
    assert counter.get() == 0.0
    counter.tick()
    time.sleep(0.01)
    counter.tick()
    fps_val = counter.get()
    assert fps_val > 0
    print(f"Measured FPS: {fps_val:.1f}")
    counter.reset()
    assert counter.get() == 0.0
    print("Test passed ✓")

