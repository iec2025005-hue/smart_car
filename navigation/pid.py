"""
navigation/pid.py — Reusable PID controller with logging and safety guards.

Reads default gains and output limits from ``config.py``
(``PID_KP``, ``PID_KI``, ``PID_KD``, ``PID_OUTPUT_LIMITS``).

Features
--------
* Integral windup protection (integral term clamped to output range).
* Output clamping to configurable ``(min, max)`` limits.
* Safe handling of ``dt = 0`` (derivative term is suppressed).
* Structured logging via the project's ``utils.logger`` module.
* Backward-compatible ``compute(error)`` method (auto-measures dt).
* Explicit ``update(error, dt)`` method for deterministic control loops.

Standalone test::

    python -m navigation.pid
"""

from __future__ import annotations

import time
from typing import Tuple

import config
from utils.logger import get_logger

log = get_logger(__name__)


class PIDController:
    """Discrete PID controller with output clamping and integral windup guard.

    Parameters
    ----------
    kp : float, optional
        Proportional gain. Defaults to ``config.PID_KP``.
    ki : float, optional
        Integral gain. Defaults to ``config.PID_KI``.
    kd : float, optional
        Derivative gain. Defaults to ``config.PID_KD``.
    output_limits : tuple[float, float], optional
        ``(min, max)`` clamp applied to the controller output.
        Defaults to ``config.PID_OUTPUT_LIMITS``.
    """

    def __init__(
        self,
        kp: float | None = None,
        ki: float | None = None,
        kd: float | None = None,
        output_limits: Tuple[float, float] | None = None,
    ) -> None:
        self.kp: float = kp if kp is not None else config.PID_KP
        self.ki: float = ki if ki is not None else config.PID_KI
        self.kd: float = kd if kd is not None else config.PID_KD

        limits = output_limits if output_limits is not None else config.PID_OUTPUT_LIMITS
        self._min_out: float = float(limits[0])
        self._max_out: float = float(limits[1])

        # Internal state
        self._prev_error: float = 0.0
        self._integral: float = 0.0
        self._last_time: float | None = None

        log.info(
            "PID initialised  [Kp=%.4f  Ki=%.4f  Kd=%.4f  limits=(%.1f, %.1f)]",
            self.kp, self.ki, self.kd, self._min_out, self._max_out,
        )

    # ── core computation ────────────────────────

    def update(self, error: float, dt: float) -> float:
        """Compute the PID output for a given *error* and explicit *dt*.

        This is the preferred entry-point when the caller already knows
        the time-step (e.g. from a fixed-rate control loop).

        Parameters
        ----------
        error : float
            Signed error signal (e.g. pixel offset from frame centre).
        dt : float
            Time elapsed since the last call, in seconds.
            If *dt* ≤ 0 the derivative term is suppressed and the
            integral is not accumulated, preventing division by zero
            and spurious wind-up.

        Returns
        -------
        float
            Controller output, clamped to ``output_limits``.
        """
        # ── Proportional ────────────────────────
        p_term = self.kp * error

        # ── Integral (with anti-windup clamp) ───
        if dt > 0:
            self._integral += error * dt
            # Clamp the raw integral accumulator to the output range
            # so it cannot grow unboundedly ("integral windup").
            self._integral = max(self._min_out,
                                 min(self._max_out, self._integral))
        i_term = self.ki * self._integral

        # ── Derivative ──────────────────────────
        if dt > 0:
            d_term = self.kd * ((error - self._prev_error) / dt)
        else:
            d_term = 0.0
            log.debug("dt=0 — derivative term suppressed")

        self._prev_error = error

        # ── Sum & clamp output ──────────────────
        raw_output = p_term + i_term + d_term
        output = max(self._min_out, min(self._max_out, raw_output))

        log.debug(
            "PID  error=%+8.2f  P=%+8.2f  I=%+8.2f  D=%+8.2f  "
            "raw=%+8.2f  out=%+8.2f  dt=%.4f",
            error, p_term, i_term, d_term, raw_output, output, dt,
        )

        return output

    # ── backward-compatible auto-dt method ──────

    def compute(self, error: float) -> float:
        """Compute the PID output, automatically measuring *dt*.

        Kept for backward compatibility with callers that do not track
        their own time-step (e.g. ``Navigator._follow``).

        Parameters
        ----------
        error : float
            Signed error signal.

        Returns
        -------
        float
            Controller output, clamped to ``output_limits``.
        """
        now = time.monotonic()
        if self._last_time is not None:
            dt = now - self._last_time
        else:
            # First call — assume a reasonable default frame interval.
            dt = 0.02
        self._last_time = now
        return self.update(error, dt)

    # ── state management ────────────────────────

    def reset(self) -> None:
        """Zero all internal state (call on mode change or re-init)."""
        self._prev_error = 0.0
        self._integral = 0.0
        self._last_time = None
        log.info("PID controller reset")


# ── standalone test ──────────────────────────────

if __name__ == "__main__":
    import math

    print("=" * 60)
    print("  PIDController -- standalone test suite")
    print("=" * 60)

    passed = 0
    failed = 0

    def _assert(cond: bool, tag: str) -> None:
        global passed, failed
        if cond:
            passed += 1
            print(f"  [PASS] {tag}")
        else:
            failed += 1
            print(f"  [FAIL] {tag}")

    # ── 1. Defaults from config ──────────────────
    print("\n[1] Default gains from config.py")
    pid_default = PIDController()
    _assert(pid_default.kp == config.PID_KP, f"kp == {config.PID_KP}")
    _assert(pid_default.ki == config.PID_KI, f"ki == {config.PID_KI}")
    _assert(pid_default.kd == config.PID_KD, f"kd == {config.PID_KD}")
    _assert(
        (pid_default._min_out, pid_default._max_out) ==
        tuple(float(v) for v in config.PID_OUTPUT_LIMITS),
        f"output_limits == {config.PID_OUTPUT_LIMITS}",
    )

    # ── 2. Pure P-controller ─────────────────────
    print("\n[2] Pure proportional controller (ki=0, kd=0)")
    pid_p = PIDController(kp=1.0, ki=0.0, kd=0.0, output_limits=(-100, 100))
    out = pid_p.update(50.0, dt=0.02)
    _assert(abs(out - 50.0) < 0.01, f"P-only: error=50 -> output={out:.2f}")

    out = pid_p.update(-30.0, dt=0.02)
    _assert(abs(out - (-30.0)) < 0.01, f"P-only: error=-30 -> output={out:.2f}")

    # ── 3. Output clamping ───────────────────────
    print("\n[3] Output clamping")
    pid_clamp = PIDController(kp=1.0, ki=0.0, kd=0.0, output_limits=(-100, 100))
    out = pid_clamp.update(200.0, dt=0.02)
    _assert(out == 100.0, f"Clamped high: error=200 -> output={out:.1f}")

    out = pid_clamp.update(-999.0, dt=0.02)
    _assert(out == -100.0, f"Clamped low: error=-999 -> output={out:.1f}")

    # ── 4. Integral accumulation ─────────────────
    print("\n[4] Integral accumulation")
    pid_i = PIDController(kp=0.0, ki=1.0, kd=0.0, output_limits=(-1000, 1000))
    # Constant error of 10 for 5 steps at dt=0.1 → integral = 10*0.5 = 5.0
    for _ in range(5):
        out = pid_i.update(10.0, dt=0.1)
    _assert(abs(out - 5.0) < 0.01, f"Integral after 5 steps: output={out:.2f}")

    # ── 5. Integral windup protection ────────────
    print("\n[5] Integral windup protection")
    pid_wu = PIDController(kp=0.0, ki=1.0, kd=0.0, output_limits=(-50, 50))
    for _ in range(1000):
        pid_wu.update(100.0, dt=0.1)
    _assert(
        pid_wu._integral <= 50.0,
        f"Integral capped at max: _integral={pid_wu._integral:.2f}",
    )

    # Negative windup
    pid_wu.reset()
    for _ in range(1000):
        pid_wu.update(-100.0, dt=0.1)
    _assert(
        pid_wu._integral >= -50.0,
        f"Integral capped at min: _integral={pid_wu._integral:.2f}",
    )

    # ── 6. Derivative term ───────────────────────
    print("\n[6] Derivative term")
    pid_d = PIDController(kp=0.0, ki=0.0, kd=1.0, output_limits=(-1000, 1000))
    pid_d.update(0.0, dt=0.1)  # establish baseline
    out = pid_d.update(10.0, dt=0.1)
    expected_d = 1.0 * (10.0 - 0.0) / 0.1  # = 100.0
    _assert(abs(out - expected_d) < 0.01, f"D-term: d_error=10, dt=0.1 -> output={out:.2f}")

    # ── 7. dt = 0 safety ─────────────────────────
    print("\n[7] dt=0 handling (no crash, derivative suppressed)")
    pid_z = PIDController(kp=1.0, ki=0.5, kd=0.5, output_limits=(-100, 100))
    integral_before = pid_z._integral
    out = pid_z.update(10.0, dt=0.0)
    _assert(math.isfinite(out), f"Finite output with dt=0: {out:.2f}")
    _assert(
        pid_z._integral == integral_before,
        "Integral unchanged when dt=0",
    )

    # ── 8. Reset clears state ────────────────────
    print("\n[8] Reset clears all internal state")
    pid_r = PIDController(kp=1.0, ki=1.0, kd=1.0, output_limits=(-100, 100))
    pid_r.update(50.0, dt=0.1)
    pid_r.compute(50.0)
    pid_r.reset()
    _assert(pid_r._prev_error == 0.0, "_prev_error reset to 0")
    _assert(pid_r._integral == 0.0, "_integral reset to 0")
    _assert(pid_r._last_time is None, "_last_time reset to None")

    # ── 9. compute() backward compatibility ──────
    print("\n[9] compute() auto-dt backward compatibility")
    pid_bc = PIDController(kp=1.0, ki=0.0, kd=0.0, output_limits=(-100, 100))
    out = pid_bc.compute(50.0)
    _assert(abs(out - 50.0) < 1.0, f"compute() P-only: output={out:.2f}")

    # ── 10. Full PID (all three terms) ───────────
    print("\n[10] Combined P + I + D")
    pid_full = PIDController(kp=1.0, ki=0.5, kd=0.2, output_limits=(-500, 500))
    pid_full.update(0.0, dt=0.1)  # baseline
    out = pid_full.update(10.0, dt=0.1)
    # P = 10, I = 0.5*(0+10*0.1)=0.5, D = 0.2*(10/0.1)=20
    expected = 10.0 + 0.5 * (10.0 * 0.1) + 0.2 * (10.0 / 0.1)
    _assert(abs(out - expected) < 0.01, f"Full PID: output={out:.2f} (expected {expected:.2f})")

    # ── summary ──────────────────────────────────
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  -- all clear!")
    print("=" * 60)
