"""
navigation/navigator.py -- High-level navigation controller for Swerve Drive.

Consumes ``Detection`` results from any vision mode and translates
them into holonomic swerve commands sent via ``SerialComm``.

The navigator supports multiple modes (``config.NAVIGATION_MODE``):
  - ``idle``    -- motors stopped; vision runs but no commands are sent.
  - ``follow``  -- move forward and strafe to keep the target centered.
  - ``avoid``   -- strafe away from the highest-confidence detection.
  - ``patrol``  -- drive forward; stop when an obstacle is detected.

Standalone test::

    python -m navigation.navigator
"""

from __future__ import annotations

from typing import Callable, List, Optional

import config
from navigation.pid import PIDController
from navigation.swerve_kinematics import SwerveKinematics
from utils.logger import get_logger

try:
    from vision.detector import Detection
except ImportError:
    # Lightweight fallback for environments without numpy / full vision stack.
    from dataclasses import dataclass, field
    from typing import Any, Dict, Tuple

    @dataclass
    class Detection:  # type: ignore[no-redef]
        """Minimal stand-in matching vision.detector.Detection's interface."""
        label: str = ""
        confidence: float = 1.0
        bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
        center: Tuple[int, int] = (0, 0)
        extra: Dict[str, Any] = field(default_factory=dict)

log = get_logger(__name__)


class Navigator:
    """Translate vision detections into swerve motor commands.

    Parameters
    ----------
    send_fn : callable(str) -> None, optional
        A function that sends a command string to the motor controller
        (typically ``SerialComm.send``).  If *None*, commands are
        computed and returned but not dispatched.
    """

    def __init__(self, send_fn: Optional[Callable[[str], None]] = None) -> None:
        self._send: Optional[Callable[[str], None]] = send_fn
        self._mode: str = config.NAVIGATION_MODE.lower()
        self._frame_w: int = config.CAMERA_WIDTH
        self._frame_cx: int = self._frame_w // 2
        self._tolerance: int = config.FRAME_CENTER_TOLERANCE
        self._base_speed: int = config.BASE_SPEED
        self._turn_speed: float = config.TURN_SPEED

        self._kinematics = SwerveKinematics()

        self._pid = PIDController(
            kp=config.PID_KP,
            ki=config.PID_KI,
            kd=config.PID_KD,
            output_limits=config.PID_OUTPUT_LIMITS,
        )

        log.info(
            "Navigator (Swerve) ready  [mode=%s  base_speed=%d  tolerance=%dpx  "
            "frame_center=%dpx]",
            self._mode, self._base_speed, self._tolerance, self._frame_cx,
        )

    # ── public API ───────────────────────────────

    def update(self, detections: List[Detection]) -> str:
        """Compute and send a motor command based on current detections.

        The *highest-confidence* detection is used as the primary target
        for follow / avoid modes.
        """
        if self._mode == "idle":
            log.debug("Mode is idle -- stopping")
            return self._cmd_stop()

        if not detections:
            log.debug("No detections -- stopping")
            return self._cmd_stop()

        # Select the highest-confidence detection as the primary target.
        primary = max(detections, key=lambda d: d.confidence)
        log.debug(
            "Primary target: label='%s'  confidence=%.2f  center=%s  "
            "bbox=%s  (of %d detections)",
            primary.label, primary.confidence, primary.center,
            primary.bbox, len(detections),
        )

        if self._mode == "follow":
            return self._follow(primary)
        elif self._mode == "avoid":
            return self._avoid(primary)
        elif self._mode == "patrol":
            return self._patrol(detections)
        else:
            log.warning("Unknown nav mode '%s', stopping.", self._mode)
            return self._cmd_stop()

    def set_mode(self, mode: str) -> None:
        """Change navigation mode at runtime."""
        old_mode = self._mode
        self._mode = mode.lower()
        self._pid.reset()
        log.info(
            "Navigation mode changed: '%s' -> '%s'", old_mode, self._mode,
        )

    # ── strategies ───────────────────────────────

    def _follow(self, det: Detection) -> str:
        """Move forward and strafe to center the target using PID correction."""
        error = det.center[0] - self._frame_cx
        correction = self._pid.compute(error)

        log.debug(
            "follow: error=%+dpx  correction=%.1f  tolerance=%dpx",
            error, correction, self._tolerance,
        )

        if abs(error) < self._tolerance:
            return self._cmd_swerve(vx=self._base_speed, vy=0, omega=0)
        else:
            # Swerve magic: Strafe (vy) to center while moving forward (vx)
            return self._cmd_swerve(vx=self._base_speed, vy=correction, omega=0)

    def _avoid(self, det: Detection) -> str:
        """Strafe away from the detection centre."""
        error = det.center[0] - self._frame_cx

        log.debug("avoid: error=%+dpx  tolerance=%dpx", error, self._tolerance)

        if abs(error) < self._tolerance:
            # Object is dead-centre -- strafe right arbitrarily
            return self._cmd_swerve(vx=0, vy=self._base_speed, omega=0)
        elif error < 0:
            # Object is on the left -- strafe right
            return self._cmd_swerve(vx=0, vy=self._base_speed, omega=0)
        else:
            # Object is on the right -- strafe left
            return self._cmd_swerve(vx=0, vy=-self._base_speed, omega=0)

    def _patrol(self, detections: List[Detection]) -> str:
        """Drive forward; stop when something is too close."""
        for det in detections:
            _, _, w, _ = det.bbox
            if w > self._frame_w * 0.40:
                log.debug(
                    "patrol: obstacle '%s' is too close (bbox_w=%d > %.0fpx)",
                    det.label, w, self._frame_w * 0.40,
                )
                return self._cmd_stop()

        log.debug("patrol: path clear -- driving forward at %d", self._base_speed)
        return self._cmd_swerve(vx=self._base_speed, vy=0, omega=0)

    # ── command helpers ──────────────────────────

    def _send_cmd(self, cmd: str) -> str:
        """Dispatch *cmd* to the motor controller and return it."""
        if self._send is not None:
            self._send(cmd)
            log.debug("Sent command: %s", cmd)
        else:
            log.debug("Command (no send_fn): %s", cmd)
        return cmd

    def _cmd_swerve(self, vx: float, vy: float, omega: float) -> str:
        """Calculate swerve angles and speeds and generate the command string."""
        wheels = self._kinematics.calculate(vx, vy, omega)
        # Format: SWERVE:a_fr,s_fr,a_fl,s_fl,a_rl,s_rl,a_rr,s_rr
        parts = []
        for angle, speed in wheels:
            parts.extend([f"{angle:.1f}", f"{speed:.1f}"])
        cmd_str = "SWERVE:" + ",".join(parts)
        return self._send_cmd(cmd_str)

    def _cmd_stop(self) -> str:
        return self._cmd_swerve(0, 0, 0)


# ── standalone test ──────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Navigator (Swerve) -- standalone test suite")
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

    # Helper: collect sent commands
    commands: list[str] = []

    def _capture(cmd: str) -> None:
        commands.append(cmd)

    # ── 1. Idle mode stops ───────────────────────
    print("\n[1] Idle mode always sends STOP (all zeros)")
    commands.clear()
    nav = Navigator(send_fn=_capture)
    nav.set_mode("idle")
    det = Detection(label="person", confidence=0.9, bbox=(400, 200, 50, 50), center=(425, 225))
    cmd = nav.update([det])
    _assert(cmd.startswith("SWERVE:0.0,0.0"), f"idle + detection -> '{cmd}'")

    # ── 2. Follow mode strafes ───────────────────
    print("\n[2] Follow mode strafes to center detection")
    nav.set_mode("follow")
    nav._pid.reset()
    
    # Target right of center (cx=425, frame_cx=320). Should strafe right.
    # A right strafe means angles should be around 90 deg for all wheels.
    cmd = nav.update([det])
    parts = cmd.split(":")[1].split(",")
    fr_angle = float(parts[0])
    _assert(fr_angle > 0, f"follows object to the right (angle={fr_angle}) -> '{cmd}'")

    # ── summary ──────────────────────────────────
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  -- all clear!")
    print("=" * 60)
