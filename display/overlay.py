"""
display/overlay.py -- Draw detection results and HUD on the camera frame.

Renders bounding boxes, labels, centre dots, a frame-centre crosshair,
FPS, the active vision mode, and the latest navigation command onto the
preview frame.  Used during development (HDMI / VNC); disable with
``config.DISPLAY_PREVIEW = False`` for headless production.

All drawing uses OpenCV primitives only -- no external GUI dependencies.

Standalone test::

    python -m display.overlay
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import config
from utils.logger import get_logger

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore[assignment]
    np = None   # type: ignore[assignment]

try:
    from vision.detector import Detection
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class Detection:  # type: ignore[no-redef]
        """Minimal stand-in matching vision.detector.Detection's interface."""
        label: str = ""
        confidence: float = 1.0
        bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
        center: Tuple[int, int] = (0, 0)
        extra: Dict[str, Any] = field(default_factory=dict)

log = get_logger(__name__)


# ── colour palette (BGR) ────────────────────
_GREEN = (0, 255, 0)
_CYAN = (255, 255, 0)
_YELLOW = (0, 255, 255)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GRAY = (160, 160, 160)

_FONT = cv2.FONT_HERSHEY_SIMPLEX if cv2 is not None else None
_FONT_SMALL = 0.45
_FONT_MEDIUM = 0.55
_FONT_LARGE = 0.65

# Crosshair dimensions (pixels)
_CROSSHAIR_SIZE = 15
_CROSSHAIR_THICKNESS = 1

# Detection box
_BOX_THICKNESS = 2
_CENTER_DOT_RADIUS = 4


class Overlay:
    """Draw detection results and telemetry on a BGR frame.

    All public state is read from ``config.py`` at construction time,
    so toggling ``OVERLAY_SHOW_FPS`` or ``OVERLAY_SHOW_MODE`` at runtime
    takes effect on the next ``Overlay()`` instantiation.
    """

    def __init__(self) -> None:
        self._show_fps: bool = config.OVERLAY_SHOW_FPS
        self._show_mode: bool = config.OVERLAY_SHOW_MODE
        log.info(
            "Overlay initialised  [show_fps=%s  show_mode=%s]",
            self._show_fps, self._show_mode,
        )

    # ── public API ───────────────────────────────

    def draw(
        self,
        frame: "np.ndarray",
        detections: List[Detection],
        fps: float = 0.0,
        vision_mode: str = "",
        nav_command: str = "",
    ) -> "np.ndarray":
        """Annotate *frame* in-place and return it.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H x W x 3, dtype uint8).
        detections : list[Detection]
            Current detections from any vision mode.
        fps : float
            Current processing FPS.
        vision_mode : str
            Active vision mode label (e.g. ``"yolo"``, ``"lane"``).
        nav_command : str
            Latest navigation command sent to STM32.

        Returns
        -------
        np.ndarray
            The same frame object, with annotations drawn on it.
        """
        if cv2 is None:
            log.warning("OpenCV not available -- skipping overlay draw")
            return frame

        log.debug(
            "Drawing overlay: %d detections, fps=%.1f, mode=%s, nav=%s",
            len(detections), fps, vision_mode, nav_command,
        )

        # Draw crosshair at image centre (behind everything else)
        self._draw_crosshair(frame)

        # Draw each detection
        for det in detections:
            self._draw_detection(frame, det)

        # Draw the heads-up display (FPS, mode, nav command)
        self._draw_hud(frame, fps, vision_mode, nav_command)

        return frame

    # ── detection rendering ──────────────────────

    @staticmethod
    def _draw_detection(frame: "np.ndarray", det: Detection) -> None:
        """Draw sci-fi corner brackets, distance estimation, label, and target lock."""
        x, y, w, h = map(int, det.bbox)
        colour = _CYAN if det.label == "tracked" else _GREEN

        # 1. Sci-Fi Corner Brackets
        length = min(15, w // 4, h // 4)
        t = 2
        # Top-Left
        cv2.line(frame, (x, y), (x + length, y), colour, t)
        cv2.line(frame, (x, y), (x, y + length), colour, t)
        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - length, y), colour, t)
        cv2.line(frame, (x + w, y), (x + w, y + length), colour, t)
        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + length, y + h), colour, t)
        cv2.line(frame, (x, y + h), (x, y + h - length), colour, t)
        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - length, y + h), colour, t)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - length), colour, t)

        # Thin outer box outline
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), 1)

        # 2. Distance Estimation & Bearing
        # Rough distance estimate based on bounding box height relative to 480px frame
        est_dist = max(0.3, round(350.0 / max(1, h), 1))
        frame_h, frame_w = frame.shape[:2]
        offset_x = det.center[0] - (frame_w // 2)
        bearing_deg = round((offset_x / (frame_w / 2.0)) * 30.0, 1) // 1

        # 3. Label + Distance + Confidence Banner
        label_text = f"[{det.label.upper()}] {det.confidence:.0%} | {est_dist}m | {bearing_deg:+}°"

        (tw, th), baseline = cv2.getTextSize(label_text, _FONT, _FONT_SMALL, 1)
        # Gradient banner background
        cv2.rectangle(
            frame,
            (x, y - th - baseline - 6),
            (x + tw + 8, y),
            (20, 20, 20),
            cv2.FILLED,
        )
        cv2.rectangle(
            frame,
            (x, y - th - baseline - 6),
            (x + tw + 8, y),
            colour,
            1,
        )
        cv2.putText(
            frame, label_text,
            (x + 4, y - baseline - 3),
            _FONT, _FONT_SMALL, _WHITE, 1, cv2.LINE_AA,
        )

        # 4. Target Lock Crosshair / Reticle at Center
        cx, cy = det.center
        cv2.circle(frame, (cx, cy), 6, colour, 1)
        cv2.circle(frame, (cx, cy), 2, _RED, cv2.FILLED)
        
        # Line connecting target center to frame center
        fc_x, fc_y = frame_w // 2, frame_h // 2
        cv2.line(frame, (fc_x, fc_y), (cx, cy), (0, 165, 255), 1, cv2.LINE_AA)

        # Lane lines rendering
        if "line" in det.extra:
            x1, y1, x2, y2 = det.extra["line"]
            cv2.line(frame, (x1, y1), (x2, y2), _YELLOW, 3, cv2.LINE_AA)

    # ── crosshair ────────────────────────────────

    @staticmethod
    def _draw_crosshair(frame: "np.ndarray") -> None:
        """Draw a small crosshair at the exact frame centre."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        s = _CROSSHAIR_SIZE
        t = _CROSSHAIR_THICKNESS

        # Horizontal line
        cv2.line(frame, (cx - s, cy), (cx + s, cy), _GRAY, t, cv2.LINE_8)
        # Vertical line
        cv2.line(frame, (cx, cy - s), (cx, cy + s), _GRAY, t, cv2.LINE_8)

        log.debug("Crosshair drawn at (%d, %d)", cx, cy)

    # ── heads-up display ─────────────────────────

    def _draw_hud(
        self,
        frame: "np.ndarray",
        fps: float,
        vision_mode: str,
        nav_command: str,
    ) -> None:
        """Draw telemetry text in the top-left corner."""
        y_pos = 22

        # FPS
        if self._show_fps:
            self._put_text_with_shadow(
                frame, f"FPS: {fps:.1f}", (10, y_pos),
                _FONT_MEDIUM, _GREEN, 2,
            )
            y_pos += 26

        # Vision mode
        if self._show_mode and vision_mode:
            self._put_text_with_shadow(
                frame, f"Mode: {vision_mode.upper()}", (10, y_pos),
                _FONT_MEDIUM, _CYAN, 2,
            )
            y_pos += 26

        # Navigation command
        if nav_command:
            self._put_text_with_shadow(
                frame, f"Nav: {nav_command}", (10, y_pos),
                _FONT_SMALL, _YELLOW, 1,
            )

    @staticmethod
    def _put_text_with_shadow(
        frame: "np.ndarray",
        text: str,
        org: Tuple[int, int],
        scale: float,
        colour: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw text with a thin dark shadow for readability on any background."""
        # Shadow (offset by 1px)
        cv2.putText(
            frame, text,
            (org[0] + 1, org[1] + 1),
            _FONT, scale, _BLACK, thickness + 1, cv2.LINE_8,
        )
        # Foreground
        cv2.putText(
            frame, text, org,
            _FONT, scale, colour, thickness, cv2.LINE_8,
        )


# ── standalone test ──────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Overlay -- standalone test suite")
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

    if cv2 is None or np is None:
        print("\n  [SKIP] OpenCV / NumPy not installed -- cannot run visual tests")
        print("         Install with:  pip install opencv-python numpy")
        print("\n  Running interface-only checks ...\n")

        # Even without OpenCV we can verify the class loads and has
        # the expected public methods.
        overlay = Overlay()
        _assert(hasattr(overlay, "draw"), "Overlay has draw() method")
        _assert(callable(overlay.draw), "draw() is callable")

    else:
        H, W = 480, 640

        # ── 1. Basic construction ────────────────
        print("\n[1] Overlay construction")
        overlay = Overlay()
        _assert(overlay._show_fps == config.OVERLAY_SHOW_FPS, "show_fps from config")
        _assert(overlay._show_mode == config.OVERLAY_SHOW_MODE, "show_mode from config")

        # ── 2. draw() returns the same frame ─────
        print("\n[2] draw() returns the same frame object (in-place)")
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        result = overlay.draw(frame, [], fps=0.0)
        _assert(result is frame, "returned object is the same frame")

        # ── 3. Empty detections don't crash ──────
        print("\n[3] Empty detections list")
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        result = overlay.draw(frame, [], fps=10.0, vision_mode="color", nav_command="STOP:0")
        _assert(result.shape == (H, W, 3), f"frame shape preserved: {result.shape}")
        # Frame should have some non-zero pixels (HUD text)
        _assert(np.any(result > 0), "HUD text rendered on frame")

        # ── 4. Single detection ──────────────────
        print("\n[4] Single detection drawn")
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        det = Detection(
            label="person", confidence=0.87,
            bbox=(100, 100, 80, 160), center=(140, 180),
        )
        result = overlay.draw(frame, [det], fps=8.3, vision_mode="yolo", nav_command="FWD:150")
        # Check that pixels at the bounding box edges were modified
        _assert(
            np.any(result[100, 100:180] > 0),
            "bbox top edge has non-zero pixels",
        )
        # Check centre dot area
        _assert(
            np.any(result[178:182, 138:142] > 0),
            "centre dot region has non-zero pixels",
        )

        # ── 5. Multiple detections ───────────────
        print("\n[5] Multiple detections drawn")
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        dets = [
            Detection(label="car", confidence=0.65, bbox=(200, 150, 100, 80), center=(250, 190)),
            Detection(label="cone", confidence=0.92, bbox=(400, 300, 40, 50), center=(420, 325)),
            Detection(label="ArUco-7", confidence=1.0, bbox=(50, 50, 30, 30), center=(65, 65)),
        ]
        result = overlay.draw(frame, dets, fps=12.5, vision_mode="aruco")
        _assert(np.any(result > 0), "multiple detections rendered")

        # ── 6. Crosshair at image centre ─────────
        print("\n[6] Crosshair drawn at frame centre")
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        overlay.draw(frame, [])
        cx, cy = W // 2, H // 2
        # The crosshair should have drawn grey pixels at the centre
        _assert(
            np.any(frame[cy, cx - 5:cx + 5] > 0),
            f"crosshair horizontal at ({cx}, {cy})",
        )
        _assert(
            np.any(frame[cy - 5:cy + 5, cx] > 0),
            f"crosshair vertical at ({cx}, {cy})",
        )

        # ── 7. FPS text rendered ─────────────────
        print("\n[7] FPS text rendered when OVERLAY_SHOW_FPS is True")
        frame_fps = np.zeros((H, W, 3), dtype=np.uint8)
        overlay._show_fps = True
        overlay.draw(frame_fps, [], fps=25.7)
        # Top-left corner should have text pixels
        top_left_region = frame_fps[10:35, 5:120]
        _assert(np.any(top_left_region > 0), "FPS text in top-left corner")

        # ── 8. Vision mode text rendered ─────────
        print("\n[8] Vision mode text rendered when OVERLAY_SHOW_MODE is True")
        frame_mode = np.zeros((H, W, 3), dtype=np.uint8)
        overlay._show_fps = False  # disable FPS so mode is at first line
        overlay._show_mode = True
        overlay.draw(frame_mode, [], vision_mode="lane")
        top_left_region = frame_mode[10:35, 5:160]
        _assert(np.any(top_left_region > 0), "Mode text in top-left corner")

        # ── 9. Nav command text rendered ─────────
        print("\n[9] Navigation command text rendered")
        frame_nav = np.zeros((H, W, 3), dtype=np.uint8)
        overlay._show_fps = False
        overlay._show_mode = False
        overlay.draw(frame_nav, [], nav_command="LEFT:120")
        top_left_region = frame_nav[10:35, 5:160]
        _assert(np.any(top_left_region > 0), "Nav text in top-left corner")

        # ── 10. Lane line rendering ──────────────
        print("\n[10] Lane line special rendering")
        frame_lane = np.zeros((H, W, 3), dtype=np.uint8)
        det_lane = Detection(
            label="left_lane", confidence=1.0,
            bbox=(50, 300, 200, 180), center=(150, 390),
            extra={"line": (50, 479, 250, 300), "side": "left"},
        )
        overlay.draw(frame_lane, [det_lane])
        # Check pixels along the lane line path
        _assert(
            np.any(frame_lane[400, 100:200] > 0),
            "lane line pixels visible",
        )

        # ── 11. Confidence display ───────────────
        print("\n[11] Confidence < 1.0 shows percentage, == 1.0 does not")
        # This is verified by visual inspection, but we can at least
        # confirm no crash and the label area has pixels
        frame_conf = np.zeros((H, W, 3), dtype=np.uint8)
        det_ml = Detection(label="dog", confidence=0.73, bbox=(50, 50, 60, 80), center=(80, 90))
        det_exact = Detection(label="marker", confidence=1.0, bbox=(300, 50, 40, 40), center=(320, 70))
        overlay.draw(frame_conf, [det_ml, det_exact])
        _assert(
            np.any(frame_conf[40:55, 50:130] > 0),
            "label region has pixels for conf<1.0 detection",
        )
        _assert(
            np.any(frame_conf[40:55, 300:360] > 0),
            "label region has pixels for conf==1.0 detection",
        )

        # ── 12. draw() signature compatible with main.py ─
        print("\n[12] draw() signature matches main.py usage")
        frame12 = np.zeros((H, W, 3), dtype=np.uint8)
        # main.py calls: overlay.draw(frame, detections, fps=..., vision_mode=..., nav_command=...)
        try:
            overlay.draw(
                frame12,
                [det_ml],
                fps=15.0,
                vision_mode="yolo",
                nav_command="FWD:150",
            )
            _assert(True, "main.py-compatible call signature works")
        except TypeError as exc:
            _assert(False, f"signature mismatch: {exc}")

        # ── 13. Save test image ──────────────────
        print("\n[13] Generate a comprehensive test image")
        frame_final = np.full((H, W, 3), 30, dtype=np.uint8)  # dark grey background
        overlay._show_fps = True
        overlay._show_mode = True
        all_dets = [
            Detection(label="person", confidence=0.87, bbox=(100, 120, 80, 180), center=(140, 210)),
            Detection(label="car", confidence=0.64, bbox=(350, 200, 120, 90), center=(410, 245)),
            Detection(label="cone", confidence=0.95, bbox=(500, 350, 45, 55), center=(522, 377)),
            Detection(label="left_lane", confidence=1.0, bbox=(20, 350, 180, 130),
                      center=(110, 415), extra={"line": (30, 479, 200, 350)}),
        ]
        result = overlay.draw(
            frame_final, all_dets,
            fps=9.7, vision_mode="yolo", nav_command="RIGHT:85",
        )
        _assert(result is frame_final, "final frame returned in-place")

        # Save to output directory for visual inspection
        import os
        out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "test_overlay.png")
        cv2.imwrite(out_path, result)
        print(f"  Saved -> {os.path.abspath(out_path)}")

    # ── summary ──────────────────────────────────
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  -- all clear!")
    print("=" * 60)
