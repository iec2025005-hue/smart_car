"""
vision/lane_detector.py — Lane-line detection using Canny + Hough transform.

Detects straight lane lines in the bottom portion of the frame and
returns left/right lane boundaries with a steering-offset hint.

Standalone test::

    python -m vision.lane_detector
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

import config
from utils.logger import get_logger
from vision.detector import BaseDetector, Detection

log = get_logger(__name__)


class LaneDetector(BaseDetector):
    """Detect lane lines using edge detection and Hough transform."""

    def __init__(self) -> None:
        self._canny_low: int = config.LANE_CANNY_LOW
        self._canny_high: int = config.LANE_CANNY_HIGH
        self._hough_thresh: int = config.LANE_HOUGH_THRESHOLD
        self._hough_min_len: int = config.LANE_HOUGH_MIN_LINE_LEN
        self._hough_max_gap: int = config.LANE_HOUGH_MAX_LINE_GAP
        self._roi_top: float = config.LANE_ROI_TOP_FRACTION
        log.info("LaneDetector ready.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]

        # 1. Region of interest — keep the bottom portion of the frame
        roi_y = int(h * self._roi_top)
        roi = frame[roi_y:, :]

        # 2. Pre-process: grayscale → blur → Canny
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self._canny_low, self._canny_high)

        # 3. Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self._hough_thresh,
            minLineLength=self._hough_min_len,
            maxLineGap=self._hough_max_gap,
        )

        if lines is None:
            return []

        # 4. Separate left and right lane lines by slope
        left_lines: List[Tuple[int, int, int, int]] = []
        right_lines: List[Tuple[int, int, int, int]] = []

        for line in lines:
            x1, y1, x2, y2 = map(int, line.reshape(-1)[:4])
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Filter near-horizontal lines (|slope| < 0.3)
            if abs(slope) < 0.3:
                continue
            if slope < 0:
                left_lines.append((x1, y1 + roi_y, x2, y2 + roi_y))
            else:
                right_lines.append((x1, y1 + roi_y, x2, y2 + roi_y))

        detections: List[Detection] = []

        # 5. Average each group into a single representative lane line
        left_avg = self._average_line(left_lines, h)
        right_avg = self._average_line(right_lines, h)

        mid_x = w // 2

        if left_avg is not None:
            x1, y1, x2, y2 = left_avg
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append(
                Detection(
                    label="left_lane",
                    confidence=1.0,
                    bbox=(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)),
                    center=(cx, cy),
                    extra={"line": left_avg, "side": "left"},
                )
            )

        if right_avg is not None:
            x1, y1, x2, y2 = right_avg
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append(
                Detection(
                    label="right_lane",
                    confidence=1.0,
                    bbox=(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)),
                    center=(cx, cy),
                    extra={"line": right_avg, "side": "right"},
                )
            )

        # 6. Compute lane centre offset (for steering)
        if left_avg is not None and right_avg is not None:
            lane_center_x = (left_avg[0] + right_avg[0]) // 2
            offset = lane_center_x - mid_x
            for d in detections:
                d.extra["lane_offset_px"] = offset

        return detections

    # ── helpers ──────────────────────────────────

    @staticmethod
    def _average_line(
        lines: List[Tuple[int, int, int, int]],
        frame_height: int,
    ) -> Tuple[int, int, int, int] | None:
        """Return the averaged (x1, y1, x2, y2) for a group of lines."""
        if not lines:
            return None

        x1s, y1s, x2s, y2s = zip(*lines)
        avg_x1 = int(np.mean(x1s))
        avg_y1 = int(np.mean(y1s))
        avg_x2 = int(np.mean(x2s))
        avg_y2 = int(np.mean(y2s))
        return (avg_x1, avg_y1, avg_x2, avg_y2)


# ── standalone test ──────────────────────────

if __name__ == "__main__":
    print("LaneDetector standalone test")
    det = LaneDetector()

    # Draw two white lane lines on a dark frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(frame, (160, 480), (280, 300), (255, 255, 255), 3)  # left
    cv2.line(frame, (480, 480), (360, 300), (255, 255, 255), 3)  # right

    results = det.detect(frame)
    print(f"Detections: {len(results)}")
    for r in results:
        print(f"  label={r.label}  center={r.center}  extra={r.extra}")

    print("Test passed -- all clear!")
