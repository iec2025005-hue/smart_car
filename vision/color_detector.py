"""
vision/color_detector.py — HSV colour-range detection.

Detects contiguous regions of a target colour using HSV thresholding.
Supports dual HSV ranges to handle hue wrap-around (e.g. red).

Standalone test::

    python -m vision.color_detector
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

import config
from utils.logger import get_logger
from vision.detector import BaseDetector, Detection

log = get_logger(__name__)


class ColorDetector(BaseDetector):
    """Detect objects by HSV colour range."""

    def __init__(self):
        self._lower1 = np.array(config.COLOR_HSV_LOWER_1, dtype=np.uint8)
        self._upper1 = np.array(config.COLOR_HSV_UPPER_1, dtype=np.uint8)
        self._lower2 = np.array(config.COLOR_HSV_LOWER_2, dtype=np.uint8)
        self._upper2 = np.array(config.COLOR_HSV_UPPER_2, dtype=np.uint8)
        self._min_area: int = config.COLOR_MIN_AREA
        log.info(
            "ColorDetector ready  [HSV %s–%s | %s–%s, min_area=%d]",
            self._lower1, self._upper1,
            self._lower2, self._upper2,
            self._min_area,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Two masks to handle hue wrap-around (e.g. red spans 0-10 & 170-180)
        mask1 = cv2.inRange(hsv, self._lower1, self._upper1)
        mask2 = cv2.inRange(hsv, self._lower2, self._upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        detections: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            detections.append(
                Detection(
                    label="color_target",
                    confidence=1.0,
                    bbox=(x, y, w, h),
                    center=(cx, cy),
                    extra={"area": area},
                )
            )

        # Sort by area descending (largest blob first)
        detections.sort(key=lambda d: d.extra.get("area", 0), reverse=True)
        return detections


# ── standalone test ──────────────────────────

if __name__ == "__main__":
    print("ColorDetector standalone test")
    det = ColorDetector()

    # Create a frame with a red rectangle
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (400, 350), (0, 0, 255), -1)  # BGR red

    results = det.detect(frame)
    print(f"Detections: {len(results)}")
    for r in results:
        print(f"  label={r.label}  bbox={r.bbox}  area={r.extra['area']}")

    assert len(results) >= 1, "Should detect the red rectangle"
    print("Test passed -- all clear!")
