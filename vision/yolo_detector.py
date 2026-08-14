"""
vision/yolo_detector.py — YOLOv8-nano object detection.

Uses the Ultralytics library to run inference on each frame.
Optimised for Raspberry Pi 4 CPU (no CUDA).

Standalone test::

    python -m vision.yolo_detector
"""

from __future__ import annotations

from typing import List

import numpy as np

import config
from utils.logger import get_logger
from vision.detector import BaseDetector, Detection

log = get_logger(__name__)


class YoloDetector(BaseDetector):
    """YOLOv8-nano detector using the Ultralytics API."""

    def __init__(self) -> None:
        try:
            import logging
            logging.getLogger("ultralytics").setLevel(logging.WARNING)
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "ultralytics package is not installed. "
                "Install with: pip install ultralytics"
            )

        log.info("Loading YOLO model from %s ...", config.MODEL_PATH)
        self._model = YOLO(config.MODEL_PATH)
        log.info("YOLO model ready.")

    # ── BaseDetector interface ───────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._model is None:
            return []

        results = self._model(
            frame,
            imgsz=320,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.NMS_THRESHOLD,
            max_det=50,
            verbose=False,
            classes=config.YOLO_TARGET_CLASSES,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2
                label = self._model.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])

                detections.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        bbox=(x1, y1, w, h),
                        center=(cx, cy),
                        extra={"class_id": cls_id},
                    )
                )
        return detections

    def set_target(self, target: str | int | List[int] | None) -> None:
        """Set active target class(es) by name or ID. Pass None, 'all', or 'any' to detect everything."""
        if target is None or str(target).strip().lower() in ("all", "none", "any", ""):
            config.YOLO_TARGET_CLASSES = None
            log.info("YOLO target set to: ALL")
            return

        if isinstance(target, list):
            config.YOLO_TARGET_CLASSES = target
            log.info("YOLO target set to IDs: %s", target)
            return

        target_str = str(target).strip().lower()
        matched_ids = []
        if self._model and hasattr(self._model, "names"):
            for cid, name in self._model.names.items():
                if target_str == str(cid) or target_str == str(name).lower():
                    matched_ids.append(cid)

        if matched_ids:
            config.YOLO_TARGET_CLASSES = matched_ids
            log.info("YOLO target set to: '%s' (IDs: %s)", target_str, matched_ids)
        else:
            log.warning("Unknown target '%s'. Target unchanged.", target)

    def release(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None  # type: ignore[assignment]
            log.info("YOLO model released.")



# ── standalone test ──────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  YoloDetector -- standalone test suite")
    print("=" * 60)

    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        print("\n  [SKIP] ultralytics package not installed -- skipping YOLO test")
        print("         Install with:  pip install ultralytics")
        print("=" * 60)
        import sys
        sys.exit(0)

    det = YoloDetector()

    # Create a dummy 640x480 black frame
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    results = det.detect(dummy)
    print(f"Detections on blank frame: {len(results)}")
    for r in results:
        print(f"  label={r.label} conf={r.confidence:.2f}")

    det.release()
    print("Test passed -- all clear!")
    print("=" * 60)
