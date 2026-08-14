"""
vision/detector.py — Abstract base class and factory for vision detectors.

Every vision mode (YOLO, colour, ArUco, lane) implements ``BaseDetector``
so the main loop can swap pipelines via ``config.VISION_MODE`` without
touching any calling code.

Usage::

    from vision.detector import create_detector
    detector = create_detector()       # reads config.VISION_MODE
    results  = detector.detect(frame)  # list[Detection]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config


# ── shared data structure ────────────────────

@dataclass
class Detection:
    """Uniform result container returned by every detector.

    Attributes
    ----------
    label : str
        Human-readable label (e.g. "person", "red", "ArUco-7", "left_lane").
    confidence : float
        0.0 – 1.0 (set to 1.0 for non-ML detectors like ArUco / colour).
    bbox : tuple[int, int, int, int]
        Bounding box as ``(x, y, w, h)`` in pixel coordinates.
    center : tuple[int, int]
        Centre point ``(cx, cy)`` of the bounding box.
    extra : dict
        Detector-specific metadata (e.g. ArUco ID, lane angle, class index).
    """

    label: str = ""
    confidence: float = 1.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    center: Tuple[int, int] = (0, 0)
    extra: Dict[str, Any] = field(default_factory=dict)


# ── abstract base ────────────────────────────

class BaseDetector(ABC):
    """Interface that every vision-mode detector must implement."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H × W × 3, dtype uint8).

        Returns
        -------
        list[Detection]
        """

    def release(self) -> None:
        """Free resources (model handles, GPU memory, etc.).

        Override in subclasses that hold heavy resources.
        """


# ── factory ──────────────────────────────────

def create_detector(mode: Optional[str] = None) -> BaseDetector:
    """Instantiate the detector for the requested vision mode.

    Parameters
    ----------
    mode : str or None
        One of ``"yolo"``, ``"color"``, ``"aruco"``, ``"lane"``.
        Defaults to ``config.VISION_MODE``.

    Returns
    -------
    BaseDetector

    Raises
    ------
    ValueError
        If the mode string is not recognised.
    """
    mode = (mode or config.VISION_MODE).lower()

    if mode == "yolo":
        from vision.yolo_detector import YoloDetector
        return YoloDetector()
    elif mode == "color":
        from vision.color_detector import ColorDetector
        return ColorDetector()
    elif mode == "aruco":
        from vision.aruco_detector import ArucoDetector
        return ArucoDetector()
    elif mode == "lane":
        from vision.lane_detector import LaneDetector
        return LaneDetector()
    else:
        raise ValueError(
            f"Unknown VISION_MODE '{mode}'. "
            "Choose from: yolo, color, aruco, lane."
        )
