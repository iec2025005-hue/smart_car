"""
vision/aruco_detector.py — ArUco fiducial marker detection.

Detects ArUco markers and optionally estimates their 6-DOF pose
when camera calibration data is available in ``config.py``.

Standalone test::

    python -m vision.aruco_detector
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

import config
from utils.logger import get_logger
from vision.detector import BaseDetector, Detection

log = get_logger(__name__)


class ArucoDetector(BaseDetector):
    """Detect ArUco markers in a frame."""

    def __init__(self) -> None:
        dict_name = config.ARUCO_DICT_TYPE
        aruco_dict_id = getattr(cv2.aruco, dict_name, None)
        if aruco_dict_id is None:
            raise ValueError(
                f"Unknown ArUco dictionary '{dict_name}'. "
                "See cv2.aruco.DICT_* for valid names."
            )

        self._aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self._params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._params)

        # Camera calibration (optional — needed for pose estimation)
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        if config.ARUCO_CAMERA_MATRIX is not None:
            self._camera_matrix = np.array(config.ARUCO_CAMERA_MATRIX, dtype=np.float64)
            self._dist_coeffs = np.array(
                config.ARUCO_DIST_COEFFS if config.ARUCO_DIST_COEFFS is not None
                else [0, 0, 0, 0, 0],
                dtype=np.float64,
            )
            log.info("ArUco pose estimation enabled (calibration loaded).")

        log.info("ArucoDetector ready  [dict=%s]", dict_name)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self._detector.detectMarkers(gray)

        detections: List[Detection] = []
        if ids is None:
            return detections

        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i][0]                       # 4×2 corner array
            x_min, y_min = pts.min(axis=0).astype(int)
            x_max, y_max = pts.max(axis=0).astype(int)
            w, h = x_max - x_min, y_max - y_min
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())

            extra = {"marker_id": int(marker_id), "corners": pts.tolist()}

            # Pose estimation (if calibrated)
            if self._camera_matrix is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i:i + 1],
                    config.ARUCO_MARKER_SIZE_CM,
                    self._camera_matrix,
                    self._dist_coeffs,
                )
                extra["rvec"] = rvecs[0][0].tolist()
                extra["tvec"] = tvecs[0][0].tolist()
                extra["distance_cm"] = float(np.linalg.norm(tvecs[0][0]))

            detections.append(
                Detection(
                    label=f"ArUco-{marker_id}",
                    confidence=1.0,
                    bbox=(x_min, y_min, w, h),
                    center=(cx, cy),
                    extra=extra,
                )
            )

        return detections


# ── standalone test ──────────────────────────

if __name__ == "__main__":
    print("ArucoDetector standalone test")
    det = ArucoDetector()

    # Blank frame — no markers expected
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    results = det.detect(blank)
    print(f"Detections on blank frame: {len(results)}")
    assert len(results) == 0

    # Generate a synthetic marker image
    marker_img = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        id=7,
        sidePixels=200,
    )
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    frame[140:340, 220:420, :] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    results = det.detect(frame)
    print(f"Detections on marker frame: {len(results)}")
    for r in results:
        print(f"  label={r.label}  center={r.center}")

    assert len(results) == 1 and results[0].extra["marker_id"] == 7
    print("Test passed -- all clear!")
