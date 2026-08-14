"""
vision/camera.py — Camera capture for the Raspberry Pi Camera Module 2.

Default back-end: **picamera2** (CSI ribbon camera via libcamera).
Fallback back-end: **opencv** (USB webcam only — not recommended for this
hardware setup).

The back-end is selected via ``config.CAMERA_BACKEND`` (default: "picamera2").

Installation (on Raspberry Pi OS Bookworm)
──────────────────────────────────────────
    sudo apt update && sudo apt install -y python3-picamera2

picamera2 ships as a system package on Bookworm and depends on libcamera,
which is already included in the default Pi OS image.  Do NOT install
picamera2 via pip — the apt package includes the required libcamera
bindings that pip cannot provide.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import numpy as np

import config
from utils.logger import get_logger

log = get_logger(__name__)


class Camera:
    """Capture frames from the Raspberry Pi Camera Module 2 (CSI).

    Preferred usage::

        with Camera() as cam:
            frame = cam.read()   # BGR numpy array
    """

    def __init__(self) -> None:
        self._picam = None          # Picamera2 instance (CSI)
        self._cap = None            # cv2.VideoCapture instance (USB fallback)
        self._backend: str = config.CAMERA_BACKEND.lower()
        self._frame_count: int = 0
        self._start_time: float = 0.0

    # ── lifecycle ────────────────────────────────

    def open(self) -> None:
        """Initialise the selected camera back-end.

        Raises
        ------
        RuntimeError
            If picamera2 is not installed or the camera cannot be opened.
        """
        if self._backend == "picamera2":
            try:
                self._open_picamera2()
            except Exception as exc:
                log.warning("picamera2 unavailable (%s) — falling back to OpenCV webcam backend.", exc)
                self._backend = "opencv"
                self._open_opencv()
        elif self._backend == "opencv":
            self._open_opencv()
        else:
            raise ValueError(
                f"Unknown CAMERA_BACKEND '{self._backend}'. "
                "Expected 'picamera2' or 'opencv'."
            )

        self._start_time = time.time()
        log.info(
            "Camera opened  [backend=%s, %dx%d @ %d fps]",
            self._backend,
            config.CAMERA_WIDTH,
            config.CAMERA_HEIGHT,
            config.CAMERA_FPS,
        )

    def close(self) -> None:
        """Release camera resources."""
        if self._picam is not None:
            try:
                self._picam.stop()
            except Exception:  # noqa: BLE001
                pass
            self._picam = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        elapsed = time.time() - self._start_time if self._start_time else 0
        avg_fps = self._frame_count / elapsed if elapsed > 0 else 0
        log.info(
            "Camera closed  [frames=%d, avg_fps=%.1f]",
            self._frame_count,
            avg_fps,
        )

    # ── frame acquisition ────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """Return the next frame as a BGR numpy array, or ``None`` on failure.

        All frames are returned in **BGR** colour order regardless of
        back-end, so downstream OpenCV code works without conversion.
        """
        frame: Optional[np.ndarray] = None

        if self._picam is not None:
            frame = self._picam.capture_array()
            if frame is not None:
                # picamera2 delivers RGB888 — convert to BGR for OpenCV
                import cv2
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        elif self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                log.warning("Frame grab failed (opencv)")
                return None

        if frame is not None:
            self._frame_count += 1
        return frame

    @property
    def is_opened(self) -> bool:
        """Return True if the camera is currently active."""
        if self._picam is not None:
            return True
        if self._cap is not None:
            return self._cap.isOpened()
        return False

    # ── private helpers ──────────────────────────

    def _open_picamera2(self) -> None:
        """Open the CSI camera via picamera2 + libcamera."""
        try:
            from picamera2 import Picamera2  # type: ignore[import-untyped]
        except ImportError:
            log.error(
                "\n"
                "╔══════════════════════════════════════════════════════╗\n"
                "║  picamera2 is NOT installed.                        ║\n"
                "║                                                     ║\n"
                "║  This project requires the Raspberry Pi Camera      ║\n"
                "║  Module 2 connected via the CSI ribbon cable.       ║\n"
                "║                                                     ║\n"
                "║  Install picamera2 on Raspberry Pi OS Bookworm:     ║\n"
                "║                                                     ║\n"
                "║    sudo apt update                                  ║\n"
                "║    sudo apt install -y python3-picamera2            ║\n"
                "║                                                     ║\n"
                "║  Do NOT use pip — the apt package includes the      ║\n"
                "║  required libcamera bindings.                       ║\n"
                "║                                                     ║\n"
                "║  After installing, verify with:                     ║\n"
                "║    python3 -c 'from picamera2 import Picamera2'     ║\n"
                "╚══════════════════════════════════════════════════════╝"
            )
            raise RuntimeError("picamera2 is not installed on this system.")

        self._picam = Picamera2()

        # Configure for still/video capture at the target resolution.
        # "RGB888" gives us 3-channel uint8 frames directly.
        cam_config = self._picam.create_preview_configuration(
            main={
                "size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
                "format": "RGB888",
            },
        )
        self._picam.configure(cam_config)
        self._picam.start()

        # Let auto-exposure / auto-white-balance converge before we
        # start feeding frames to the detector.
        time.sleep(2.0)
        log.info("Picamera2 ready  (CSI Camera Module 2, IMX219)")

    def _open_opencv(self) -> None:
        """Fallback: open a USB camera via OpenCV VideoCapture.

        This is provided for development on non-Pi machines only.
        It will NOT work with the CSI ribbon camera.
        """
        import cv2

        log.warning(
            "Using OpenCV backend — this is a USB-webcam fallback. "
            "For the CSI Camera Module 2, set CAMERA_BACKEND='picamera2'."
        )
        if sys.platform == "win32":
            self._cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Failed to open USB camera via OpenCV. "
                "Check that a webcam is connected."
            )

    # ── context manager ──────────────────────────

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ── standalone test ──────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Camera -- standalone test suite")
    print("=" * 60)

    cam = Camera()
    print(f"Backend configured: {cam._backend}")
    assert cam.is_opened is False, "Camera is_opened is False before open()"
    cam.close()
    assert cam.is_opened is False, "Camera is_opened remains False after close()"
    print("Standalone camera interface tests passed -- all clear!")

