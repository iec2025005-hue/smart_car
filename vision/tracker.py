"""
vision/tracker.py -- Lightweight single-object tracker (OpenCV).

Bridges the gap between expensive detection frames: initialise with a
bounding box from any detector, then call ``update()`` on subsequent
frames until the track is lost or a fresh detection arrives.

Supported OpenCV tracker backends (``config.TRACKER_TYPE``):
  - ``MOSSE``  -- ~10x faster than CSRT on ARM; good for walking speed.
  - ``KCF``    -- balanced accuracy / speed.
  - ``CSRT``   -- most accurate; heavier on CPU.

Standalone test::

    python -m vision.tracker
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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


# ── tracker factory ─────────────────────────────

def _make_tracker(tracker_type: str):
    """Instantiate an OpenCV tracker by name.

    Parameters
    ----------
    tracker_type : str
        One of ``"MOSSE"``, ``"KCF"``, ``"CSRT"`` (case-insensitive).

    Returns
    -------
    cv2.Tracker
        A freshly-created OpenCV tracker instance.

    Raises
    ------
    RuntimeError
        If OpenCV is not installed.
    ValueError
        If *tracker_type* is not recognised.
    """
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not installed.  "
            "Install with:  pip install opencv-python"
        )

    typ = tracker_type.upper()
    known_types = {"MOSSE", "KCF", "CSRT", "MIL", "NANO", "VIT", "DASIAMPN"}
    if typ not in known_types:
        raise ValueError(
            f"Unsupported tracker type: '{typ}'.  "
            "Choose from: MOSSE, KCF, CSRT, MIL."
        )

    # Search order for tracker creation function
    candidates = [
        (cv2, f"Tracker{typ}_create"),
        (cv2, f"Tracker{typ}"),
        (getattr(cv2, "legacy", None), f"Tracker{typ}_create"),
        (getattr(cv2, "legacy", None), f"Tracker{typ}"),
    ]

    for module, name in candidates:
        if module is None:
            continue
        obj = getattr(module, name, None)
        if obj is None:
            continue
        if hasattr(obj, "create") and callable(getattr(obj, "create")):
            return obj.create()
        if callable(obj):
            try:
                return obj()
            except Exception:
                pass

    # Fallback to MIL or Nano if requested contrib tracker is missing from opencv build
    log.warning(
        "Tracker '%s' not found in this OpenCV build. Falling back to TrackerMIL. "
        "(Install opencv-contrib-python-headless for full MOSSE/KCF/CSRT support)",
        typ,
    )
    for fallback_name in ["TrackerMIL", "TrackerNano"]:
        fn = getattr(cv2, f"{fallback_name}_create", None)
        if fn and callable(fn):
            return fn()
        cls = getattr(cv2, fallback_name, None)
        if cls and hasattr(cls, "create"):
            return cls.create()

    raise ValueError(
        f"Unsupported tracker type: '{typ}' and no fallback tracker available."
    )


def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Compute the centre point ``(cx, cy)`` of an ``(x, y, w, h)`` box."""
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


# ── main tracker class ──────────────────────────

class Tracker:
    """Single-object tracker wrapping an OpenCV tracking algorithm.

    Parameters
    ----------
    tracker_type : str or None
        One of ``"MOSSE"``, ``"KCF"``, ``"CSRT"``.
        Defaults to ``config.TRACKER_TYPE``.
    max_lost_frames : int or None
        Number of consecutive lost frames before the tracker auto-resets.
        Defaults to ``config.TRACKER_MAX_LOST_FRAMES``.
    """

    def __init__(
        self,
        tracker_type: Optional[str] = None,
        max_lost_frames: Optional[int] = None,
    ) -> None:
        self._type: str = (tracker_type or config.TRACKER_TYPE).upper()
        self._max_lost: int = (
            max_lost_frames
            if max_lost_frames is not None
            else config.TRACKER_MAX_LOST_FRAMES
        )

        self._tracker = None        # OpenCV tracker instance
        self._active: bool = False   # True while a track is live
        self._lost_frames: int = 0   # consecutive frames without a match
        self._bbox: Optional[Tuple[int, int, int, int]] = None
        self._center: Optional[Tuple[int, int]] = None

        log.info(
            "Tracker created  [type=%s  max_lost_frames=%d]",
            self._type, self._max_lost,
        )

    # ── public API ───────────────────────────────

    def init(
        self,
        frame: "np.ndarray",
        bbox: Tuple[int, int, int, int],
    ) -> bool:
        """Start tracking a target in *frame* at *bbox*.

        Parameters
        ----------
        frame : np.ndarray
            BGR image containing the target.
        bbox : tuple[int, int, int, int]
            Bounding box ``(x, y, w, h)`` of the target.

        Returns
        -------
        bool
            *True* if the tracker initialised successfully.
        """
        h_img, w_img = frame.shape[:2]
        x, y, w, h = map(int, bbox)
        x = max(0, min(x, w_img - 2))
        y = max(0, min(y, h_img - 2))
        w = max(2, min(w, w_img - x))
        h = max(2, min(h, h_img - y))
        safe_bbox = (float(x), float(y), float(w), float(h))

        try:
            self._tracker = _make_tracker(self._type)
        except (RuntimeError, ValueError) as exc:
            log.error("Cannot create tracker: %s", exc)
            self._active = False
            return False

        try:
            res = self._tracker.init(frame, safe_bbox)
            ok = True if res is None else bool(res)
        except Exception as exc:
            log.warning("Tracker init failed for %s: %s", safe_bbox, exc)
            ok = False

        self._active = ok
        self._lost_frames = 0

        if ok:
            self._bbox = tuple(int(v) for v in bbox)  # type: ignore[assignment]
            self._center = _bbox_center(self._bbox)
            log.info(
                "Tracker initialised  [bbox=%s  center=%s]",
                self._bbox, self._center,
            )
        else:
            self._bbox = None
            self._center = None
            log.warning("Tracker init failed for bbox=%s", bbox)

        return self._active

    def update(
        self,
        frame: "np.ndarray",
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int]]]:
        """Feed a new frame to the tracker.

        Parameters
        ----------
        frame : np.ndarray
            The next BGR frame.

        Returns
        -------
        (ok, bbox, center)
            *ok* is ``True`` if the target was found.
            *bbox* is ``(x, y, w, h)`` or ``None``.
            *center* is ``(cx, cy)`` or ``None``.
        """
        if not self._active or self._tracker is None:
            log.debug("update() called but tracker is not active")
            return False, None, None

        ok, box = self._tracker.update(frame)

        if ok:
            self._lost_frames = 0
            self._bbox = tuple(int(v) for v in box)  # type: ignore[assignment]
            self._center = _bbox_center(self._bbox)
            log.debug(
                "Track OK  [bbox=%s  center=%s]",
                self._bbox, self._center,
            )
            return True, self._bbox, self._center

        # Track lost this frame
        self._lost_frames += 1
        log.debug(
            "Track lost  [lost_frames=%d/%d]",
            self._lost_frames, self._max_lost,
        )

        if self._lost_frames >= self._max_lost:
            log.info(
                "Track lost for %d consecutive frames -- auto-resetting",
                self._lost_frames,
            )
            self.reset()

        return False, None, None

    def reset(self) -> None:
        """Clear the current track and release the tracker."""
        was_active = self._active
        self._tracker = None
        self._active = False
        self._lost_frames = 0
        self._bbox = None
        self._center = None

        if was_active:
            log.info("Tracker reset")
        else:
            log.debug("Tracker reset (was already inactive)")

    def is_tracking(self) -> bool:
        """Return *True* if a track is currently active."""
        return self._active

    # ── convenience ──────────────────────────────

    def to_detection(self, label: str = "tracked") -> Optional[Detection]:
        """Convert the current track state to a ``Detection`` object.

        Returns *None* if the tracker is not active.

        Parameters
        ----------
        label : str
            Label to assign to the Detection (default ``"tracked"``).
        """
        if not self._active or self._bbox is None or self._center is None:
            return None
        return Detection(
            label=label,
            confidence=1.0,
            bbox=self._bbox,
            center=self._center,
            extra={"source": "tracker", "tracker_type": self._type},
        )

    @property
    def bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Last known bounding box, or *None*."""
        return self._bbox

    @property
    def center(self) -> Optional[Tuple[int, int]]:
        """Last known centre point, or *None*."""
        return self._center

    @property
    def lost_frames(self) -> int:
        """Number of consecutive frames without a match."""
        return self._lost_frames


# ── backward compatibility alias ────────────────
# The existing codebase references ObjectTracker in some places.
ObjectTracker = Tracker


# ── standalone test ──────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Tracker -- standalone test suite")
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

    # ── 1. Config defaults ───────────────────────
    print("\n[1] Constructor reads config defaults")
    t1 = Tracker()
    _assert(t1._type == config.TRACKER_TYPE.upper(), f"type = {t1._type}")
    _assert(
        t1._max_lost == config.TRACKER_MAX_LOST_FRAMES,
        f"max_lost = {t1._max_lost}",
    )

    # ── 2. Custom overrides ──────────────────────
    print("\n[2] Constructor accepts custom overrides")
    t2 = Tracker(tracker_type="kcf", max_lost_frames=10)
    _assert(t2._type == "KCF", f"type = {t2._type}")
    _assert(t2._max_lost == 10, f"max_lost = {t2._max_lost}")

    # ── 3. Initial state ─────────────────────────
    print("\n[3] Initial state is inactive")
    t3 = Tracker()
    _assert(t3.is_tracking() is False, "is_tracking() == False before init")
    _assert(t3.bbox is None, "bbox is None before init")
    _assert(t3.center is None, "center is None before init")
    _assert(t3.lost_frames == 0, "lost_frames == 0 before init")

    # ── 4. update() before init ──────────────────
    print("\n[4] update() before init() returns (False, None, None)")
    if np is not None:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ok, bbox, center = t3.update(dummy_frame)
        _assert(ok is False, "ok == False")
        _assert(bbox is None, "bbox is None")
        _assert(center is None, "center is None")
    else:
        print("  [SKIP] numpy not available")

    # ── 5. reset() is always safe ────────────────
    print("\n[5] reset() is safe on an inactive tracker")
    t5 = Tracker()
    t5.reset()
    _assert(t5.is_tracking() is False, "still inactive after reset()")
    t5.reset()  # double reset
    _assert(True, "double reset() did not crash")

    # ── 6. to_detection() when inactive ──────────
    print("\n[6] to_detection() returns None when not tracking")
    t6 = Tracker()
    det = t6.to_detection()
    _assert(det is None, "to_detection() is None when inactive")

    # ── 7. ObjectTracker alias ───────────────────
    print("\n[7] ObjectTracker is an alias for Tracker")
    _assert(ObjectTracker is Tracker, "ObjectTracker is Tracker")

    # ── 8. _bbox_center helper ───────────────────
    print("\n[8] _bbox_center computes correct centre")
    c = _bbox_center((100, 200, 80, 60))
    _assert(c == (140, 230), f"centre of (100,200,80,60) = {c}")
    c2 = _bbox_center((0, 0, 640, 480))
    _assert(c2 == (320, 240), f"centre of (0,0,640,480) = {c2}")

    # ── 9. _make_tracker validates type ──────────
    print("\n[9] _make_tracker rejects unknown types")
    try:
        _make_tracker("INVALID")
        _assert(False, "should have raised ValueError")
    except ValueError as exc:
        _assert("Unsupported" in str(exc), f"ValueError: {exc}")
    except RuntimeError:
        # OpenCV not installed -- that's fine too
        _assert(True, "RuntimeError (no OpenCV) is acceptable")

    # ── 10-14. Full tracking cycle (requires OpenCV + numpy) ──
    if cv2 is not None and np is not None:
        print("\n[10] init() with a synthetic frame")
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (100, 100), (200, 200), (255, 255, 255), -1)

        # Try each supported tracker type
        for ttype in ("MOSSE", "KCF", "CSRT"):
            print(f"\n--- Testing {ttype} tracker ---")
            trk = Tracker(tracker_type=ttype)

            # init
            ok = trk.init(frame1, (100, 100, 100, 100))
            _assert(ok is True, f"{ttype}: init() succeeded")
            _assert(trk.is_tracking() is True, f"{ttype}: is_tracking() == True")
            _assert(trk.bbox == (100, 100, 100, 100), f"{ttype}: bbox after init")
            _assert(trk.center == (150, 150), f"{ttype}: center after init")

            # update with a slightly shifted frame
            frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame2, (110, 105), (210, 205), (255, 255, 255), -1)

            ok, bbox, center = trk.update(frame2)
            _assert(ok is True, f"{ttype}: update() found target")
            _assert(bbox is not None, f"{ttype}: bbox is not None")
            _assert(center is not None, f"{ttype}: center is not None")
            if bbox is not None:
                _assert(len(bbox) == 4, f"{ttype}: bbox has 4 elements")
                _assert(
                    all(isinstance(v, int) for v in bbox),
                    f"{ttype}: bbox values are ints",
                )
            if center is not None:
                _assert(len(center) == 2, f"{ttype}: center has 2 elements")

            # to_detection()
            det = trk.to_detection(label="test_obj")
            _assert(det is not None, f"{ttype}: to_detection() is not None")
            if det is not None:
                _assert(det.label == "test_obj", f"{ttype}: detection label")
                _assert(det.bbox == bbox, f"{ttype}: detection bbox matches")
                _assert(det.center == center, f"{ttype}: detection center matches")
                _assert(
                    det.extra.get("source") == "tracker",
                    f"{ttype}: detection extra['source']",
                )

            # reset
            trk.reset()
            _assert(trk.is_tracking() is False, f"{ttype}: inactive after reset")
            _assert(trk.bbox is None, f"{ttype}: bbox is None after reset")
            _assert(trk.center is None, f"{ttype}: center is None after reset")

        # ── Auto-reset after max lost frames ─────
        print("\n[14] Auto-reset after max_lost_frames")
        trk_ar = Tracker(tracker_type="MOSSE", max_lost_frames=3)
        trk_ar.init(frame1, (100, 100, 100, 100))

        # Simulate consecutive update failures
        class _FailingTracker:
            def update(self, _frame):
                return False, (0, 0, 0, 0)

        trk_ar._tracker = _FailingTracker()

        for _ in range(3):
            ok, _, _ = trk_ar.update(frame1)

        _assert(
            trk_ar.is_tracking() is False,
            "auto-reset after max_lost_frames exceeded",
        )
        _assert(trk_ar.lost_frames == 0, "lost_frames reset to 0")

    else:
        print("\n  [SKIP] OpenCV / NumPy not installed -- skipping tracking tests")
        print("         Install with:  pip install opencv-python numpy")

    # ── summary ──────────────────────────────────
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  -- all clear!")
    print("=" * 60)
