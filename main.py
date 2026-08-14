"""
main.py — Entry point for the Smart Car autonomous robot.

Pipeline
────────
  1. Capture frame  (vision/camera.py)
  2. Detect          (vision/<mode>_detector.py via detector.py factory)
  3. Track           (vision/tracker.py — optional, between detections)
  4. Navigate        (navigation/navigator.py → PID → motor commands)
  5. Communicate     (communication/serial_comm.py → UART → STM32)
  6. Display         (display/overlay.py — optional HUD)

Usage::

    python main.py                  # run with settings from config.py
    python main.py --mode color     # override vision mode
    python main.py --preview        # enable live preview window
"""

from __future__ import annotations

import argparse
import signal
import sys

import cv2

import config
from communication.serial_comm import SerialComm
from display.overlay import Overlay
from navigation.navigator import Navigator
from utils.fps import FPSCounter
from utils.logger import get_logger
from vision.camera import Camera
from vision.detector import create_detector
from vision.tracker import Tracker

log = get_logger(__name__)

# ── graceful shutdown ────────────────────────
_running = True


def _shutdown(sig, frame):
    global _running
    log.info("Received signal %s — shutting down …", sig)
    _running = False


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ── CLI ──────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Car — main loop")
    parser.add_argument(
        "--mode",
        choices=["yolo", "color", "aruco", "lane"],
        default=None,
        help="Override config.VISION_MODE",
    )
    parser.add_argument(
        "--preview",
        action="store_const",
        const=True,
        default=None,
        help="Enable live camera preview window",
    )
    parser.add_argument(
        "--no-serial",
        action="store_true",
        default=False,
        help="Run without opening the serial port (dry-run)",
    )
    return parser.parse_args()


# ── main loop ────────────────────────────────

def main() -> None:
    args = _parse_args()
    vision_mode = args.mode or config.VISION_MODE
    show_preview = args.preview if args.preview is not None else config.DISPLAY_PREVIEW

    log.info("═══════════════════════════════════════════")
    log.info("  Smart Car starting")
    log.info("  Vision mode : %s", vision_mode)
    log.info("  Preview     : %s", show_preview)
    log.info("  Serial      : %s", "disabled" if args.no_serial else config.SERIAL_PORT)
    log.info("═══════════════════════════════════════════")

    # ── initialise components ────────────────
    camera = Camera()
    detector = create_detector(vision_mode)
    tracker = Tracker()
    overlay = Overlay()
    fps = FPSCounter()
    serial = SerialComm()
    navigator = Navigator(send_fn=serial.send if not args.no_serial else None)

    # ── web dashboard integration ──────────────
    if config.WEB_ENABLE:
        try:
            from web.app import start_server, update_frame, update_status, register_callbacks

            def _on_web_target(target):
                if hasattr(detector, "set_target"):
                    detector.set_target(target)

            def _on_web_nav_mode(mode):
                navigator.set_mode(mode)

            register_callbacks(set_target=_on_web_target, set_nav_mode=_on_web_nav_mode)
            start_server()
        except Exception as exc:
            log.warning("Failed to start Web Dashboard: %s", exc)

    try:
        camera.open()
        if not args.no_serial:
            try:
                serial.open()
            except Exception as exc:
                log.warning("Could not open serial port %s: %s — proceeding in simulation mode.", config.SERIAL_PORT, exc)

        global _running
        while _running:
            frame = camera.read()
            if frame is None:
                continue

            # Detect
            detections = detector.detect(frame)

            # Track
            if detections:
                best_det = max(detections, key=lambda d: d.confidence)
                tracker.init(frame, best_det.bbox)
            else:
                ok, _, _ = tracker.update(frame)
                if ok:
                    det = tracker.to_detection()
                    if det:
                        detections = [det]

            # Navigate
            nav_cmd = navigator.update(detections)

            # FPS
            fps.tick()

            # Update Web Dashboard
            if config.WEB_ENABLE:
                web_frame = frame.copy()
                overlay.draw(
                    web_frame,
                    detections,
                    fps=fps.get(),
                    vision_mode=vision_mode,
                    nav_command=nav_cmd,
                )
                update_frame(web_frame)
                update_status("fps", fps.get())
                update_status("nav_command", nav_cmd)


            # Display
            if show_preview:
                overlay.draw(
                    frame,
                    detections,
                    fps=fps.get(),
                    vision_mode=vision_mode,
                    nav_command=nav_cmd,
                )
                cv2.imshow("Smart Car", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("User pressed 'q' — exiting.")
                    break

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — exiting.")
    finally:
        camera.close()
        detector.release()
        serial.close()
        if show_preview:
            cv2.destroyAllWindows()
        log.info("Smart Car stopped.")


if __name__ == "__main__":
    main()
