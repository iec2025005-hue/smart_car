"""
web/app.py -- Web Dashboard server for Smart Car control and live stream.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

import cv2
from flask import Flask, Response, jsonify, render_template, request

import config
from utils.logger import get_logger

log = get_logger(__name__)

# Flask app initialization
template_dir = os.path.join(os.path.dirname(__file__), "templates")
app = Flask(__name__, template_folder=template_dir)

# Global thread-safe state shared with main.py
_lock = threading.Lock()
_latest_jpeg: Optional[bytes] = None

_status: Dict[str, Any] = {
    "vision_mode": config.VISION_MODE,
    "nav_mode": config.NAVIGATION_MODE,
    "target": "all",
    "nav_command": "STOP:0",
    "fps": 0.0,
    "serial_status": "connected",
}

# Callback references set by main.py
_set_target_fn = None
_set_nav_mode_fn = None
_set_vision_mode_fn = None


def update_frame(frame) -> None:
    """Encode OpenCV BGR frame to JPEG and update global buffer."""
    global _latest_jpeg
    if frame is None:
        return
    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ret:
        with _lock:
            _latest_jpeg = jpeg.tobytes()


def update_status(key: str, value: Any) -> None:
    """Update telemetry status dictionary."""
    with _lock:
        _status[key] = value


def get_status() -> Dict[str, Any]:
    with _lock:
        return dict(_status)


def register_callbacks(set_target=None, set_nav_mode=None, set_vision_mode=None):
    global _set_target_fn, _set_nav_mode_fn, _set_vision_mode_fn
    _set_target_fn = set_target
    _set_nav_mode_fn = set_nav_mode
    _set_vision_mode_fn = set_vision_mode


# ── Flask Routes ───────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


def _generate_video_stream():
    """MJPEG stream generator."""
    while True:
        with _lock:
            jpeg = _latest_jpeg
        if jpeg is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        time.sleep(0.04)  # ~25 FPS max stream cap


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify(get_status())


@app.route("/api/set_target", methods=["POST"])
def api_set_target():
    data = request.get_json(silent=True) or {}
    target = data.get("target", "all")
    update_status("target", target)

    if _set_target_fn:
        _set_target_fn(target)

    log.info("Web Dashboard set target to: '%s'", target)
    return jsonify({"success": True, "target": target})


@app.route("/api/set_nav_mode", methods=["POST"])
def api_set_nav_mode():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "idle")
    update_status("nav_mode", mode)

    if _set_nav_mode_fn:
        _set_nav_mode_fn(mode)

    log.info("Web Dashboard set nav_mode to: '%s'", mode)
    return jsonify({"success": True, "nav_mode": mode})


@app.route("/api/set_vision_mode", methods=["POST"])
def api_set_vision_mode():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "yolo")
    update_status("vision_mode", mode)

    if _set_vision_mode_fn:
        _set_vision_mode_fn(mode)

    log.info("Web Dashboard set vision_mode to: '%s'", mode)
    return jsonify({"success": True, "vision_mode": mode})


@app.route("/api/set_confidence", methods=["POST"])
def api_set_confidence():
    data = request.get_json(silent=True) or {}
    conf = float(data.get("confidence", 0.25))
    config.CONFIDENCE_THRESHOLD = max(0.05, min(0.95, conf))
    update_status("confidence", config.CONFIDENCE_THRESHOLD)
    log.info("Web Dashboard set confidence threshold to: %.2f", config.CONFIDENCE_THRESHOLD)
    return jsonify({"success": True, "confidence": config.CONFIDENCE_THRESHOLD})


@app.route("/api/set_speed", methods=["POST"])
def api_set_speed():
    data = request.get_json(silent=True) or {}
    speed = int(data.get("speed", 150))
    config.BASE_SPEED = max(30, min(255, speed))
    update_status("base_speed", config.BASE_SPEED)
    log.info("Web Dashboard set base speed to: %d", config.BASE_SPEED)
    return jsonify({"success": True, "base_speed": config.BASE_SPEED})


@app.route("/api/emergency_stop", methods=["POST"])
def api_emergency_stop():
    update_status("nav_mode", "idle")
    update_status("nav_command", "SWERVE:0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0")
    if _set_nav_mode_fn:
        _set_nav_mode_fn("idle")
    log.warning("🚨 EMERGENCY STOP ACTIVATED FROM DASHBOARD 🚨")
    return jsonify({"success": True, "status": "STOPPED"})


def start_server(host=config.WEB_HOST, port=config.WEB_PORT):
    """Start Flask server in a daemon thread."""
    # Quiet Flask logging
    import logging
    log_flask = logging.getLogger("werkzeug")
    log_flask.setLevel(logging.ERROR)

    t = threading.Thread(
        target=app.run,
        kwargs={"host": host, "port": port, "threaded": True, "use_reloader": False},
        daemon=True,
    )
    t.start()
    log.info("Web Dashboard running at http://%s:%d", host if host != "0.0.0.0" else "localhost", port)
    return t
