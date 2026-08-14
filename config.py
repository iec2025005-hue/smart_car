"""
config.py — Central configuration for the Smart Car robot.

Target Hardware
───────────────
  • Raspberry Pi 4 (4 GB RAM), Cortex-A72 quad-core, no GPU accel.
  • Raspberry Pi Camera Module v2 (Sony IMX219, 8 MP) — CSI ribbon.
  • STM32 Blue Pill (STM32F103C8T6) — motor controller, connected via UART.
  • Stepper motors for steering, encoder DC motors for drive.

Software Stack
──────────────
  • Raspberry Pi OS Bookworm (Debian 12)
  • picamera2 + libcamera for camera capture
  • OpenCV for image processing
  • Ultralytics YOLOv8-nano for object detection

Performance Strategy
────────────────────
  • 640×480 resolution keeps per-frame cost under ~200 ms.
  • YOLOv8-nano (~3.2 M params, ~6.5 GFLOPs) → ~5-10 FPS on Pi 4 CPU.
  • MOSSE tracker fills in between expensive detection frames.
  • Vision mode is selectable so only the active pipeline runs.
"""

# ══════════════════════════════════════════════
#  VISION MODE  (select the active pipeline)
# ══════════════════════════════════════════════
# Supported modes:
#   "yolo"   — YOLOv8-nano object detection (general purpose)
#   "color"  — HSV colour-range detection (fast, lightweight)
#   "aruco"  — ArUco fiducial marker detection & pose estimation
#   "lane"   — Lane-line detection (Canny + Hough transform)
VISION_MODE = "yolo"

# ──────────────────────────────────────────────
#  Camera
# ──────────────────────────────────────────────
# Pi Camera Module v2 over CSI ribbon cable.
# picamera2 auto-detects the CSI camera; CAMERA_INDEX is only used
# by the OpenCV fallback for USB webcams during off-Pi development.
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_BACKEND = "picamera2"         # "picamera2" (CSI for Pi) | "opencv" (USB/webcam fallback)

# ──────────────────────────────────────────────
#  YOLO Detection
# ──────────────────────────────────────────────
# YOLOv8-nano: smallest variant, ~37 mAP on COCO.
# Larger variants (s/m/l) won't run at usable FPS on Pi 4.
MODEL_PATH = "models/yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.50
# No class filtering; detect all COCO classes
YOLO_TARGET_CLASSES = None

# ──────────────────────────────────────────────
#  Colour Detection
# ──────────────────────────────────────────────
# HSV lower/upper bounds for the target colour.
# Default: red (two ranges to handle hue wrap-around).
COLOR_HSV_LOWER_1 = (0, 120, 70)
COLOR_HSV_UPPER_1 = (10, 255, 255)
COLOR_HSV_LOWER_2 = (170, 120, 70)
COLOR_HSV_UPPER_2 = (180, 255, 255)
COLOR_MIN_AREA = 500                # ignore contours smaller than this (px²)

# ──────────────────────────────────────────────
#  ArUco Marker Detection
# ──────────────────────────────────────────────
ARUCO_DICT_TYPE = "DICT_4X4_50"     # OpenCV ArUco dictionary name
# Camera calibration matrix & distortion coefficients.
# Replace with values from your own calibration for accurate pose.
ARUCO_CAMERA_MATRIX = None          # 3×3 numpy array or None
ARUCO_DIST_COEFFS = None            # 1×5 numpy array or None
ARUCO_MARKER_SIZE_CM = 5.0          # physical marker side length (cm)

# ──────────────────────────────────────────────
#  Lane Detection
# ──────────────────────────────────────────────
LANE_CANNY_LOW = 50
LANE_CANNY_HIGH = 150
LANE_HOUGH_THRESHOLD = 50
LANE_HOUGH_MIN_LINE_LEN = 50
LANE_HOUGH_MAX_LINE_GAP = 150
# Region-of-interest: fraction of the frame height to keep (bottom portion).
LANE_ROI_TOP_FRACTION = 0.6         # keep bottom 40 % of the frame

# ──────────────────────────────────────────────
#  Object Tracking
# ──────────────────────────────────────────────
# MOSSE is ~10× faster than CSRT on ARM — good enough for a car
# moving at walking speed.  Upgrade to KCF/CSRT on Pi 5 if needed.
TRACKER_TYPE = "MOSSE"              # MOSSE | KCF | CSRT
TRACKER_MAX_LOST_FRAMES = 30
TRACKER_IOU_THRESHOLD = 0.3

# ──────────────────────────────────────────────
#  Navigation / PID / Swerve
# ──────────────────────────────────────────────
# Swerve Kinematics Dimensions (Update with real measurements!)
SWERVE_WHEELBASE_LENGTH = 20.0      # distance between front and rear axles (cm)
SWERVE_TRACK_WIDTH = 15.0           # distance between left and right wheels (cm)
MAX_DRIVE_SPEED = 255               # Maximum PWM value for drive motors

BASE_SPEED = 150                    # Default translational speed
TURN_SPEED = 1.0                    # Default rotational speed (rad/s) for avoid mode
STOP_DISTANCE_CM = 20.0             # ultrasonic emergency-stop threshold
FRAME_CENTER_TOLERANCE = 50         # pixels — dead-zone around centre
NAVIGATION_MODE = "idle"            # "idle" | "follow" | "avoid" | "patrol"

PID_KP = 0.4
PID_KI = 0.0
PID_KD = 0.1
PID_OUTPUT_LIMITS = (-255, 255)     # clamp PID output to motor PWM range

# ──────────────────────────────────────────────
#  Serial / UART  (STM32/ESP32)
# ──────────────────────────────────────────────
SERIAL_PORT = "COM7"                # Windows COM port for ESP32
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0                # seconds
COMMAND_TERMINATOR = "\n"

# ──────────────────────────────────────────────
#  Display / Overlay
# ──────────────────────────────────────────────
# Enable on-screen overlay during development (HDMI / VNC).
# Disable in headless production to save CPU.
DISPLAY_PREVIEW = False
OVERLAY_SHOW_FPS = True
OVERLAY_SHOW_MODE = True

# ──────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────
LOG_LEVEL = "INFO"                  # DEBUG | INFO | WARNING | ERROR
LOG_FILE = "logs/smart_car.log"
LOG_MAX_BYTES = 5 * 1024 * 1024     # 5 MB per log file
LOG_BACKUP_COUNT = 3

# ──────────────────────────────────────────────
#  Output
# ──────────────────────────────────────────────
SAVE_DETECTIONS = False
DETECTIONS_DIR = "output/detections"

# ──────────────────────────────────────────────
#  Web Dashboard
# ──────────────────────────────────────────────
WEB_ENABLE = True                   # Start web server for live dashboard & control
WEB_HOST = "0.0.0.0"                # Accessible on local network (e.g. phone)
WEB_PORT = 5000                     # HTTP port

