# Smart Car — Modular Image-Processing Robot

Autonomous image-processing car built on **Raspberry Pi 4** with the
**Pi Camera Module v2** (CSI) and an **STM32 Blue Pill** motor controller.

## Hardware

| Component | Role |
|---|---|
| Raspberry Pi 4 (4 GB) | Main processor — vision & navigation |
| Pi Camera Module v2 (CSI) | Image capture (Sony IMX219, 8 MP) |
| STM32 Blue Pill (STM32F103) | Motor controller (receives UART commands) |
| Stepper motors | Steering |
| Encoder DC motors | Drive / movement |

## Software Stack

- **OS**: Raspberry Pi OS Bookworm (Debian 12)
- **Camera**: picamera2 + libcamera
- **Vision**: OpenCV, Ultralytics YOLOv8-nano
- **Comms**: pyserial (UART to STM32)

## Vision Modes

The active vision pipeline is set in [`config.py`](config.py) → `VISION_MODE`:

| Mode | Module | Description |
|---|---|---|
| `yolo` | `vision/yolo_detector.py` | YOLOv8-nano general object detection |
| `color` | `vision/color_detector.py` | HSV colour-range detection |
| `aruco` | `vision/aruco_detector.py` | ArUco fiducial marker detection & pose |
| `lane` | `vision/lane_detector.py` | Canny + Hough lane-line detection |

## Project Structure

```
smart_car/
├── config.py                  # Central configuration (vision mode, hardware, PID)
├── main.py                    # Entry point — main loop
├── requirements.txt
├── README.md
│
├── vision/
│   ├── camera.py              # Picamera2 / OpenCV capture
│   ├── detector.py            # BaseDetector ABC + factory
│   ├── yolo_detector.py       # YOLOv8-nano
│   ├── color_detector.py      # HSV colour detection
│   ├── aruco_detector.py      # ArUco markers
│   ├── lane_detector.py       # Lane lines
│   └── tracker.py             # OpenCV object tracker (MOSSE/KCF/CSRT)
│
├── navigation/
│   ├── navigator.py           # Detection → motor commands
│   └── pid.py                 # PID controller
│
├── communication/
│   └── serial_comm.py         # UART to STM32 Blue Pill
│
├── display/
│   └── overlay.py             # HUD overlay (bboxes, FPS, mode)
│
├── utils/
│   ├── logger.py              # Rotating-file logger
│   └── fps.py                 # FPS counter
│
└── models/                    # Place YOLOv8 .pt weights here
```

## Quick Start

### 1. Install dependencies

```bash
# On Raspberry Pi OS Bookworm
sudo apt update && sudo apt install -y python3-picamera2

# Python packages
pip install -r requirements.txt
```

### 2. Download YOLOv8-nano weights

```bash
# Ultralytics auto-downloads on first run, or manually:
mkdir -p models
wget -O models/yolov8n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
```

### 3. Run

```bash
# Default mode (reads VISION_MODE from config.py)
python main.py

# Override vision mode
python main.py --mode color

# Enable live preview (requires display)
python main.py --preview

# Dry-run without serial (no STM32 needed)
python main.py --no-serial --preview
```

### 4. Test individual modules

Every module can be tested independently:

```bash
python -m vision.yolo_detector
python -m vision.color_detector
python -m vision.aruco_detector
python -m vision.lane_detector
python -m vision.tracker
python -m navigation.navigator
python -m navigation.pid
python -m communication.serial_comm
python -m display.overlay
```

## Configuration

All tuneable parameters are in [`config.py`](config.py). Key settings:

| Setting | Default | Description |
|---|---|---|
| `VISION_MODE` | `"yolo"` | Active vision pipeline |
| `CAMERA_BACKEND` | `"picamera2"` | Camera interface |
| `MODEL_PATH` | `"models/yolov8n.pt"` | YOLO weights file |
| `SERIAL_PORT` | `"/dev/ttyUSB0"` | STM32 UART port |
| `NAVIGATION_MODE` | `"idle"` | Motor behaviour |
| `PID_KP / KI / KD` | `0.4 / 0.0 / 0.1` | Steering PID gains |

## UART Protocol (Pi ↔ STM32)

Commands are ASCII, newline-terminated:

| Command | Description |
|---|---|
| `FWD:<speed>` | Drive forward (speed 0–255) |
| `LEFT:<speed>` | Steer left |
| `RIGHT:<speed>` | Steer right |
| `STOP:0` | All stop |

## License

Private project — not for redistribution.
