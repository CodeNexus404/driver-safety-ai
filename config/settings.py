"""
Global Settings & Configuration
"""


class Settings:
    # ── Video ──────────────────────────────────────────────────────────────
    FRAME_WIDTH = 960
    FRAME_HEIGHT = 540
    # YOLO input size — 512 improves small-object recall vs 416 (use 416 if CPU-bound)
    INFERENCE_IMGSZ = 512

    # ── Model paths ────────────────────────────────────────────────────────
    # yolov8s.pt = better accuracy than n; use yolov8n.pt for maximum FPS
    YOLO_MODEL_PATH = "yolov8s.pt"
    PHONE_MODEL_PATH = "models/phone_classifier.h5"
    SEATBELT_MODEL_PATH = "models/seatbelt_detector.pt"
    HELMET_MODEL_PATH = "models/helmet_detector.pt"

    # ── Thresholds ─────────────────────────────────────────────────────────
    YOLO_CONF = 0.42                # Base detection confidence (balance precision/recall)
    YOLO_IOU = 0.50                 # NMS IoU
    PHONE_THRESHOLD = 0.55          # Fused phone score (classifier or YOLO cell phone)
    PHONE_YOLO_MIN_CONF = 0.48      # Min YOLO "cell phone" conf when fused with person IoU
    PHONE_YOLO_MIN_IOU = 0.10       # Min IoU(person, cell_phone) to treat as phone cue
    SEATBELT_THRESHOLD = 0.35       # YOLO conf for seatbelt model classes
    HELMET_THRESHOLD = 0.35         # YOLO conf for helmet model classes
    # Person must overlap a car (IoU) or sit inside padded car box — cuts pedestrian FPs
    PERSON_VEHICLE_MIN_IOU = 0.06
    PERSON_VEHICLE_CENTER_PAD = 0.12  # expand car box fraction for center-in test
    # Fallback if IoU is weak: max distance person center → car center (pixels)
    SEATBELT_NEAR_CAR_PX = 200
    PHONE_NEAR_CAR_PX = 160
    # Person ↔ motorcycle/bicycle association (helmet rule)
    HELMET_NEAR_BIKE_PX = 220
    # Consecutive inference hits before alerting (reduces single-frame noise)
    DETECTION_CONFIRM_FRAMES = 2
    CONFIRM_GRID_PX = 72            # quantize person position for streak grouping
    # Tailgating: require vertical overlap + similar scale (reduces adjacent-lane FPs)
    TAILGATE_RATIO = 0.38           # horizontal gap / avg width
    TAILGATE_MIN_VERTICAL_OVERLAP = 0.32  # fraction of min box height
    TAILGATE_MAX_HEIGHT_RATIO_DIFF = 0.42  # abs(h1-h2)/max(h1,h2)

    # Alert overlay stays visible this many seconds after a fired alert (reduces blink)
    ALERT_OVERLAY_TTL_SEC = 3.0

    # ── Alert cooldowns (seconds) ─────────────────────────────────────────
    ALERT_COOLDOWN = {
        "PHONE_WHILE_DRIVING": 8,
        "NO_SEATBELT": 10,
        "NO_HELMET": 10,
        "TAILGATING": 6,
    }

    # ── YOLO class colors (BGR) ────────────────────────────────────────────
    CLASS_COLORS = {
        "person":     (0, 200, 255),
        "car":        (50, 200, 50),
        "motorcycle": (255, 140, 0),
        "truck":      (200, 50, 200),
        "bus":        (50, 50, 255),
        "bicycle":    (0, 255, 200),
        "cell phone": (0, 0, 255),
    }

    # ── Email alerts ───────────────────────────────────────────────────────
    EMAIL_ENABLED = False           # Set True and fill below to enable
    EMAIL_CONFIG = {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 465,
        "sender":   "your_email@gmail.com",
        "password": "your_app_password",       # Use Gmail App Password
        "receiver": "traffic_authority@example.com",
    }
