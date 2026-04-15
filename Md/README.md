# 🚦 Driver Safety AI
### SDG 3 – Good Health & Well-being
### SDG 9: Industry, Innovation and Infrastructure
### SDG 11: Sustainable Cities, and Communities


Detects dangerous road behaviors in real-time using Computer Vision + Deep Learning:
- 📱 **Phone usage while driving**
- 🪢 **No seatbelt** (car occupants)
- ⛑️ **Riding without a helmet** (motorcycle / bicycle riders)
- ⚠️ **Dangerous road actions** (e.g. tailgating heuristics)

---

## 📁 Project Structure

```
driver-safety-ai/
├── detector.py                 ← Main entry point
├── requirements.txt
├── config/
│   └── settings.py             ← All thresholds, paths, email config
├── models/
│   ├── yolo_detector.py        ← YOLOv8 wrapper (persons, vehicles, phone)
│   ├── phone_detector.py       ← Phone classifier + optional training (`python models/phone_detector.py`)
│   ├── seatbelt_detector.py    ← Seatbelt YOLO + heuristic fallback
│   └── helmet_detector.py      ← Helmet YOLO + heuristic fallback
├── utils/
│   ├── alert_manager.py        ← Alert dispatch (console, email, snapshot)
│   └── logger.py               ← File + console logger
├── output/
│   └── alerts/                 ← Auto-saved alert snapshots
└── logs/                       ← Session log files
```

---

## ⚡ Quick Start

```bash
# 1. Clone / download this project
git clone https://github.com/yourname/driver-safety-ai.git
cd driver-safety-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run with webcam (default)
python detector.py

# 4. Run with a video file
python detector.py --source path/to/video.mp4

# 5. Run with IP camera
python detector.py --source rtsp://192.168.1.100:554/stream
```

**Controls during detection:**
- `Q` → Quit
- `S` → Save snapshot

---

## 🤖 Model Downloads

### 1. YOLOv8 (Primary – auto-downloads)

| Model | Size | Speed | Download |
|-------|------|-------|----------|
| YOLOv8n | 6 MB | Fastest | Auto |
| YOLOv8s | 22 MB | Fast | Auto |
| **YOLOv8m** | 52 MB | **Recommended** ✅ | Auto |
| YOLOv8l | 87 MB | Accurate | Manual |

YOLOv8 weights are **auto-downloaded** by `ultralytics` on first run.
Manual download: https://github.com/ultralytics/assets/releases/tag/v8.1.0

```python
# Change model in config/settings.py:
YOLO_MODEL_PATH = "yolov8m.pt"   # or yolov8n.pt, yolov8s.pt, yolov8l.pt
```

---

### 2. Phone Usage Detection

**Option A – Use YOLO class 67 (cell phone) — Zero setup needed ✅**
Already integrated in `yolo_detector.py`. No extra model required.

**Option B – Custom MobileNetV2 Classifier**
```bash
# Prepare dataset
mkdir -p dataset/phone/phone dataset/phone/no_phone
# Add images: ~500+ per class recommended

# Train
python models/phone_detector.py
# Outputs: models/phone_classifier.h5
```

---

### 3. Helmet Detection

**Option A – Fine-tuned YOLOv8 (Recommended) ✅**
```bash
# Download pre-trained helmet detection model:
# https://universe.roboflow.com/joseph-nelson/hardhat-universe
# Select: YOLOv8 → Download → Place at models/helmet_detector.pt
```

**Option B – Heuristic fallback (no download needed)**
Automatically used if no `helmet_detector.pt` is found (skin-cue heuristic; prefer a trained model for production).

---

### 4. Seatbelt Detection

Place a **YOLOv8** model trained on seatbelt / buckled vs unbuckled data at `models/seatbelt_detector.pt` (e.g. search [Roboflow Universe](https://universe.roboflow.com/) for “seatbelt detection”). If the file is missing, a **heuristic** on the torso crop is used (conservative, weak signal).

Tune `SEATBELT_THRESHOLD`, `SEATBELT_NEAR_CAR_PX`, and vehicle association settings in `config/settings.py`.

---

## 🔔 Alert System

| Channel | Trigger | Config |
|---------|---------|--------|
| Console | All alerts | Always on |
| Terminal beep | All alerts | Always on |
| Screenshot | All alerts | `output/alerts/` |
| Email (Gmail) | HIGH severity | Set in `config/settings.py` |

### Enable Email Alerts

```python
# config/settings.py
EMAIL_ENABLED = True
EMAIL_CONFIG = {
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 465,
    "sender":   "your_gmail@gmail.com",
    "password": "xxxx xxxx xxxx xxxx",   # Gmail App Password
    "receiver": "authority@example.com",
}
```
Gmail App Password: https://myaccount.google.com/apppasswords

---

## 🧠 Detection Logic

```
Camera Frame
     │
     ▼
YOLOv8 Detection
     │
     ├── Person near car?
     │        ├── Torso crop → SeatbeltDetector → ALERT: No seatbelt (if unbuckled)
     │        └── Person crop → PhoneDetector (+ YOLO cell phone) → ALERT: Phone while driving
     │
     ├── Person near motorcycle/bicycle? → Head crop → HelmetDetector → ALERT: No helmet
     │
     ├── Multiple vehicles close? → ALERT: Tailgating
     │
     └── Draw boxes + HUD overlay → Display frame
```

---

## ⚙️ Configuration

Edit `config/settings.py`:

```python
PHONE_THRESHOLD = 0.55       # Fused phone score (classifier + YOLO cell phone)
TAILGATE_RATIO  = 0.38       # Tailgating sensitivity (gap / avg vehicle width)
DETECTION_CONFIRM_FRAMES = 2 # Consecutive hits before alerting

ALERT_COOLDOWN = {
    "PHONE_WHILE_DRIVING": 8,
    "NO_SEATBELT": 10,
    "NO_HELMET": 10,
    "TAILGATING": 6,
}
```

---

## 🖥️ Hardware Requirements

| Setup | GPU | CPU | Camera |
|-------|-----|-----|--------|
| Minimum | None | i5 / Ryzen 5 | 720p webcam |
| Recommended | NVIDIA GTX 1060+ | i7 / Ryzen 7 | 1080p USB/IP cam |
| Production | NVIDIA RTX 3060+ | Any modern | 4K IP cameras |

For Raspberry Pi / Jetson Nano: use `yolov8n.pt` and lower `INFERENCE_IMGSZ` in `config/settings.py` if needed.

---

## 📊 Accuracy Benchmarks

| Detection | Model | mAP / Accuracy |
|-----------|-------|---------------|
| Person / Vehicle | YOLOv8m COCO | ~72% mAP50 |
| Phone usage | YOLO class 67 | ~60–70% |
| Helmet | Roboflow fine-tuned | ~85–90% |

---

## 🔮 Future Improvements

- [ ] Speed estimation (optical flow)
- [ ] Wrong-way driving (direction vector analysis)
- [ ] Web dashboard with live RTSP stream
- [ ] Twilio SMS alerts
- [ ] ANPR (Automatic Number Plate Recognition) integration
- [ ] Multi-camera support

---

## 📜 License
MIT — Free for educational and research use.

## 🙏 Credits
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [Roboflow Universe](https://universe.roboflow.com/)
