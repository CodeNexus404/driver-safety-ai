# Driver Safety AI — Project Explanation

**Purpose:** This document summarizes what the project does, which technologies it uses, and how the pieces fit together. It is written for a professor or technical reviewer.

---

## 1. What this project is about

This is an **AI-based road safety monitoring system** framed around **UN Sustainable Development Goal 3 (Good Health and Well-being)**. It processes **live video** (webcam, file, or IP/RTSP camera) and tries to flag **unsafe behaviors** that matter for traffic safety:

- **Phone use** in contexts associated with driving (person near a vehicle, fused with phone cues).
- **No seatbelt** for occupants associated with cars (when a dedicated seatbelt model is available).
- **No helmet** for people associated with bicycles or motorcycles (when a helmet model is available).
- **Dangerous spacing** between vehicles, interpreted as **tailgating**-style risk using geometric rules on detected boxes.

The system is **real-time oriented**: it shows annotated video, counts alerts, and can **log**, **save snapshots**, and optionally **email** high-severity alerts.

---

## 2. High-level idea (how it works)

1. **Object detection** finds people, vehicles (car, bus, truck), two-wheelers (motorcycle, bicycle), and optionally **cell phones** (COCO class 67) in each processed frame.
2. **Spatial reasoning** links a person to a vehicle or bike (overlap, distance, and padding rules in configuration) so alerts are not fired for arbitrary pedestrians.
3. **Specialized detectors** refine specific risks:
   - **Phone:** can combine YOLO “cell phone” detections with a **Keras/TensorFlow** classifier on cropped regions, depending on what models you provide.
   - **Seatbelt / helmet:** typically **custom YOLO**-style models (`.pt`) trained on seatbelt and helmet datasets, loaded when those files exist.
4. **Temporal confirmation** requires behavior to appear across **consecutive inference hits** (with position quantization) to reduce one-frame false alarms.
5. **Alerts** go through a central **alert manager** with **cooldowns** so the same event does not spam notifications.

The main program also uses **multi-threading**: one thread runs inference so the display loop can stay smoother.

---

## 3. Technologies used (and why they matter)

| Technology | Role in this project |
|------------|----------------------|
| **Python 3** | Main language: glues computer vision, models, and I/O. |
| **OpenCV (`opencv-python`)** | Video capture, drawing boxes/overlays, image crops, saving snapshots. |
| **NumPy** | Array math for geometry (IoU, distances, overlaps) on bounding boxes. |
| **Ultralytics YOLOv8 (`ultralytics`)** | State-of-the-art **object detection**; uses **PyTorch** under the hood when you run YOLO. Weights (e.g. `yolov8s.pt`) can **auto-download** on first use. |
| **TensorFlow / Keras (`tensorflow`)** | Optional **image classification** for phone-in-hand style checks (e.g. `.h5` classifier), if you train or supply that model. |
| **Pillow** | Image handling helpers where needed alongside OpenCV. |
| **Standard library** | `threading` for async inference vs UI loop; `smtplib` / `email` for optional Gmail alerts; `pathlib`, `datetime`, `time` for files and timing. |

**Conceptual labels you can use in a report:** *computer vision*, *deep learning*, *object detection*, *optional classification*, *real-time video analytics*, *multi-threaded pipeline*, *alerting and logging*.

---

## 4. Project structure (main files)

- **`detector.py`** — Entry point and orchestration: loads settings, models, capture device, inference thread, visualization, and alert triggers.
- **`config/settings.py`** — Thresholds, model paths, frame size, cooldowns, email toggles. This is where accuracy vs speed is tuned.
- **`models/yolo_detector.py`** — Wrapper around YOLO for COCO classes of interest (person, vehicles, bike, phone).
- **`models/phone_detector.py`**, **`models/seatbelt_detector.py`**, **`models/helmet_detector.py`** — Specialized logic for those behaviors (as implemented in your repo).
- **`utils/alert_manager.py`** — Cooldowns, console/log output, snapshots, optional email for HIGH severity.
- **`utils/logger.py`** — Session logging to files under `logs/`.
- **`requirements.txt`** — Python dependencies and versions.

---

## 5. Configuration and models (talking points)

- **Base detector:** YOLO variant (e.g. `yolov8s.pt` in settings) trades off **speed vs accuracy**.
- **Phone:** Can lean on YOLO’s cell-phone class and/or a separate classifier path defined in settings.
- **Seatbelt / helmet:** Expect **domain-specific** weights (often fine-tuned YOLO on traffic or worker-safety datasets). Without files, those modules may degrade gracefully or skip (depending on implementation).
- **Ethics / deployment:** Real-world use needs privacy review, consent, jurisdiction-specific rules, and validation on your target cameras; this codebase is a **technical prototype** for learning and research.

---

## 6. How to run (short)

```bash
pip install -r requirements.txt
python detector.py                 # default webcam
python detector.py --source path/to/video.mp4