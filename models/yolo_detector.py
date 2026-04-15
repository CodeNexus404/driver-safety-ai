"""
YOLOv8 / YOLOv5 Object Detector Wrapper
Detects: person, car, motorcycle, truck, bus, bicycle
"""

class YOLODetector:
    """
    Wraps Ultralytics YOLOv8 for object detection.

    Model download:
      pip install ultralytics
      # Model is auto-downloaded on first run, or manually:
      # https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt  (nano, fastest)
      # https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt  (small)
      # https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt  (medium, recommended)
    """

    COCO_CLASSES_OF_INTEREST = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        67: "cell phone",
    }

    def __init__(
        self,
        model_path="yolov8m.pt",
        conf_threshold=0.45,
        device="cpu",
        imgsz=480,
        iou=0.50,
    ):
        self.conf_threshold = conf_threshold
        self.device = device
        self.imgsz = imgsz
        self.iou = iou
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)  # auto-downloads if not present
            print(f"[YOLO] Loaded model: {model_path}")
            return model
        except ImportError:
            print("[YOLO] ultralytics not installed. Running in MOCK mode.")
            print("       Install with: pip install ultralytics")
            return None
        except Exception as e:
            print(f"[YOLO] Failed to load model ({e}). Running in MOCK mode.")
            return None

    def detect(self, frame, imgsz=None):
        """
        Run inference on a BGR frame.
        Returns list of dicts: {label, confidence, box: (x,y,w,h)}
        """
        if self.model is None:
            return self._mock_detect(frame)

        size = imgsz if imgsz is not None else self.imgsz
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou,
            device=self.device,
            imgsz=size,
            max_det=50,
        )[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.COCO_CLASSES_OF_INTEREST.get(cls_id)
            if label is None:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": conf,
                "box": (x1, y1, x2 - x1, y2 - y1),
            })
        return detections

    def _mock_detect(self, frame):
        """Returns dummy detections for testing without a real model."""
        h, w = frame.shape[:2]
        return [
            {"label": "person",     "confidence": 0.91, "box": (int(w*0.1), int(h*0.1), 120, 300)},
            {"label": "motorcycle", "confidence": 0.85, "box": (int(w*0.1), int(h*0.3), 160, 200)},
            {"label": "car",        "confidence": 0.88, "box": (int(w*0.5), int(h*0.2), 200, 180)},
        ]
