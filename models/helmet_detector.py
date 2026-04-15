"""
Helmet detection for two-wheeler riders (motorcycle / bicycle).

Place a YOLOv8 model trained on helmet datasets at models/helmet_detector.pt
(e.g. Roboflow: search "helmet detection motorcycle" → export YOLOv8).

Class names are matched flexibly: no_helmet, without_helmet, bare_head → not wearing;
helmet, hardhat, with_helmet, buckled (rare) → wearing.
"""

from pathlib import Path


class HelmetDetector:
    def __init__(self, model_path="models/helmet_detector.pt", conf_threshold=0.35):
        self.conf_threshold = conf_threshold
        self.model, self.backend = self._load_model(model_path)

    def _load_model(self, model_path):
        path = Path(model_path)
        if not path.is_file():
            print(
                f"[HelmetDet] No model at {model_path} — add a YOLOv8 helmet .pt "
                "or inference stays in heuristic mode (skin-cue on head crop)."
            )
            return None, "heuristic"
        try:
            from ultralytics import YOLO

            model = YOLO(str(path))
            print(f"[HelmetDet] Loaded YOLO model: {model_path}")
            return model, "yolo"
        except Exception as e:
            print(f"[HelmetDet] Failed to load YOLO ({e}) — using heuristic fallback.")
            return None, "heuristic"

    def detect(self, head_region):
        """
        head_region: BGR crop of upper head / helmet area.
        Returns: {"wearing_helmet": bool, "confidence": float}
        """
        if head_region is None or head_region.size == 0:
            return {"wearing_helmet": True, "confidence": 0.5}

        if self.backend == "yolo":
            return self._yolo_detect(head_region)
        return self._heuristic_detect(head_region)

    def _normalize_label(self, name):
        return name.lower().replace(" ", "_").replace("-", "_")

    def _yolo_detect(self, region):
        results = self.model(
            region, verbose=False, conf=self.conf_threshold, imgsz=320
        )[0]
        best_helmet = 0.0
        best_no = 0.0
        no_terms = (
            "no_helmet",
            "without_helmet",
            "nohelmet",
            "bare_head",
            "bare",
            "head_only",
            "person_without",
            "unhelmeted",
        )
        yes_terms = (
            "with_helmet",
            "wearing_helmet",
            "hard_hat",
            "hardhat",
            "helmet",
            "safe",
        )
        for box in results.boxes:
            label = self._normalize_label(results.names[int(box.cls[0])])
            conf = float(box.conf[0])
            if any(t in label for t in no_terms):
                best_no = max(best_no, conf)
            elif any(t in label for t in yes_terms):
                best_helmet = max(best_helmet, conf)

        if best_no > best_helmet and best_no >= self.conf_threshold:
            return {"wearing_helmet": False, "confidence": best_no}
        if best_helmet >= self.conf_threshold:
            return {"wearing_helmet": True, "confidence": best_helmet}
        return {"wearing_helmet": True, "confidence": 0.5}

    def _heuristic_detect(self, region):
        """HSV skin-like pixels in head crop — exposed skin suggests no helmet (weak signal)."""
        import cv2
        import numpy as np

        h, w = region.shape[:2]
        if h < 16 or w < 16:
            return {"wearing_helmet": True, "confidence": 0.5}

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # Broad skin-ish range (works for many lighting conditions; tune if needed)
        lower = np.array([0, 30, 60], dtype=np.uint8)
        upper = np.array([25, 180, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        # Focus upper half of crop (forehead / hairline area)
        top = mask[0 : max(1, h // 2), :]
        ratio = cv2.countNonZero(top) / max(top.size, 1)

        if ratio > 0.22:
            return {"wearing_helmet": False, "confidence": min(0.82, 0.45 + ratio)}
        if ratio < 0.045:
            return {"wearing_helmet": True, "confidence": min(0.8, 0.5 + (0.08 - ratio) * 3)}
        return {"wearing_helmet": True, "confidence": 0.5}
