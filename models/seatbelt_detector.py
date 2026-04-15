"""
Seatbelt detection for vehicle occupants (driver / front passenger).

Place a YOLOv8 model trained on seatbelt datasets at models/seatbelt_detector.pt
(e.g. Roboflow: search "seatbelt detection" → export YOLOv8).

Class names are matched flexibly: no_seatbelt, unbuckled, without → not wearing;
seatbelt, buckled, with_seatbelt → wearing.
"""

from pathlib import Path


class SeatbeltDetector:
    def __init__(self, model_path="models/seatbelt_detector.pt", conf_threshold=0.35):
        self.conf_threshold = conf_threshold
        self.model, self.backend = self._load_model(model_path)

    def _load_model(self, model_path):
        path = Path(model_path)
        if not path.is_file():
            print(
                f"[SeatbeltDet] No model at {model_path} — add a YOLOv8 seatbelt .pt "
                "or inference stays in heuristic mode (conservative)."
            )
            return None, "heuristic"
        try:
            from ultralytics import YOLO

            model = YOLO(str(path))
            print(f"[SeatbeltDet] Loaded YOLO model: {model_path}")
            return model, "yolo"
        except Exception as e:
            print(f"[SeatbeltDet] Failed to load YOLO ({e}) — using heuristic fallback.")
            return None, "heuristic"

    def detect(self, torso_region):
        """
        torso_region: BGR crop of upper body (chest / shoulder area).
        Returns: {"wearing_seatbelt": bool, "confidence": float}
        """
        if torso_region is None or torso_region.size == 0:
            return {"wearing_seatbelt": True, "confidence": 0.5}

        if self.backend == "yolo":
            return self._yolo_detect(torso_region)
        return self._heuristic_detect(torso_region)

    def _normalize_label(self, name):
        return name.lower().replace(" ", "_").replace("-", "_")

    def _yolo_detect(self, region):
        results = self.model(
            region, verbose=False, conf=self.conf_threshold, imgsz=384
        )[0]
        best_buckled = 0.0
        best_unbuckled = 0.0
        for box in results.boxes:
            label = self._normalize_label(results.names[int(box.cls[0])])
            conf = float(box.conf[0])
            if any(
                k in label
                for k in (
                    "no_seat",
                    "without",
                    "unbuckled",
                    "no_belt",
                    "unbelted",
                    "not_wearing",
                )
            ):
                best_unbuckled = max(best_unbuckled, conf)
            elif any(
                k in label
                for k in ("seatbelt", "buckled", "belt", "with_seat", "fastened")
            ):
                best_buckled = max(best_buckled, conf)

        if best_unbuckled > best_buckled and best_unbuckled >= self.conf_threshold:
            return {"wearing_seatbelt": False, "confidence": best_unbuckled}
        if best_buckled >= self.conf_threshold:
            return {"wearing_seatbelt": True, "confidence": best_buckled}
        # Ambiguous: assume buckled to limit false positives without a good model
        return {"wearing_seatbelt": True, "confidence": 0.5}

    def _heuristic_detect(self, region):
        """Very light strap cue: diagonal edges in torso; weak signal, biased to 'buckled'."""
        import cv2
        import numpy as np

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 40, 120)
        h, w = edges.shape[:2]
        if h < 20 or w < 20:
            return {"wearing_seatbelt": True, "confidence": 0.5}

        # Focus central band (typical strap path)
        band = edges[int(h * 0.15) : int(h * 0.85), int(w * 0.2) : int(w * 0.8)]
        length = cv2.countNonZero(band)
        ratio = length / max(band.size, 1)

        # Strong edge density may indicate visible strap; still keep bar high
        if ratio > 0.12:
            return {"wearing_seatbelt": True, "confidence": min(0.85, ratio * 4)}
        if ratio < 0.024:
            return {"wearing_seatbelt": False, "confidence": 0.58}
        return {"wearing_seatbelt": True, "confidence": 0.5}
