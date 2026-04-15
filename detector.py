"""
Driver Safety AI Detection System
SDG 3 – Good Health & Well-being
SDG 9: Industry, Innovation and Infrastructure
SDG 11: Sustainable Cities, and Communities
Detects: Phone usage while driving, No seatbelt (car occupants), No helmet (bike riders),
         Dangerous road actions
"""

import cv2
import numpy as np
import threading
import time
from datetime import datetime
from pathlib import Path

from utils.alert_manager import AlertManager
from utils.logger import Logger
from models.yolo_detector import YOLODetector
from models.phone_detector import PhoneDetector
from models.seatbelt_detector import SeatbeltDetector
from models.helmet_detector import HelmetDetector
from config.settings import Settings


class RiskyBehaviorDetector:
    def __init__(self, source=0, output_dir="output"):
        self.settings = Settings()
        self.source = source
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = Logger(log_dir="logs")
        self.alert_manager = AlertManager(self.logger)

        print("[INIT] Loading AI models...")
        self.yolo = YOLODetector(
            self.settings.YOLO_MODEL_PATH,
            imgsz=self.settings.INFERENCE_IMGSZ,
            conf_threshold=self.settings.YOLO_CONF,
            iou=self.settings.YOLO_IOU,
        )
        self.phone_detector = PhoneDetector(self.settings.PHONE_MODEL_PATH)
        self.seatbelt_detector = SeatbeltDetector(
            self.settings.SEATBELT_MODEL_PATH,
            conf_threshold=self.settings.SEATBELT_THRESHOLD,
        )
        self.helmet_detector = HelmetDetector(
            self.settings.HELMET_MODEL_PATH,
            conf_threshold=self.settings.HELMET_THRESHOLD,
        )
        print("[INIT] All models loaded successfully!")

        self.cap = None
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.alerts_today = 0
        self.detection_stats = {
            "phone_usage": 0,
            "no_seatbelt": 0,
            "no_helmet": 0,
            "dangerous_action": 0,
        }

        self._frame_lock = threading.Lock()
        self._viz_lock = threading.Lock()
        self._latest_frame = None
        self._viz_boxes = []
        self._overlay_alert = None
        self._overlay_until = 0.0
        self._inference_thread = None
        self._inference_running = False
        # Temporal confirmation: "alert_key" -> consecutive hit count
        self._confirm_streak = {}

    def start(self):
        """Start the detection pipeline."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.FRAME_HEIGHT)

        self.running = True
        self._inference_running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()

        print(f"[START] Detection started on source: {self.source}")
        self._run_loop()

    def _run_loop(self):
        """Display loop: keep video smooth; inference runs in a background thread."""
        prev_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] Frame not received. Retrying...")
                time.sleep(0.05)
                continue

            self.frame_count += 1
            with self._frame_lock:
                self._latest_frame = frame

            now = time.time()
            self.fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            display = frame.copy()
            with self._viz_lock:
                boxes = list(self._viz_boxes)
                overlay = self._overlay_alert
                show_overlay = time.time() < self._overlay_until and overlay is not None

            for det in boxes:
                self._draw_box(
                    display,
                    det["box"],
                    det["label"],
                    det["confidence"],
                    det["color"],
                )
            if show_overlay:
                self._draw_alert_overlay(display, overlay)

            display = self._draw_hud(display)
            cv2.imshow("Driver Safety AI", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.stop()
            elif key == ord("s"):
                self._save_snapshot(display)

        self._cleanup()

    def _inference_loop(self):
        """Run heavy models without blocking the display thread."""
        while self._inference_running:
            with self._frame_lock:
                fr = None if self._latest_frame is None else self._latest_frame.copy()
            if fr is None:
                time.sleep(0.01)
                continue

            boxes_out, fired = self._run_detection(fr)
            with self._viz_lock:
                self._viz_boxes = boxes_out
                if fired:
                    self._overlay_alert = fired
                    self._overlay_until = (
                        time.time() + self.settings.ALERT_OVERLAY_TTL_SEC
                    )

            time.sleep(0.001)

    def _run_detection(self, frame):
        """Returns (list of draw dicts, last_fired_alert_or_None)."""
        alerts = []
        boxes_out = []
        hit_keys = set()

        yolo_results = self.yolo.detect(
            frame, imgsz=self.settings.INFERENCE_IMGSZ
        )
        persons = [r for r in yolo_results if r["label"] == "person"]
        cars = [r for r in yolo_results if r["label"] == "car"]
        bikes = [
            r
            for r in yolo_results
            if r["label"] in ("motorcycle", "bicycle")
        ]
        cell_phones = [r for r in yolo_results if r["label"] == "cell phone"]

        for det in yolo_results:
            color = self.settings.CLASS_COLORS.get(det["label"], (200, 200, 200))
            boxes_out.append(
                {
                    "box": det["box"],
                    "label": det["label"],
                    "confidence": det["confidence"],
                    "color": color,
                }
            )

        need = self.settings.DETECTION_CONFIRM_FRAMES
        grid = self.settings.CONFIRM_GRID_PX

        for person in persons:
            if not self._person_associated_with_vehicle(
                person["box"], cars, for_phone=True
            ):
                continue
            region = self._crop_region(frame, person["box"])
            if region is None:
                continue
            phone_result = self.phone_detector.detect(region)
            cls_conf = (
                float(phone_result["confidence"])
                if phone_result["detected"]
                else 0.0
            )
            yolo_ph = self._max_phone_overlap_conf(person["box"], cell_phones)
            fused = max(cls_conf, yolo_ph)
            if fused < self.settings.PHONE_THRESHOLD:
                continue
            if cls_conf < self.settings.PHONE_THRESHOLD:
                if yolo_ph < self.settings.PHONE_YOLO_MIN_CONF:
                    continue
                if (
                    self._max_phone_iou(person["box"], cell_phones)
                    < self.settings.PHONE_YOLO_MIN_IOU
                ):
                    continue

            k = f"phone:{self._confirm_grid_key(person['box'], grid)}"
            hit_keys.add(k)
            self._confirm_streak[k] = self._confirm_streak.get(k, 0) + 1
            if self._confirm_streak[k] < need:
                continue
            alerts.append(
                {
                    "type": "PHONE_WHILE_DRIVING",
                    "severity": "HIGH",
                    "confidence": fused,
                    "box": person["box"],
                    "message": "Phone usage while driving detected!",
                }
            )
            self._confirm_streak[k] = 0

        for person in persons:
            if not self._person_associated_with_vehicle(
                person["box"], cars, for_phone=False
            ):
                continue
            torso = self._crop_torso_region(frame, person["box"])
            sb = self.seatbelt_detector.detect(torso)
            if sb["wearing_seatbelt"]:
                continue
            k = f"seat:{self._confirm_grid_key(person['box'], grid)}"
            hit_keys.add(k)
            self._confirm_streak[k] = self._confirm_streak.get(k, 0) + 1
            if self._confirm_streak[k] < need:
                continue
            alerts.append(
                {
                    "type": "NO_SEATBELT",
                    "severity": "HIGH",
                    "confidence": sb["confidence"],
                    "box": person["box"],
                    "message": "Seatbelt not fastened (driver/passenger)!",
                }
            )
            self._confirm_streak[k] = 0

        if bikes:
            for person in persons:
                if not self._person_associated_with_bike(person["box"], bikes):
                    continue
                head = self._crop_head_region(frame, person["box"])
                hm = self.helmet_detector.detect(head)
                if hm["wearing_helmet"]:
                    continue
                k = f"helmet:{self._confirm_grid_key(person['box'], grid)}"
                hit_keys.add(k)
                self._confirm_streak[k] = self._confirm_streak.get(k, 0) + 1
                if self._confirm_streak[k] < need:
                    continue
                alerts.append(
                    {
                        "type": "NO_HELMET",
                        "severity": "HIGH",
                        "confidence": hm["confidence"],
                        "box": person["box"],
                        "message": "Rider without helmet (motorcycle/bicycle)!",
                    }
                )
                self._confirm_streak[k] = 0

        self._decay_confirm_streaks(hit_keys)

        dangerous = self._detect_dangerous_actions(yolo_results, frame)
        alerts.extend(dangerous)

        last_fired = None
        for alert in alerts:
            if self._fire_alert(alert, frame):
                last_fired = alert

        return boxes_out, last_fired

    def _fire_alert(self, alert, frame):
        """Trigger alert channels and bump stats only when not in cooldown."""
        if not self.alert_manager.trigger(alert, frame):
            return False
        self.alerts_today += 1
        at = alert["type"]
        if at == "PHONE_WHILE_DRIVING":
            self.detection_stats["phone_usage"] += 1
        elif at == "NO_SEATBELT":
            self.detection_stats["no_seatbelt"] += 1
        elif at == "NO_HELMET":
            self.detection_stats["no_helmet"] += 1
        elif at == "TAILGATING":
            self.detection_stats["dangerous_action"] += 1
        return True

    def _detect_dangerous_actions(self, detections, frame):
        """Same-lane following distance heuristic (geometric filters)."""
        alerts = []
        vehicles = [
            d
            for d in detections
            if d["label"] in ("car", "truck", "bus", "motorcycle")
        ]

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                b1, b2 = vehicles[i]["box"], vehicles[j]["box"]
                if not self._is_likely_same_lane_following(b1, b2):
                    continue
                gap = abs(b1[0] - b2[0])
                avg_width = (b1[2] + b2[2]) / 2
                if gap < avg_width * self.settings.TAILGATE_RATIO:
                    alerts.append(
                        {
                            "type": "TAILGATING",
                            "severity": "MEDIUM",
                            "confidence": 0.78,
                            "box": b1,
                            "message": "Tailgating / Unsafe following distance!",
                        }
                    )

        return alerts

    @staticmethod
    def _iou_xywh(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        iw = max(0, x2 - x1)
        ih = max(0, y2 - y1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _vertical_overlap_ratio(b1, b2):
        y1a, y1b = b1[1], b1[1] + b1[3]
        y2a, y2b = b2[1], b2[1] + b2[3]
        inter = max(0, min(y1b, y2b) - max(y1a, y2a))
        h = min(b1[3], b2[3])
        return inter / max(h, 1)

    def _person_center_in_padded_car(self, person_box, car_box):
        px = person_box[0] + person_box[2] // 2
        py = person_box[1] + person_box[3] // 2
        x, y, w, h = car_box
        p = self.settings.PERSON_VEHICLE_CENTER_PAD
        pw, ph = w * p, h * p
        return (x - pw <= px <= x + w + pw) and (y - ph <= py <= y + h + ph)

    def _person_associated_with_vehicle(self, person_box, cars, for_phone=False):
        """True if person is inside/overlapping a car (not just nearby on sidewalk)."""
        min_iou = self.settings.PERSON_VEHICLE_MIN_IOU
        dist_cap = (
            self.settings.PHONE_NEAR_CAR_PX if for_phone else self.settings.SEATBELT_NEAR_CAR_PX
        )
        for c in cars:
            cb = c["box"]
            if self._iou_xywh(person_box, cb) >= min_iou:
                return True
            if self._person_center_in_padded_car(person_box, cb):
                return True
        return self._is_near_vehicle(person_box, cars, threshold=dist_cap)

    def _person_associated_with_bike(self, person_box, bikes):
        """True if person overlaps / sits on a detected motorcycle or bicycle."""
        if not bikes:
            return False
        min_iou = self.settings.PERSON_VEHICLE_MIN_IOU
        dist_cap = self.settings.HELMET_NEAR_BIKE_PX
        for b in bikes:
            bb = b["box"]
            if self._iou_xywh(person_box, bb) >= min_iou:
                return True
            if self._person_center_in_padded_car(person_box, bb):
                return True
        return self._is_near_vehicle(person_box, bikes, threshold=dist_cap)

    def _max_phone_iou(self, person_box, cell_phones):
        best = 0.0
        for ph in cell_phones:
            best = max(best, self._iou_xywh(person_box, ph["box"]))
        return best

    def _max_phone_overlap_conf(self, person_box, cell_phones):
        best = 0.0
        for ph in cell_phones:
            if self._iou_xywh(person_box, ph["box"]) >= self.settings.PHONE_YOLO_MIN_IOU:
                best = max(best, float(ph["confidence"]))
        return best

    @staticmethod
    def _confirm_grid_key(box, grid_px):
        cx = (box[0] + box[2] // 2) // max(grid_px, 1)
        cy = (box[1] + box[3] // 2) // max(grid_px, 1)
        return f"{cx}:{cy}"

    def _decay_confirm_streaks(self, hit_keys):
        for k in list(self._confirm_streak.keys()):
            if k not in hit_keys:
                self._confirm_streak[k] = 0

    def _is_likely_same_lane_following(self, b1, b2):
        vov = self._vertical_overlap_ratio(b1, b2)
        if vov < self.settings.TAILGATE_MIN_VERTICAL_OVERLAP:
            return False
        h1, h2 = b1[3], b2[3]
        if max(h1, h2) <= 0:
            return False
        if abs(h1 - h2) / max(h1, h2) > self.settings.TAILGATE_MAX_HEIGHT_RATIO_DIFF:
            return False
        return True

    def _crop_region(self, frame, box):
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return None
        return frame[max(0, y) : y + h, max(0, x) : x + w]

    def _crop_torso_region(self, frame, box):
        """Chest / shoulder area where a shoulder belt is usually visible."""
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return None
        y1 = y + int(h * 0.25)
        y2 = y + int(h * 0.72)
        return frame[max(0, y1) : y2, max(0, x) : x + w]

    def _crop_head_region(self, frame, box):
        """Upper head / helmet area (top of person bbox)."""
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return None
        y2 = y + int(h * 0.42)
        inset = int(w * 0.08)
        x1 = max(0, x + inset)
        x2 = min(frame.shape[1], x + w - inset)
        if x2 <= x1:
            x1, x2 = max(0, x), min(frame.shape[1], x + w)
        return frame[max(0, y) : y2, x1:x2]

    def _is_near_vehicle(self, person_box, vehicles, threshold=150):
        px, py, pw, ph = person_box
        pcx, pcy = px + pw // 2, py + ph // 2
        for v in vehicles:
            vx, vy, vw, vh = v["box"]
            vcx, vcy = vx + vw // 2, vy + vh // 2
            dist = np.hypot(pcx - vcx, pcy - vcy)
            if dist < threshold:
                return True
        return False

    def _draw_box(self, frame, box, label, conf, color):
        x, y, w, h = box
        cv2.rectangle(frame, (x - 1, y - 1), (x + w + 1, y + h + 1), (45, 45, 48), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.52, 1
        (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
        pad = 4
        ty = y - pad
        top = ty - th - pad
        if top < 2:
            ty = min(y + h + th + pad * 2, frame.shape[0] - 2)
            top = ty - th - pad
        cv2.rectangle(
            frame, (x, top), (x + tw + pad * 2, ty + bl), (32, 34, 40), -1
        )
        cv2.rectangle(frame, (x, top), (x + tw + pad * 2, ty + bl), color, 1)
        cv2.putText(
            frame,
            text,
            (x + pad, ty - 2),
            font,
            scale,
            (248, 248, 252),
            thick,
            cv2.LINE_AA,
        )

    def _draw_alert_overlay(self, frame, alert):
        h, w = frame.shape[:2]
        border = {
            "HIGH": (52, 52, 240),
            "MEDIUM": (64, 150, 255),
            "LOW": (64, 255, 255),
        }.get(alert["severity"], (200, 200, 200))

        if alert["severity"] == "HIGH":
            cv2.rectangle(frame, (3, 3), (w - 4, h - 4), (42, 42, 220), 3)

        msg = alert["message"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, thick = 0.58, 2
        (tw, th), _ = cv2.getTextSize(msg, font, fs, thick)
        pad_x, pad_y = 18, 14
        bw = min(tw + pad_x * 2, w - 24)
        bh = th + pad_y * 2
        x0, y0 = 12, h - bh - 16
        x1, y1 = x0 + bw, y0 + bh
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return
        tint = np.empty_like(roi)
        tint[:, :] = (44, 38, 36)
        frame[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.35, tint, 0.65, 0)
        cv2.rectangle(frame, (x0, y0), (x1, y1), border, 2)
        cv2.putText(
            frame,
            msg,
            (x0 + pad_x, y0 + bh - pad_y + 4),
            font,
            fs,
            border,
            thick,
            cv2.LINE_AA,
        )

    def _draw_hud(self, frame):
        h, w = frame.shape[:2]
        accent = (255, 188, 72)  # amber/cyan mix BGR
        panel_w, panel_h = 292, 180
        ph = min(panel_h, h - 4)
        pw = min(panel_w, w - 4)
        roi = frame[0:ph, 0:pw].copy()
        shade = np.empty_like(roi)
        shade[:, :] = (44, 40, 38)
        frame[0:ph, 0:pw] = cv2.addWeighted(roi, 0.34, shade, 0.66, 0)
        cv2.line(frame, (0, 0), (0, ph - 1), accent, 4)

        fd = cv2.FONT_HERSHEY_DUPLEX
        fs = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Driver Safety", (14, 30), fd, 0.72, accent, 1, cv2.LINE_AA)
        cv2.putText(frame, "AI", (14, 52), fd, 0.52, (198, 202, 208), 1, cv2.LINE_AA)

        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (14, 76), fs, 0.48, (150, 155, 162), 1, cv2.LINE_AA)

        fps_col = (110, 230, 130) if self.fps >= 18 else (120, 200, 255)
        cv2.putText(
            frame,
            f"FPS {self.fps:.1f}",
            (14, 98),
            fs,
            0.52,
            fps_col,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Alerts (session)  {self.alerts_today}",
            (120, 98),
            fs,
            0.48,
            (210, 215, 220),
            1,
            cv2.LINE_AA,
        )

        def chip(x0, y0, dot_bgr, label, value):
            cv2.circle(frame, (x0 + 6, y0 - 4), 5, dot_bgr, -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{label} {value}",
                (x0 + 16, y0),
                fs,
                0.44,
                (220, 224, 228),
                1,
                cv2.LINE_AA,
            )

        chip(14, 120, (255, 140, 90), "Phone", self.detection_stats["phone_usage"])
        chip(130, 120, (255, 200, 100), "No belt", self.detection_stats["no_seatbelt"])
        chip(14, 142, (100, 220, 100), "No helm", self.detection_stats["no_helmet"])
        chip(130, 142, (120, 180, 255), "Tail/risk", self.detection_stats["dangerous_action"])

        hint = "Q quit   S snapshot"
        (hw, hh), _ = cv2.getTextSize(hint, fs, 0.42, 1)
        cv2.putText(
            frame,
            hint,
            (w - hw - 14, h - 12),
            fs,
            0.42,
            (130, 135, 145),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _save_snapshot(self, frame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"snapshot_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"[SNAP] Saved: {path}")

    def stop(self):
        self.running = False
        self._inference_running = False

    def _cleanup(self):
        self._inference_running = False
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=3.0)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Detection session ended.")
        print("[STOP] Detector stopped. Session log saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Driver Safety AI")
    parser.add_argument(
        "--source",
        default=0,
        help="Video source: 0=webcam, path=video file, rtsp://... for IP cam",
    )
    parser.add_argument(
        "--output", default="output", help="Output directory for snapshots"
    )
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    det = RiskyBehaviorDetector(source=source, output_dir=args.output)
    det.start()
