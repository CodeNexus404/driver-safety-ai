"""
Alert Manager
Handles: console alerts, sound beeps, email/SMS notifications, snapshot saving.
"""

import cv2
import smtplib
import threading
import time
from datetime import datetime
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from config.settings import Settings


class AlertManager:
    def __init__(self, logger):
        self.logger = logger
        self.settings = Settings()
        self._cooldown = {}           # alert_type → last_triggered time

    def trigger(self, alert: dict, frame=None):
        """
        Main entry point. Dispatches alert through all configured channels.
        alert = {"type", "severity", "confidence", "box", "message"}
        """
        atype = alert["type"]
        now = time.time()

        # Cooldown: don't repeat same alert within N seconds
        cooldown_sec = self.settings.ALERT_COOLDOWN.get(atype, 5)
        if now - self._cooldown.get(atype, 0) < cooldown_sec:
            return False
        self._cooldown[atype] = now

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Console
        self._console_alert(alert, ts)

        # Log to file
        self.logger.warning(
            f"[ALERT] {atype} | {alert['severity']} | conf={alert['confidence']:.0%} | {ts}"
        )

        # Save snapshot with alert overlay
        snap_path = None
        if frame is not None:
            snap_path = self._save_alert_snapshot(frame, alert, ts)

        # Email (runs in background thread to avoid blocking)
        if self.settings.EMAIL_ENABLED and alert["severity"] == "HIGH":
            threading.Thread(
                target=self._send_email, args=(alert, ts, snap_path), daemon=True
            ).start()

        # Sound beep
        self._beep(alert["severity"])
        return True

    # ── Channels ─────────────────────────────────────────────────────────────

    def _console_alert(self, alert, ts):
        colors = {"HIGH": "\033[91m", "MEDIUM": "\033[93m", "LOW": "\033[96m"}
        reset = "\033[0m"
        c = colors.get(alert["severity"], "")
        print(f"\n{c}{'='*60}")
        print(f"  🚨 ALERT  [{alert['severity']}]  {ts}")
        print(f"  Type      : {alert['type']}")
        print(f"  Message   : {alert['message']}")
        print(f"  Confidence: {alert['confidence']:.0%}")
        print(f"{'='*60}{reset}\n")

    def _save_alert_snapshot(self, frame, alert, ts):
        snap_dir = Path("output/alerts")
        snap_dir.mkdir(parents=True, exist_ok=True)
        safe_ts = ts.replace(":", "-").replace(" ", "_")
        path = snap_dir / f"{alert['type']}_{safe_ts}.jpg"
        cv2.imwrite(str(path), frame)
        self.logger.info(f"[SNAP] Alert snapshot saved: {path}")
        return str(path)

    def _send_email(self, alert, ts, snap_path=None):
        cfg = self.settings.EMAIL_CONFIG
        try:
            msg = MIMEMultipart()
            msg["From"] = cfg["sender"]
            msg["To"] = cfg["receiver"]
            msg["Subject"] = f"🚨 Driver Safety Alert: {alert['type']} [{alert['severity']}]"

            body = f"""
            <h2 style="color:red;">Driver Safety Alert</h2>
            <table>
              <tr><td><b>Type</b></td><td>{alert['type']}</td></tr>
              <tr><td><b>Severity</b></td><td>{alert['severity']}</td></tr>
              <tr><td><b>Message</b></td><td>{alert['message']}</td></tr>
              <tr><td><b>Confidence</b></td><td>{alert['confidence']:.0%}</td></tr>
              <tr><td><b>Timestamp</b></td><td>{ts}</td></tr>
            </table>
            """
            msg.attach(MIMEText(body, "html"))

            if snap_path:
                with open(snap_path, "rb") as f:
                    img = MIMEImage(f.read())
                    img.add_header("Content-Disposition", "attachment",
                                   filename=Path(snap_path).name)
                    msg.attach(img)

            with smtplib.SMTP_SSL(cfg["smtp_host"], cfg["smtp_port"]) as server:
                server.login(cfg["sender"], cfg["password"])
                server.sendmail(cfg["sender"], cfg["receiver"], msg.as_string())
            self.logger.info(f"[EMAIL] Alert email sent for {alert['type']}")
        except Exception as e:
            self.logger.error(f"[EMAIL] Failed to send: {e}")

    def _beep(self, severity):
        """Cross-platform terminal beep."""
        if severity == "HIGH":
            print("\a\a", end="", flush=True)
        else:
            print("\a", end="", flush=True)
