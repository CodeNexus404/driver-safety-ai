"""Simple file + console logger."""

import logging
from pathlib import Path
from datetime import datetime


class Logger:
    def __init__(self, log_dir="logs"):
        Path(log_dir).mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"rbd_{ts}.log"

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self._log = logging.getLogger("RBD")

    def info(self, msg):    self._log.info(msg)
    def warning(self, msg): self._log.warning(msg)
    def error(self, msg):   self._log.error(msg)
    def debug(self, msg):   self._log.debug(msg)
