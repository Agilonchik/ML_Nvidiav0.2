import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers(): return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    log_file = log_dir / "project.log"
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
