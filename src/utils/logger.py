import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

# -------------------------------
# Setup log directory and file
# -------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")

# -------------------------------
# Create logger
# -------------------------------
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)

# -------------------------------
# File handler (Rotating) with UTF-8
# -------------------------------
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5_000_000,
    backupCount=5,
    encoding="utf-8",  # important for Unicode emojis
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# -------------------------------
# Console handler with UTF-8
# -------------------------------
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

# -------------------------------
# Add handlers to logger
# -------------------------------
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# -------------------------------
# Test emoji logging
# -------------------------------
if __name__ == "__main__":
    logger.info("Logger initialized successfully!")
    logger.info("Testing emojis in logs...")
