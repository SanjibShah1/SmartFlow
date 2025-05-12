import os
from pathlib import Path

# Suppress OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set CUDA_VISIBLE_DEVICES (adjust as needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Global constants
COUNTING_LINE_Y = 350  # Adjust this based on your video
SPEED_LIMIT = 50  # Speed limit in km/h
TRACKER_CONFIGS = Path(os.getenv("YOLO_TRACKER_DIR", "trackers"))
TRACKING_CONFIG = TRACKER_CONFIGS / "botsort.yaml"