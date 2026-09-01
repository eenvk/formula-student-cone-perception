# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path

from dataset.dataset_utils import DETECTION_ID_TO_NAME

SEED = 2026

SPLIT_RATIO = 0.20
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 35
GLOBAL_CLIPNORM = 10.0

IMAGE_SIZE = (800, 800)

# Number of prefetched batches.
PREFETCH_BUFFER = 2

# Number of parallel calls used in the heavier tf.data operations.
NUM_PARALLEL_CALLS = 2

# Validation / COCO evaluation frequency.
EVAL_EVERY = 5
VALIDATION_FREQ = 5

RESUME_TRAINING = False
RESUME_FROM_EPOCH = 0

CLASS_MAPPING = DETECTION_ID_TO_NAME
NUM_CLASSES = len(CLASS_MAPPING)

DETECTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DETECTION_DIR.parent.parent

CHECKPOINT_PATH = DETECTION_DIR / "best_yolov8.weights.h5"
DETECTIONS_PATH = DETECTION_DIR / "detections.png"
