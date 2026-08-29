# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path

SEED = 2026

SPLIT_RATIO = 0.20
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 25
GLOBAL_CLIPNORM = 10.0

IMAGE_SIZE = (800, 800)

# Number of prefetched batches.
PREFETCH_BUFFER = 2

# Number of parallel calls used in the heavier tf.data operations.
NUM_PARALLEL_CALLS = 2

# Validation / COCO evaluation frequency.
EVAL_EVERY = 5
VALIDATION_FREQ = 5

CLASS_IDS = [
    "blue_cone",
    "yellow_cone",
    "orange_cone",
    "large_orange_cone",
    "unknown_cone",
]

RESUME_TRAINING = False
RESUME_FROM_EPOCH = 0

NUM_CLASSES = len(CLASS_IDS)

CLASS_TO_ID = {
    class_name: class_id
    for class_id, class_name in enumerate(CLASS_IDS)
}

CLASS_MAPPING = {
    class_id: class_name
    for class_id, class_name in enumerate(CLASS_IDS)
}

DETECTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DETECTION_DIR.parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
    / "fsoco_bounding_boxes_train"
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CHECKPOINT_PATH = DETECTION_DIR / "best_yolov8.weights.h5"
DETECTIONS_PATH = DETECTION_DIR / "detections.png"
