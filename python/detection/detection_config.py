#Granati

from pathlib import Path
from dataset.dataset_utils import DETECTION_ID_TO_NAME


SEED = 2026

# Dataset split and batching
SPLIT_RATIO = 0.20
BATCH_SIZE = 4
IMAGE_SIZE = (800, 800)

PREFETCH_BUFFER = 2
NUM_PARALLEL_CALLS = 2

# Training hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 35
GLOBAL_CLIPNORM = 10.0

# Validation / evaluation
# Run validation every N epochs during model.fit().
VALIDATION_FREQ = 2
# Run the inference evaluation
EVAL_EVERY = 5

# Training crops
MIN_RETAINED_AREA = 0.3
NUM_NEGATIVE_CROP_ATTEMPTS = 40

# Classes
CLASS_MAPPING = DETECTION_ID_TO_NAME
NUM_CLASSES = len(CLASS_MAPPING)

# Project paths
DETECTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DETECTION_DIR.parent.parent
DETECTION_WEIGHTS_PATH = PROJECT_DIR / "models" / "best_yolov8.weights.h5"

# Inference
PATCH_BATCH_SIZE = 8
PATCH_BORDER_MARGIN = 20
OVERLAP = 200

#Prediction filters
CONFIDENCE_THRESHOLD = 0.25
GLOBAL_NMS_IOU_THRESHOLD = 0.3

# Containment threshold used for predictions of the same class.
GLOBAL_CONTAINMENT_THRESHOLD = 0.6
# Stricter containment threshold used for predictions of different classes.
GLOBAL_CROSS_CONTAINMENT_THRESHOLD = 0.8

#yolo standard values
yolo_SCORE_THRESHOLD = 0.25
yolo_NMS_IOU_THRESHOLD = 0.70
yolo_NMS_MAX_DETECTIONS = 100

#hyperparam
yolo_PRE_NMS_TOP_K = 200
