#Granati

from pathlib import Path
from dataset.dataset_utils import DETECTION_ID_TO_NAME


SEED = 2026

# Dataset split and batching
SPLIT_RATIO = 0.20
BATCH_SIZE = 4

IMAGE_SIZE = (800, 800)

# Number of prefetched batches.
PREFETCH_BUFFER = 2

# Number of parallel calls used in heavier tf.data operations.
NUM_PARALLEL_CALLS = 2


# ============================================================
# Training hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 45
GLOBAL_CLIPNORM = 10.0


# ============================================================
# Validation / evaluation
# Run validation every N epochs during model.fit().
VALIDATION_FREQ = 2
# Run the full inference / COCO evaluation every N epochs.
EVAL_EVERY = 5

# ============================================================
# Classes
CLASS_MAPPING = DETECTION_ID_TO_NAME
NUM_CLASSES = len(CLASS_MAPPING)

# ============================================================
# Project paths
DETECTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DETECTION_DIR.parent.parent

DETECTION_WEIGHTS_PATH = PROJECT_DIR / "models" / "best_yolov8.weights.h5"
DETECTIONS_PATH = DETECTION_DIR / "detections.png"

RAW_PREDICTIONS_DIR = PROJECT_DIR / "python" / "evaluation" / "raw_detection_prediction"

# ============================================================
# Inference batching
# Number of 800x800 views processed by the model in one batch.
PATCH_BATCH_SIZE = 8

# ============================================================
# Patch inference
# Distance in pixels between consecutive patch origins.
PATCH_STRIDE = 600
# Margin used to identify predictions close to internal patch borders.
PATCH_BORDER_MARGIN = 20

# ============================================================
# Prediction filtering and NMS
# Minimum confidence required to keep a prediction.
CONFIDENCE_THRESHOLD = 0.25
# IoU threshold used to suppress overlapping predictions.
GLOBAL_NMS_IOU_THRESHOLD = 0.5
# Containment threshold used for predictions of the same class.
GLOBAL_CONTAINMENT_THRESHOLD = 0.8
# Stricter containment threshold used for predictions of different classes.
GLOBAL_CROSS_CONTAINMENT_THRESHOLD = 0.90

# ============================================================
# Training crops
# Minimum fraction of the original bounding-box area
# that must remain inside the crop.
MIN_RETAINED_AREA = 0.3
# Maximum number of attempts used to generate a valid negative crop.
NUM_NEGATIVE_CROP_ATTEMPTS = 20

#Evaluation
RUN_MODEL_INFERENCE = True