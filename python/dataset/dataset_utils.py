"""Dataset extraction, preprocessing, validation, and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import base64
import json
import random
import zlib

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Dataset paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

BOUNDING_BOXES_TRAIN_ROOT = DATASET_ROOT / "fsoco_bounding_boxes_train" / "iitb"
BOUNDING_BOXES_TRAIN_ANNOTATION_DIR = BOUNDING_BOXES_TRAIN_ROOT / "ann"
BOUNDING_BOXES_TRAIN_IMAGE_DIR = BOUNDING_BOXES_TRAIN_ROOT / "img"

SEGMENTATION_TRAIN_ROOT = DATASET_ROOT / "fsoco_segmentation_train" / "amz"
SEGMENTATION_TRAIN_ANNOTATION_DIR = SEGMENTATION_TRAIN_ROOT / "ann"
SEGMENTATION_TRAIN_IMAGE_DIR = SEGMENTATION_TRAIN_ROOT / "img"

SEGMENTATION_TEST_ROOT = DATASET_ROOT / "segmentation_test"
SEGMENTATION_TEST_ANNOTATION_DIR = SEGMENTATION_TEST_ROOT / "ann"
SEGMENTATION_TEST_IMAGE_DIR = SEGMENTATION_TEST_ROOT / "img"


# -----------------------------------------------------------------------------
# Class definitions
# -----------------------------------------------------------------------------

BACKGROUND_ID = 0
YELLOW_CONE_ID = 1
BLUE_CONE_ID = 2
SMALL_ORANGE_CONE_ID = 3
BIG_ORANGE_CONE_ID = 4
IGNORE_ID = 255

NUM_CLASSES = 5

CONE_CLASS_IDS = (YELLOW_CONE_ID, BLUE_CONE_ID, SMALL_ORANGE_CONE_ID, BIG_ORANGE_CONE_ID)

CLASS_ID_TO_NAME = {
    BACKGROUND_ID: "background",
    YELLOW_CONE_ID: "yellow_cone",
    BLUE_CONE_ID: "blue_cone",
    SMALL_ORANGE_CONE_ID: "small_orange_cone",
    BIG_ORANGE_CONE_ID: "big_orange_cone",
}

# FSOCO uses both unprefixed detection labels and seg_* segmentation labels.
CLASS_NAME_TO_ID = {
    "yellow_cone": YELLOW_CONE_ID,
    "blue_cone": BLUE_CONE_ID,
    "orange_cone": SMALL_ORANGE_CONE_ID,
    "small_orange_cone": SMALL_ORANGE_CONE_ID,
    "large_orange_cone": BIG_ORANGE_CONE_ID,
    "big_orange_cone": BIG_ORANGE_CONE_ID,
    "unknown_cone": IGNORE_ID,
    "seg_yellow_cone": YELLOW_CONE_ID,
    "seg_blue_cone": BLUE_CONE_ID,
    "seg_orange_cone": SMALL_ORANGE_CONE_ID,
    "seg_small_orange_cone": SMALL_ORANGE_CONE_ID,
    "seg_large_orange_cone": BIG_ORANGE_CONE_ID,
    "seg_big_orange_cone": BIG_ORANGE_CONE_ID,
    "seg_unknown_cone": IGNORE_ID,
}

# Detection models use four zero-based classes; segmentation reserves 0 for background.
SEGMENTATION_ID_TO_DETECTION_ID = {
    YELLOW_CONE_ID: 0,
    BLUE_CONE_ID: 1,
    SMALL_ORANGE_CONE_ID: 2,
    BIG_ORANGE_CONE_ID: 3,
}

DETECTION_ID_TO_SEGMENTATION_ID = {detection_id: segmentation_id for segmentation_id, detection_id in SEGMENTATION_ID_TO_DETECTION_ID.items()}

DETECTION_ID_TO_NAME = {
    0: "yellow_cone",
    1: "blue_cone",
    2: "small_orange_cone",
    3: "big_orange_cone",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

MASK_COLORS = {
    BACKGROUND_ID: (0, 0, 0),
    YELLOW_CONE_ID: (255, 255, 0),
    BLUE_CONE_ID: (0, 0, 255),
    SMALL_ORANGE_CONE_ID: (255, 165, 0),
    BIG_ORANGE_CONE_ID: (255, 0, 255),
    IGNORE_ID: (255, 255, 255),
}


# -----------------------------------------------------------------------------
# Shared data structures and loading utilities
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Box:
    """Bounding box using half-open xyxy coordinates: [x_min, x_max) x [y_min, y_max)."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    class_id: int
    score: float | None = None

    def __post_init__(self) -> None:
        if self.x_min < 0 or self.y_min < 0:
            raise ValueError("Box minimum coordinates must be non-negative.")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Box maximum coordinates must be greater than minimum coordinates.")
        if self.class_id not in CONE_CLASS_IDS:
            raise ValueError(f"Invalid shared cone class ID: {self.class_id}")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("Box score must be in the range [0, 1].")

    @property
    def area(self) -> int:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


def class_name_to_id(class_name: str) -> int:
    """Convert an FSOCO class name to the shared segmentation class ID."""

    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("class_name must be a non-empty string.")

    normalized_name = class_name.strip().lower()

    if normalized_name not in CLASS_NAME_TO_ID:
        raise ValueError(f"Unknown FSOCO class: {class_name}")

    return CLASS_NAME_TO_ID[normalized_name]


def segmentation_id_to_detection_id(class_id: int) -> int:
    """Convert a shared segmentation class ID into a zero-based detection ID."""

    if class_id not in SEGMENTATION_ID_TO_DETECTION_ID:
        raise ValueError(f"Class ID {class_id} cannot be used as a detection class.")

    return SEGMENTATION_ID_TO_DETECTION_ID[class_id]


def detection_id_to_segmentation_id(class_id: int) -> int:
    """Convert a zero-based detection ID into the shared segmentation class ID."""

    if class_id not in DETECTION_ID_TO_SEGMENTATION_ID:
        raise ValueError(f"Invalid detection class ID: {class_id}")

    return DETECTION_ID_TO_SEGMENTATION_ID[class_id]


def prediction_to_box(bbox_xyxy: Sequence[float], detection_class_id: int, score: float) -> Box:
    """Convert one model prediction to the shared Box representation."""

    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain exactly four coordinates.")

    x_min, y_min, x_max, y_max = (int(round(float(value))) for value in bbox_xyxy)
    return Box(x_min, y_min, x_max, y_max, detection_id_to_segmentation_id(int(detection_class_id)), float(score))


def load_annotation(annotation_path: str | Path) -> dict[str, Any]:
    """Load and validate the top-level structure of a JSON annotation."""

    annotation_path = Path(annotation_path)

    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    try:
        with annotation_path.open("r", encoding="utf-8") as file:
            annotation = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON annotation: {annotation_path}") from error

    if not isinstance(annotation, dict):
        raise ValueError(f"Annotation root must be a JSON object: {annotation_path}")
    if not isinstance(annotation.get("objects", []), list):
        raise ValueError(f"Annotation 'objects' must be a list: {annotation_path}")

    return annotation


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image in OpenCV BGR format."""

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Cannot decode image: {image_path}")

    return image


# -----------------------------------------------------------------------------
# Image and annotation pairing
# -----------------------------------------------------------------------------

def find_dataset_pairs(image_dir: str | Path, annotation_dir: str | Path) -> list[tuple[Path, Path]]:
    """Find deterministic image/annotation pairs inside one FSOCO subset."""

    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    images_by_name: dict[str, Path] = {}
    images_by_stem: dict[str, list[Path]] = {}

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        images_by_name[image_path.name] = image_path
        images_by_stem.setdefault(image_path.stem, []).append(image_path)

    pairs: list[tuple[Path, Path]] = []
    missing_images: list[str] = []

    for annotation_path in sorted(annotation_dir.glob("*.json")):
        expected_image_name = annotation_path.name.removesuffix(".json")
        image_path = images_by_name.get(expected_image_name)

        if image_path is None:
            stem_candidates = images_by_stem.get(annotation_path.stem, [])
            if len(stem_candidates) > 1:
                raise ValueError(f"Ambiguous image stem for annotation {annotation_path.name}: {stem_candidates}")
            if len(stem_candidates) == 1:
                image_path = stem_candidates[0]

        if image_path is None:
            missing_images.append(annotation_path.name)
            continue

        pairs.append((image_path, annotation_path))

    if missing_images:
        preview = ", ".join(missing_images[:5])
        suffix = "" if len(missing_images) <= 5 else f" and {len(missing_images) - 5} more"
        raise FileNotFoundError(f"Missing images for annotations in {annotation_dir}: {preview}{suffix}")

    return pairs


def find_all_dataset_pairs(dataset_root: str | Path) -> list[tuple[Path, Path]]:
    """Find pairs in a direct ann/img root or in all one-level subset roots."""

    dataset_root = Path(dataset_root)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    direct_image_dir = dataset_root / "img"
    direct_annotation_dir = dataset_root / "ann"

    if direct_image_dir.is_dir() or direct_annotation_dir.is_dir():
        if not direct_image_dir.is_dir() or not direct_annotation_dir.is_dir():
            raise FileNotFoundError(f"Both img and ann directories are required in {dataset_root}")
        return find_dataset_pairs(direct_image_dir, direct_annotation_dir)

    all_pairs: list[tuple[Path, Path]] = []

    for subset_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        image_dir = subset_dir / "img"
        annotation_dir = subset_dir / "ann"
        if image_dir.is_dir() or annotation_dir.is_dir():
            if not image_dir.is_dir() or not annotation_dir.is_dir():
                raise FileNotFoundError(f"Both img and ann directories are required in {subset_dir}")
            all_pairs.extend(find_dataset_pairs(image_dir, annotation_dir))

    if not all_pairs:
        raise ValueError(f"No image/annotation pairs found in {dataset_root}")

    return sorted(all_pairs, key=lambda pair: str(pair[0]))


def get_bounding_boxes_train_pairs() -> list[tuple[Path, Path]]:
    """Return pairs from dataset/fsoco_bounding_boxes_train/iitb."""

    return find_dataset_pairs(BOUNDING_BOXES_TRAIN_IMAGE_DIR, BOUNDING_BOXES_TRAIN_ANNOTATION_DIR)


def get_segmentation_train_pairs() -> list[tuple[Path, Path]]:
    """Return pairs from dataset/fsoco_segmentation_train/amz."""

    return find_dataset_pairs(SEGMENTATION_TRAIN_IMAGE_DIR, SEGMENTATION_TRAIN_ANNOTATION_DIR)


def get_segmentation_test_pairs() -> list[tuple[Path, Path]]:
    """Return pairs from dataset/segmentation_test."""

    return find_dataset_pairs(SEGMENTATION_TEST_IMAGE_DIR, SEGMENTATION_TEST_ANNOTATION_DIR)


def train_validation_split(pairs: Sequence[tuple[Path, Path]], validation_fraction: float = 0.20, seed: int = 42) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Create a deterministic train/validation split without mutating the input."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in the range [0.0, 1.0).")

    shuffled_pairs = list(pairs)
    random.Random(seed).shuffle(shuffled_pairs)
    validation_size = round(len(shuffled_pairs) * validation_fraction)

    if len(shuffled_pairs) > 1 and validation_fraction > 0.0:
        validation_size = min(max(1, validation_size), len(shuffled_pairs) - 1)

    return shuffled_pairs[validation_size:], shuffled_pairs[:validation_size]


# -----------------------------------------------------------------------------
# Supervisely bitmap decoding and ground-truth extraction
# -----------------------------------------------------------------------------

def decode_bitmap(bitmap_data: str) -> np.ndarray:
    """Decode a compressed Supervisely bitmap into a local boolean mask."""

    if not isinstance(bitmap_data, str) or not bitmap_data:
        raise ValueError("bitmap_data must be a non-empty base64 string.")

    try:
        compressed_data = base64.b64decode(bitmap_data, validate=True)
        image_data = zlib.decompress(compressed_data)
    except (ValueError, zlib.error) as error:
        raise ValueError("Invalid compressed Supervisely bitmap data.") from error

    decoded_image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if decoded_image is None:
        raise ValueError("Could not decode Supervisely bitmap image.")
    if decoded_image.ndim == 3 and decoded_image.shape[2] == 4:
        return decoded_image[:, :, 3] > 0
    if decoded_image.ndim == 2:
        return decoded_image > 0

    raise ValueError(f"Unexpected decoded bitmap shape: {decoded_image.shape}")


def bitmap_object_to_full_mask(obj: dict[str, Any], image_height: int, image_width: int) -> np.ndarray:
    """Convert one Supervisely bitmap object into a full-image boolean mask."""

    _validate_image_size(image_height, image_width)

    if obj.get("geometryType") != "bitmap":
        raise ValueError("The annotation object is not a bitmap.")

    bitmap = obj.get("bitmap")

    if not isinstance(bitmap, dict):
        raise ValueError("Missing bitmap data in annotation object.")

    bitmap_data = bitmap.get("data")
    origin = bitmap.get("origin")

    if bitmap_data is None or not isinstance(origin, (list, tuple)) or len(origin) != 2:
        raise ValueError("Bitmap annotation is missing valid data or origin.")

    local_mask = decode_bitmap(bitmap_data)
    origin_x, origin_y = int(origin[0]), int(origin[1])
    object_height, object_width = local_mask.shape

    x_start = max(0, origin_x)
    y_start = max(0, origin_y)
    x_end = min(image_width, origin_x + object_width)
    y_end = min(image_height, origin_y + object_height)

    full_mask = np.zeros((image_height, image_width), dtype=bool)

    if x_start >= x_end or y_start >= y_end:
        return full_mask

    bitmap_x_start = x_start - origin_x
    bitmap_y_start = y_start - origin_y
    bitmap_x_end = bitmap_x_start + (x_end - x_start)
    bitmap_y_end = bitmap_y_start + (y_end - y_start)
    full_mask[y_start:y_end, x_start:x_end] = local_mask[bitmap_y_start:bitmap_y_end, bitmap_x_start:bitmap_x_end]

    return full_mask


def annotation_to_semantic_mask(annotation_path: str | Path, image_height: int, image_width: int) -> np.ndarray:
    """Convert one Supervisely annotation to a five-class semantic mask."""

    _validate_image_size(image_height, image_width)
    annotation = load_annotation(annotation_path)
    semantic_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for obj in annotation.get("objects", []):
        if not isinstance(obj, dict) or obj.get("geometryType") != "bitmap":
            continue

        class_name = obj.get("classTitle")

        if class_name is None:
            continue

        class_id = class_name_to_id(class_name)
        semantic_mask[bitmap_object_to_full_mask(obj, image_height, image_width)] = class_id

    return semantic_mask


def binary_mask_to_bbox(binary_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the tight half-open xyxy box of a binary mask."""

    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be a 2D array.")

    y_coordinates, x_coordinates = np.nonzero(binary_mask)

    if x_coordinates.size == 0:
        return None

    return int(x_coordinates.min()), int(y_coordinates.min()), int(x_coordinates.max()) + 1, int(y_coordinates.max()) + 1


def rectangle_object_to_bbox(obj: dict[str, Any], image_height: int, image_width: int) -> tuple[int, int, int, int] | None:
    """Convert one Supervisely rectangle into a clipped half-open xyxy box."""

    _validate_image_size(image_height, image_width)

    if obj.get("geometryType") != "rectangle":
        raise ValueError("The annotation object is not a rectangle.")

    exterior = obj.get("points", {}).get("exterior", [])

    if not isinstance(exterior, list) or len(exterior) < 2:
        return None

    try:
        x_values = [float(point[0]) for point in exterior]
        y_values = [float(point[1]) for point in exterior]
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError("Invalid rectangle exterior coordinates.") from error

    x_min = max(0, int(np.floor(min(x_values))))
    y_min = max(0, int(np.floor(min(y_values))))
    x_max = min(image_width, int(np.floor(max(x_values))) + 1)
    y_max = min(image_height, int(np.floor(max(y_values))) + 1)

    if x_max <= x_min or y_max <= y_min:
        return None

    return x_min, y_min, x_max, y_max


def object_to_bbox(obj: dict[str, Any], image_height: int, image_width: int) -> tuple[int, int, int, int] | None:
    """Convert one bitmap or rectangle annotation object to a half-open xyxy box."""

    geometry_type = obj.get("geometryType")

    if geometry_type == "bitmap":
        return binary_mask_to_bbox(bitmap_object_to_full_mask(obj, image_height, image_width))
    if geometry_type == "rectangle":
        return rectangle_object_to_bbox(obj, image_height, image_width)

    return None


def annotation_to_instances(annotation_path: str | Path, image_height: int, image_width: int, bitmap_only: bool = False) -> list[Box]:
    """Extract one shared-class Box for every known cone instance."""

    _validate_image_size(image_height, image_width)
    annotation = load_annotation(annotation_path)
    instances: list[Box] = []

    for obj in annotation.get("objects", []):
        if not isinstance(obj, dict):
            continue
        if bitmap_only and obj.get("geometryType") != "bitmap":
            continue

        class_name = obj.get("classTitle")

        if class_name is None:
            continue

        class_id = class_name_to_id(class_name)

        if class_id == IGNORE_ID:
            continue

        coordinates = object_to_bbox(obj, image_height, image_width)

        if coordinates is not None:
            instances.append(Box(*coordinates, class_id))

    return instances


def annotation_to_bboxes(annotation_path: str | Path, image_height: int, image_width: int, bitmap_only: bool = False) -> list[dict[str, Any]]:
    """Extract serializable bounding boxes with shared and detection class IDs."""

    boxes = annotation_to_instances(annotation_path, image_height, image_width, bitmap_only=bitmap_only)

    return [
        {
            "class_name": CLASS_ID_TO_NAME[box.class_id],
            "segmentation_class_id": box.class_id,
            "detection_class_id": segmentation_id_to_detection_id(box.class_id),
            "bbox_xyxy": [box.x_min, box.y_min, box.x_max, box.y_max],
        }
        for box in boxes
    ]


def annotation_to_bboxes_from_masks(annotation_path: str | Path, image_height: int, image_width: int) -> list[dict[str, Any]]:
    """Derive one tight bounding box per bitmap instance for test evaluation."""

    return annotation_to_bboxes(annotation_path, image_height, image_width, bitmap_only=True)


def generate_bbox_ground_truth_from_segmentation(dataset_root: str | Path = SEGMENTATION_TEST_ROOT, output_json_path: str | Path | None = None) -> dict[str, Any]:
    """Generate detection ground truth from all test-set bitmap annotations."""

    pairs = find_all_dataset_pairs(dataset_root)
    ground_truth: dict[str, Any] = {"classes": DETECTION_ID_TO_NAME, "images": []}

    for image_path, annotation_path in pairs:
        image = load_image(image_path)
        image_height, image_width = image.shape[:2]
        boxes = annotation_to_bboxes_from_masks(annotation_path, image_height, image_width)
        ground_truth["images"].append({"image_name": image_path.name, "width": image_width, "height": image_height, "objects": boxes})

    if output_json_path is not None:
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open("w", encoding="utf-8") as file:
            json.dump(ground_truth, file, indent=2)

    return ground_truth


# -----------------------------------------------------------------------------
# Shared preprocessing and validation
# -----------------------------------------------------------------------------

def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image with bilinear interpolation."""

    _validate_image_array(image)
    _validate_image_size(height, width)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a class mask with nearest-neighbor interpolation."""

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")
    _validate_image_size(height, width)
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert an integer image to float32 values in the [0, 1] range."""

    _validate_image_array(image)

    if np.issubdtype(image.dtype, np.floating):
        minimum = float(np.min(image))
        maximum = float(np.max(image))
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError("Floating-point images must already be in the range [0, 1].")
        return image.astype(np.float32, copy=False)

    if not np.issubdtype(image.dtype, np.integer):
        raise TypeError(f"Unsupported image dtype: {image.dtype}")

    maximum_value = np.iinfo(image.dtype).max
    return image.astype(np.float32) / float(maximum_value)


def validate_dataset_pairs(pairs: Sequence[tuple[Path, Path]]) -> None:
    """Check that every image and JSON annotation can be loaded."""

    for image_path, annotation_path in pairs:
        load_image(image_path)
        load_annotation(annotation_path)


def validate_segmentation_dataset(pairs: Sequence[tuple[Path, Path]]) -> None:
    """Validate segmentation masks and their class IDs."""

    validate_dataset_pairs(pairs)

    for image_path, annotation_path in pairs:
        image = load_image(image_path)
        image_height, image_width = image.shape[:2]
        mask = annotation_to_semantic_mask(annotation_path, image_height, image_width)

        if mask.shape != (image_height, image_width):
            raise ValueError(f"Shape mismatch for {image_path.name}: {mask.shape}")

        invalid_ids = [int(class_id) for class_id in np.unique(mask) if class_id != IGNORE_ID and not 0 <= class_id < NUM_CLASSES]

        if invalid_ids:
            raise ValueError(f"Invalid class IDs {invalid_ids} in {annotation_path}")


def validate_detection_dataset(pairs: Sequence[tuple[Path, Path]], bitmap_only: bool = False) -> None:
    """Validate detection annotations and extracted boxes."""

    validate_dataset_pairs(pairs)

    for image_path, annotation_path in pairs:
        image_height, image_width = load_image(image_path).shape[:2]
        annotation_to_instances(annotation_path, image_height, image_width, bitmap_only=bitmap_only)


def validate_dataset(pairs: Sequence[tuple[Path, Path]]) -> None:
    """Backward-compatible alias for segmentation dataset validation."""

    validate_segmentation_dataset(pairs)


def validate_project_dataset() -> None:
    """Validate the three dataset branches required by the project."""

    validate_detection_dataset(get_bounding_boxes_train_pairs())
    validate_segmentation_dataset(get_segmentation_train_pairs())
    validate_segmentation_dataset(get_segmentation_test_pairs())


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a semantic mask into an RGB image."""

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")

    unknown_ids = set(int(value) for value in np.unique(mask)) - set(MASK_COLORS)

    if unknown_ids:
        raise ValueError(f"Mask contains unknown class IDs: {sorted(unknown_ids)}")

    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for class_id, color in MASK_COLORS.items():
        color_mask[mask == class_id] = color

    return color_mask


def create_overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay non-background semantic labels on a BGR image."""

    _validate_image_array(image_bgr)

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in the range [0, 1].")
    if image_bgr.shape[:2] != mask.shape:
        raise ValueError(f"Shape mismatch: image={image_bgr.shape[:2]}, mask={mask.shape}")

    color_mask_bgr = cv2.cvtColor(colorize_mask(mask), cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, color_mask_bgr, alpha, 0.0)
    overlay = image_bgr.copy()
    annotated_pixels = mask != BACKGROUND_ID
    overlay[annotated_pixels] = blended[annotated_pixels]

    return overlay


# -----------------------------------------------------------------------------
# Segmentation evaluation: per-class IoU and cone-class mIoU
# -----------------------------------------------------------------------------

class SegmentationEvaluator:
    """Accumulate a dataset-level pixel confusion matrix and compute IoU."""

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_id: int = IGNORE_ID) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        """Clear all accumulated samples."""

        self.confusion_matrix.fill(0)

    def update(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> None:
        """Add one ground-truth/prediction mask pair."""

        if gt_mask.ndim != 2 or pred_mask.ndim != 2:
            raise ValueError("gt_mask and pred_mask must be 2D arrays.")
        if gt_mask.shape != pred_mask.shape:
            raise ValueError(f"Shape mismatch: gt={gt_mask.shape}, pred={pred_mask.shape}")

        gt_flat = gt_mask.reshape(-1).astype(np.int64, copy=False)
        pred_flat = pred_mask.reshape(-1).astype(np.int64, copy=False)
        valid = gt_flat != self.ignore_id
        gt_valid = gt_flat[valid]
        pred_valid = pred_flat[valid]

        if np.any((gt_valid < 0) | (gt_valid >= self.num_classes)):
            raise ValueError("Ground-truth mask contains invalid class IDs.")
        if np.any((pred_valid < 0) | (pred_valid >= self.num_classes)):
            raise ValueError("Prediction mask contains invalid class IDs.")

        indices = gt_valid * self.num_classes + pred_valid
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self) -> dict[int, float]:
        """Return dataset-level IoU for every class."""

        result: dict[int, float] = {}

        for class_id in range(self.num_classes):
            true_positive = int(self.confusion_matrix[class_id, class_id])
            false_positive = int(self.confusion_matrix[:, class_id].sum()) - true_positive
            false_negative = int(self.confusion_matrix[class_id, :].sum()) - true_positive
            denominator = true_positive + false_positive + false_negative
            result[class_id] = true_positive / denominator if denominator > 0 else float("nan")

        return result

    def mean_iou(self, class_ids: Sequence[int] = CONE_CLASS_IDS) -> float:
        """Return the unweighted mean IoU over the requested classes."""

        iou_per_class = self.per_class_iou()
        values = [iou_per_class[class_id] for class_id in class_ids if class_id in iou_per_class and not np.isnan(iou_per_class[class_id])]
        return float(np.mean(values)) if values else float("nan")

    def report(self) -> dict[str, Any]:
        """Return named per-class IoU values and cone-class mIoU."""

        iou_per_class = self.per_class_iou()
        return {
            "iou_per_class": {CLASS_ID_TO_NAME.get(class_id, str(class_id)): value for class_id, value in iou_per_class.items()},
            "mIoU": self.mean_iou(),
        }


# -----------------------------------------------------------------------------
# Shared box matching and classification evaluation
# -----------------------------------------------------------------------------

def box_iou(box_a: Box, box_b: Box) -> float:
    """Compute intersection over union for two half-open xyxy boxes."""

    x_min = max(box_a.x_min, box_b.x_min)
    y_min = max(box_a.y_min, box_b.y_min)
    x_max = min(box_a.x_max, box_b.x_max)
    y_max = min(box_a.y_max, box_b.y_max)
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    union = box_a.area + box_b.area - intersection
    return intersection / union if union > 0 else 0.0


def match_by_iou(gt_boxes: Sequence[Box], pred_boxes: Sequence[Box], iou_threshold: float = 0.5) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily match boxes one-to-one by descending IoU, ignoring class."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in the range [0, 1].")

    candidates = []

    for gt_index, gt_box in enumerate(gt_boxes):
        for pred_index, pred_box in enumerate(pred_boxes):
            iou = box_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int]] = []

    for _, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_index, pred_index))

    unmatched_gt = [index for index in range(len(gt_boxes)) if index not in matched_gt]
    unmatched_pred = [index for index in range(len(pred_boxes)) if index not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


class ClassificationEvaluator:
    """Compute macro F1 after geometry-only one-to-one box matching."""

    def __init__(self, class_ids: Sequence[int] = CONE_CLASS_IDS, iou_threshold: float = 0.5) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in the range [0, 1].")
        self.class_ids = tuple(class_ids)
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated counts."""

        self.true_positive = {class_id: 0 for class_id in self.class_ids}
        self.false_positive = {class_id: 0 for class_id in self.class_ids}
        self.false_negative = {class_id: 0 for class_id in self.class_ids}

    def update(self, gt_boxes: Sequence[Box], pred_boxes: Sequence[Box]) -> None:
        """Add one image worth of ground truth and predictions."""

        matches, unmatched_gt, unmatched_pred = match_by_iou(gt_boxes, pred_boxes, self.iou_threshold)

        for gt_index, pred_index in matches:
            gt_class = gt_boxes[gt_index].class_id
            pred_class = pred_boxes[pred_index].class_id
            if gt_class == pred_class:
                if gt_class in self.true_positive:
                    self.true_positive[gt_class] += 1
            else:
                if pred_class in self.false_positive:
                    self.false_positive[pred_class] += 1
                if gt_class in self.false_negative:
                    self.false_negative[gt_class] += 1

        for gt_index in unmatched_gt:
            gt_class = gt_boxes[gt_index].class_id
            if gt_class in self.false_negative:
                self.false_negative[gt_class] += 1

        for pred_index in unmatched_pred:
            pred_class = pred_boxes[pred_index].class_id
            if pred_class in self.false_positive:
                self.false_positive[pred_class] += 1

    def per_class_f1(self) -> dict[int, float]:
        """Return F1 for every cone class."""

        result: dict[int, float] = {}

        for class_id in self.class_ids:
            true_positive = self.true_positive[class_id]
            false_positive = self.false_positive[class_id]
            false_negative = self.false_negative[class_id]
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
            result[class_id] = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0

        return result

    def macro_f1(self) -> float:
        """Return the unweighted mean F1 across cone classes."""

        values = list(self.per_class_f1().values())
        return float(np.mean(values)) if values else float("nan")

    def report(self) -> dict[str, Any]:
        """Return named per-class F1 values and macro F1."""

        f1_per_class = self.per_class_f1()
        return {
            "f1_per_class": {CLASS_ID_TO_NAME[class_id]: f1_per_class[class_id] for class_id in self.class_ids},
            "macro_f1": self.macro_f1(),
        }


# -----------------------------------------------------------------------------
# Detection evaluation: COCO-style mAP@0.5:0.95
# -----------------------------------------------------------------------------

def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute COCO-style 101-point interpolated average precision."""

    if recall.ndim != 1 or precision.ndim != 1 or recall.shape != precision.shape:
        raise ValueError("recall and precision must be equally sized 1D arrays.")
    if recall.size == 0:
        return 0.0

    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    interpolated = np.zeros_like(recall_thresholds)

    for index, threshold in enumerate(recall_thresholds):
        valid = recall >= threshold
        interpolated[index] = np.max(precision_envelope[valid]) if np.any(valid) else 0.0

    return float(np.mean(interpolated))


class DetectionEvaluator:
    """Compute per-class AP and mAP over IoU thresholds 0.50 through 0.95."""

    IOU_THRESHOLDS = tuple(float(value) for value in np.round(np.arange(0.50, 1.00, 0.05), 2))

    def __init__(self, class_ids: Sequence[int] = CONE_CLASS_IDS) -> None:
        self.class_ids = tuple(class_ids)
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated images."""

        self._gt_by_image: list[list[Box]] = []
        self._pred_by_image: list[list[Box]] = []

    def update(self, gt_boxes: Sequence[Box], pred_boxes: Sequence[Box]) -> None:
        """Add one image worth of ground truth and scored predictions."""

        if any(box.score is not None for box in gt_boxes):
            raise ValueError("Ground-truth boxes must not have a confidence score.")
        if any(box.score is None for box in pred_boxes):
            raise ValueError("Every predicted box must have a confidence score.")

        self._gt_by_image.append(list(gt_boxes))
        self._pred_by_image.append(list(pred_boxes))

    def _average_precision_for_class(self, class_id: int, iou_threshold: float) -> float:
        gt_per_image = [[box for box in boxes if box.class_id == class_id] for boxes in self._gt_by_image]
        num_gt = sum(len(boxes) for boxes in gt_per_image)
        predictions: list[tuple[float, int, int, Box]] = []

        for image_index, boxes in enumerate(self._pred_by_image):
            for prediction_index, box in enumerate(boxes):
                if box.class_id == class_id:
                    predictions.append((float(box.score), image_index, prediction_index, box))

        if num_gt == 0:
            return float("nan") if not predictions else 0.0
        if not predictions:
            return 0.0

        predictions.sort(key=lambda item: (-item[0], item[1], item[2]))
        gt_used = [[False] * len(boxes) for boxes in gt_per_image]
        true_positive = np.zeros(len(predictions), dtype=np.float64)
        false_positive = np.zeros(len(predictions), dtype=np.float64)

        for rank, (_, image_index, _, pred_box) in enumerate(predictions):
            best_iou = -1.0
            best_gt_index = -1

            for gt_index, gt_box in enumerate(gt_per_image[image_index]):
                if gt_used[image_index][gt_index]:
                    continue
                iou = box_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            if best_gt_index >= 0 and best_iou >= iou_threshold:
                gt_used[image_index][best_gt_index] = True
                true_positive[rank] = 1.0
            else:
                false_positive[rank] = 1.0

        cumulative_tp = np.cumsum(true_positive)
        cumulative_fp = np.cumsum(false_positive)
        recall = cumulative_tp / float(num_gt)
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, np.finfo(np.float64).eps)
        return _average_precision(recall, precision)

    def per_class_ap(self) -> dict[int, float]:
        """Return AP@0.5:0.95 for every cone class."""

        result: dict[int, float] = {}

        for class_id in self.class_ids:
            values = [self._average_precision_for_class(class_id, threshold) for threshold in self.IOU_THRESHOLDS]
            finite_values = [value for value in values if not np.isnan(value)]
            result[class_id] = float(np.mean(finite_values)) if finite_values else float("nan")

        return result

    def mean_average_precision(self) -> float:
        """Return the unweighted mean AP across classes present in the ground truth."""

        values = [value for value in self.per_class_ap().values() if not np.isnan(value)]
        return float(np.mean(values)) if values else float("nan")

    def report(self) -> dict[str, Any]:
        """Return named per-class AP values and mAP@0.5:0.95."""

        ap_per_class = self.per_class_ap()
        return {
            "AP@0.5:0.95_per_class": {CLASS_ID_TO_NAME[class_id]: ap_per_class[class_id] for class_id in self.class_ids},
            "mAP@0.5:0.95": self.mean_average_precision(),
        }


# -----------------------------------------------------------------------------
# Internal validation helpers
# -----------------------------------------------------------------------------

def _validate_image_size(image_height: int, image_width: int) -> None:
    if not isinstance(image_height, int) or not isinstance(image_width, int):
        raise TypeError("image_height and image_width must be integers.")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be positive.")


def _validate_image_array(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D array.")
    if image.size == 0:
        raise ValueError("image must not be empty.")
