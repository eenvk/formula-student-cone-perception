#renzi

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
# Project and dataset paths
# -----------------------------------------------------------------------------

PYTHON_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"

BOUNDING_BOXES_TRAIN_ROOT = DATASET_ROOT / "fsoco_bounding_boxes_train"

SEGMENTATION_TRAIN_ROOT = DATASET_ROOT / "fsoco_segmentation_train"

SEGMENTATION_TEST_ROOT = DATASET_ROOT / "test_set" / "segmentation_test"
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

@dataclass(frozen=True)
class SegmentationInstance:
    """One cone instance with its class, bounding box, and binary mask."""

    class_id: int
    bbox: Box
    mask: np.ndarray

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
    """Return pairs from dataset/fsoco_bounding_boxes_train."""
    return find_all_dataset_pairs(BOUNDING_BOXES_TRAIN_ROOT)


def get_segmentation_train_pairs() -> list[tuple[Path, Path]]:
    """Return pairs from dataset/fsoco_segmentation_train."""
    return find_all_dataset_pairs(SEGMENTATION_TRAIN_ROOT)


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


def annotation_to_segmentation_instances(annotation_path: str | Path, image_height: int, image_width: int) -> list[SegmentationInstance]:
    """Extract cone instances with their individual binary masks and bounding boxes."""

    _validate_image_size(image_height, image_width)
    annotation = load_annotation(annotation_path)
    instances: list[SegmentationInstance] = []

    for obj in annotation.get("objects", []):
        if not isinstance(obj, dict):
            continue

        if obj.get("geometryType") != "bitmap":
            continue

        class_name = obj.get("classTitle")

        if class_name is None:
            continue

        class_id = class_name_to_id(class_name)

        if class_id == IGNORE_ID:
            continue

        binary_mask = bitmap_object_to_full_mask(obj, image_height, image_width)
        coordinates = binary_mask_to_bbox(binary_mask)

        if coordinates is None:
            continue

        bbox = Box(*coordinates, class_id)
        instances.append(SegmentationInstance(class_id=class_id, bbox=bbox, mask=binary_mask))

    return instances


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
