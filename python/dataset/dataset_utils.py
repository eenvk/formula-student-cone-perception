#renzi

from pathlib import Path
import json
import random

import cv2
import numpy as np

import base64
import zlib


#classes

BACKGROUND_ID = 0
YELLOW_CONE_ID = 1
BLUE_CONE_ID = 2
SMALL_ORANGE_CONE_ID = 3
BIG_ORANGE_CONE_ID = 4

IGNORE_ID = 255

NUM_CLASSES = 5


CLASS_ID_TO_NAME = {
    0: "background",
    1: "yellow_cone",
    2: "blue_cone",
    3: "small_orange_cone",
    4: "big_orange_cone",
}


CLASS_NAME_TO_ID = {
    "seg_yellow_cone": YELLOW_CONE_ID,
    "seg_blue_cone": BLUE_CONE_ID,
    "seg_orange_cone": SMALL_ORANGE_CONE_ID,
    "seg_large_orange_cone": BIG_ORANGE_CONE_ID,
    "seg_unknown_cone": IGNORE_ID,
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# class utilities

def class_name_to_id(class_name: str) -> int:

    normalized_name = class_name.strip().lower()

    if normalized_name not in CLASS_NAME_TO_ID:
        raise ValueError(
            f"Unknown class: {class_name}"
        )

    return CLASS_NAME_TO_ID[normalized_name]



# IMAGE / ANNOTATION PAIRS

def find_dataset_pairs(
        image_dir: str | Path,
        annotation_dir: str | Path
) -> list[tuple[Path, Path]]:

    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)

    images = {}

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images[image_path.name] = image_path

    pairs = []

    for annotation_path in annotation_dir.glob("*.json"):

        # Esempio:
        # amz_00851.png.json
        #          ↓ rimuovo solo ".json"
        # amz_00851.png
        image_name = annotation_path.name.removesuffix(".json")

        if image_name not in images:
            print(
                f"[WARNING] Missing image for "
                f"{annotation_path.name}"
            )
            continue

        pairs.append(
            (
                images[image_name],
                annotation_path
            )
        )

    pairs.sort(key=lambda pair: pair[0].name)

    return pairs

#
def find_all_dataset_pairs(
        dataset_root: str | Path
) -> list[tuple[Path, Path]]:
    """
    Find image-annotation pairs across all FSOCO sub-datasets.
    """

    dataset_root = Path(dataset_root)

    all_pairs = []

    for subset_dir in dataset_root.iterdir():

        if not subset_dir.is_dir():
            continue

        image_dir = subset_dir / "img"
        annotation_dir = subset_dir / "ann"

        if not image_dir.is_dir() or not annotation_dir.is_dir():
            print(
                f"[WARNING] Skipping {subset_dir.name}: "
                f"missing img or ann directory."
            )
            continue

        subset_pairs = find_dataset_pairs(
            image_dir,
            annotation_dir
        )

        all_pairs.extend(subset_pairs)

    all_pairs.sort(
        key=lambda pair: str(pair[0])
    )

    return all_pairs
#
def decode_bitmap(bitmap_data: str) -> np.ndarray:
    """
    Decode a Supervisely bitmap annotation into a binary mask.

    Returns
    -------
    binary_mask:
        Boolean array with shape (H, W).
        True pixels belong to the annotated object.
    """

    compressed_data = base64.b64decode(bitmap_data)
    image_data = zlib.decompress(compressed_data)

    image_array = np.frombuffer(
        image_data,
        dtype=np.uint8
    )

    decoded_image = cv2.imdecode(
        image_array,
        cv2.IMREAD_UNCHANGED
    )

    if decoded_image is None:
        raise ValueError(
            "Could not decode bitmap annotation."
        )

    # Supervisely bitmaps normally store the object mask
    # in the alpha channel.
    if decoded_image.ndim == 3 and decoded_image.shape[2] == 4:
        binary_mask = decoded_image[:, :, 3] > 0

    elif decoded_image.ndim == 2:
        binary_mask = decoded_image > 0

    else:
        raise ValueError(
            f"Unexpected decoded bitmap shape: "
            f"{decoded_image.shape}"
        )

    return binary_mask


# JSON -> SEMANTIC MASK

def annotation_to_semantic_mask(
        annotation_path: str | Path,
        image_height: int,
        image_width: int
) -> np.ndarray:
    """
    Convert a Supervisely JSON annotation into a semantic mask.

    Mask values:
        0   = background
        1   = yellow cone
        2   = blue cone
        3   = small orange cone
        4   = big orange cone
        255 = unknown cone / ignored pixel
    """

    annotation_path = Path(annotation_path)

    with annotation_path.open(
            "r",
            encoding="utf-8"
    ) as file:
        annotation = json.load(file)

    semantic_mask = np.zeros(
        (image_height, image_width),
        dtype=np.uint8
    )

    for obj in annotation.get("objects", []):

        class_name = obj.get("classTitle")

        if class_name is None:
            continue

        class_id = class_name_to_id(
            class_name
        )

        if obj.get("geometryType") != "bitmap":
            continue

        bitmap = obj.get("bitmap")

        if bitmap is None:
            continue

        bitmap_data = bitmap.get("data")
        origin = bitmap.get("origin")

        if bitmap_data is None or origin is None:
            continue

        object_mask = decode_bitmap(
            bitmap_data
        )

        origin_x = int(origin[0])
        origin_y = int(origin[1])

        object_height, object_width = object_mask.shape

        # Compute the region occupied by the bitmap
        # inside the original image.
        x_start = max(0, origin_x)
        y_start = max(0, origin_y)

        x_end = min(
            image_width,
            origin_x + object_width
        )

        y_end = min(
            image_height,
            origin_y + object_height
        )

        if x_start >= x_end or y_start >= y_end:
            continue

        # Compute the corresponding region
        # inside the decoded bitmap.
        bitmap_x_start = x_start - origin_x
        bitmap_y_start = y_start - origin_y

        bitmap_x_end = bitmap_x_start + (
                x_end - x_start
        )

        bitmap_y_end = bitmap_y_start + (
                y_end - y_start
        )

        object_region = object_mask[
                        bitmap_y_start:bitmap_y_end,
                        bitmap_x_start:bitmap_x_end
                        ]

        semantic_region = semantic_mask[
                          y_start:y_end,
                          x_start:x_end
                          ]

        semantic_region[object_region] = class_id

    return semantic_mask

# TRAIN / VALIDATION SPLIT

def train_validation_split(
        pairs: list[tuple[Path, Path]],
        validation_fraction: float = 0.20,
        seed: int = 42
):

    shuffled_pairs = list(pairs)

    rng = random.Random(seed)
    rng.shuffle(shuffled_pairs)

    validation_size = round(
        len(shuffled_pairs)
        * validation_fraction
    )

    validation_pairs = (
        shuffled_pairs[:validation_size]
    )

    training_pairs = (
        shuffled_pairs[validation_size:]
    )

    return training_pairs, validation_pairs



# RESIZE


def resize_mask(
        mask: np.ndarray,
        width: int,
        height: int
) -> np.ndarray:

    return cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )



# VALIDATION


def validate_dataset(
        pairs: list[tuple[Path, Path]]
) -> None:

    for index, (
            image_path,
            annotation_path
    ) in enumerate(pairs):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read {image_path}"
            )

        height, width = image.shape[:2]

        mask = annotation_to_semantic_mask(
            annotation_path,
            height,
            width
        )

        if mask.shape != (height, width):

            raise ValueError(
                f"Shape mismatch for "
                f"{image_path.name}"
            )

        ids = np.unique(mask)

        for class_id in ids:
            if class_id != IGNORE_ID and not 0 <= class_id < NUM_CLASSES:
                raise ValueError(
                    f"Invalid class ID {class_id}"
                )

    print(
        f"Dataset validation completed: "
        f"{len(pairs)} samples OK."
    )



MASK_COLORS = {
    0: (0, 0, 0),        # Background
    1: (255, 255, 0),    # Yellow cone
    2: (0, 0, 255),      # Blue cone
    3: (255, 165, 0),    # Small orange cone
    4: (255, 0, 255),    # Big orange cone
    255: (255, 255, 255) # Unknown cone
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:

    #Convert a semantic mask into an RGB image for visualization.


    height, width = mask.shape

    color_mask = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for class_id, color in MASK_COLORS.items():
        color_mask[mask == class_id] = color

    return color_mask

def create_overlay(
        image_bgr: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.45
) -> np.ndarray:
    """
    Overlay the semantic mask on the original image.
    """

    if image_bgr.shape[:2] != mask.shape:
        raise ValueError(
            f"Shape mismatch: image={image_bgr.shape[:2]}, "
            f"mask={mask.shape}"
        )

    color_mask_rgb = colorize_mask(mask)

    color_mask_bgr = cv2.cvtColor(
        color_mask_rgb,
        cv2.COLOR_RGB2BGR
    )

    blended = cv2.addWeighted(
        image_bgr,
        1.0 - alpha,
        color_mask_bgr,
        alpha,
        0.0
    )

    overlay = image_bgr.copy()

    # Show only annotated pixels in the overlay.
    annotated_pixels = mask != BACKGROUND_ID
    overlay[annotated_pixels] = blended[annotated_pixels]

    return overlay