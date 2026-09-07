# tino

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from dataset.dataset_utils import (
    Box,
    DETECTION_ID_TO_NAME,
    prediction_to_box,
)


from detection.detection_config import (
    IMAGE_SIZE,
    PATCH_STRIDE,
    PATCH_BATCH_SIZE,
    CONFIDENCE_THRESHOLD,
    GLOBAL_NMS_IOU_THRESHOLD,
    GLOBAL_CONTAINMENT_THRESHOLD,
    PATCH_BORDER_MARGIN,
    GLOBAL_CROSS_CONTAINMENT_THRESHOLD
)

PATCH_COLORS = [
    (255, 0, 0),       # red
    (0, 255, 0),       # green
    (0, 0, 255),       # blue
    (255, 255, 0),     # yellow
    (255, 0, 255),     # magenta
    (0, 255, 255),     # cyan
    (255, 128, 0),     # orange
    (128, 0, 255),     # purple
    (0, 128, 255),
    (255, 0, 128),
    (128, 255, 0),
    (0, 255, 128),
    (128, 128, 255),
    (255, 128, 128),
    (128, 255, 255),
    (255, 255, 128),
]

NO_NMS_OUTPUT_DIR = Path("inference_No_nms")
NO_NMS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_no_nms_image_counter = 0

def save_predictions_before_nms(image_rgb, predictions):
    """
    Save one visualization containing every prediction BEFORE global NMS.

    Images are saved sequentially inside inference_No_nms/.
    """

    global _no_nms_image_counter

    visualization = image_rgb.copy()
    image_height, image_width = visualization.shape[:2]

    for prediction in predictions:
        x_min, y_min, x_max, y_max = prediction["bbox"]
        class_id = int(prediction["class_id"])
        score = float(prediction["score"])
        patch_index = prediction["patch_index"]


        # Same patch -> same color
        color = PATCH_COLORS[
            patch_index % len(PATCH_COLORS)
        ]

        x_min = int(round(max(0.0, min(x_min, image_width - 1))))
        y_min = int(round(max(0.0, min(y_min, image_height - 1))))
        x_max = int(round(max(0.0, min(x_max, image_width - 1))))
        y_max = int(round(max(0.0, min(y_max, image_height - 1))))

        if x_max <= x_min or y_max <= y_min:
            continue

        class_name = DETECTION_ID_TO_NAME.get(
            class_id,
            str(class_id)
        )

        near_border = is_near_internal_patch_border(
            prediction,
            image_width,
            image_height
        )

        border_text = " BORDER" if near_border else ""

        label = (
            f"P{patch_index} "
            f"{class_name} "
            f"{score:.2f}"
            f"{border_text}"
        )

        cv2.rectangle(
            visualization,
            (x_min, y_min),
            (x_max, y_max),
            color,
            2
        )

        text_y = max(15, y_min - 5)

        cv2.putText(
            visualization,
            label,
            (x_min, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    # The image is RGB here, while cv2.imwrite expects BGR.
    visualization_bgr = cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )

    output_file = (
        NO_NMS_OUTPUT_DIR /
        f"inference_no_nms_{_no_nms_image_counter:05d}.jpg"
    )

    cv2.imwrite(
        str(output_file),
        visualization_bgr
    )

    _no_nms_image_counter += 1


# Patch extraction
def get_patch_starts(image_size, patch_size, stride):
    """
    Return the starting coordinates of the patches along one axis.

    Example with:
        image_size = 2048
        patch_size = 800
        stride = 600

    returns:
        [0, 600, 1200, 1800]

    The last patch can be smaller than 800 pixels and will
    later be padded.
    """

    if stride <= 0:
        raise ValueError("stride must be greater than 0.")

    if stride >= patch_size:
        raise ValueError(
            "stride must be smaller than patch_size "
            "to guarantee overlapping patches."
        )

    starts = [0]

    while starts[-1] + patch_size < image_size:
        starts.append(starts[-1] + stride)

    return starts

def extract_patch(image_rgb, x_start, y_start, patch_width, patch_height):
    """
    Extract a patch from the original image.

    If the patch reaches the image boundary and is smaller than
    the required size, black padding is added on the right/bottom.

    The actual image content is NOT resized.
    """

    image_height, image_width = image_rgb.shape[:2]

    x_end = min(x_start + patch_width, image_width)
    y_end = min(y_start + patch_height, image_height)

    crop = image_rgb[y_start:y_end, x_start:x_end]

    valid_height, valid_width = crop.shape[:2]

    patch = np.zeros(
        (patch_height, patch_width, 3),
        dtype=image_rgb.dtype
    )

    patch[:valid_height, :valid_width] = crop

    return patch, valid_width, valid_height

# Box Geometry
def compute_areas_and_intersection(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    
    intersection_area = (intersection_width * intersection_height)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    return intersection_area,area_a,area_b

def calculate_iou(box_a, box_b):
    """
    Compute IoU between two xyxy bounding boxes.
    """

    intersection_area, area_a, area_b = compute_areas_and_intersection(box_a, box_b)
    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area

def calculate_containment(box_a, box_b):
    """
    Compute how much of the smaller box is contained
    inside the other box.
    """

    intersection_area, area_a, area_b = compute_areas_and_intersection(box_a, box_b)
    smaller_area = min(area_a, area_b)

    if smaller_area <= 0:
        return 0.0

    return intersection_area / smaller_area

def is_near_internal_patch_border(prediction, image_width, image_height, margin=PATCH_BORDER_MARGIN):
    """
    Check whether a detection is close to an INTERNAL patch border.

    Borders that coincide with the real image boundary are not
    considered problematic.
    """

    x_min, y_min, x_max, y_max = prediction["local_bbox"]

    patch_x = prediction["patch_x"]
    patch_y = prediction["patch_y"]

    valid_width = prediction["patch_valid_width"]
    valid_height = prediction["patch_valid_height"]

    # Determine whether each patch border is internal
    # or coincides with the real image border.
    has_internal_left = patch_x > 0
    has_internal_top = patch_y > 0

    has_internal_right = (patch_x + valid_width < image_width)
    has_internal_bottom = (patch_y + valid_height < image_height)

    near_left = (has_internal_left and x_min <= margin)
    near_top = (has_internal_top and y_min <= margin)

    near_right = (has_internal_right and valid_width - x_max <= margin)
    near_bottom = (has_internal_bottom and valid_height - y_max <= margin)

    return (near_left or near_top or near_right or near_bottom)

def distance_from_patch_border(prediction):

    x_min, y_min, x_max, y_max = prediction["local_bbox"]

    valid_width = prediction["patch_valid_width"]
    valid_height = prediction["patch_valid_height"]

    return min(
        x_min,
        y_min,
        valid_width - x_max,
        valid_height - y_max
    )

def is_near_box(box_a, box_b, expansion_factor=0.30):
    """
    Check whether box_b is close to box_a.

    box_a is expanded by a fraction of its width/height.
    Useful for detecting border fragments that may have
    little or no IoU with the main detection.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    width_a = ax2 - ax1
    height_a = ay2 - ay1

    if width_a <= 0 or height_a <= 0:
        return False

    pad_x = width_a * expansion_factor
    pad_y = height_a * expansion_factor

    expanded_a = [
        ax1 - pad_x,
        ay1 - pad_y,
        ax2 + pad_x,
        ay2 + pad_y,
    ]

    ex1, ey1, ex2, ey2 = expanded_a

    # Does box_b intersect the expanded region?
    intersection_x1 = max(ex1, bx1)
    intersection_y1 = max(ey1, by1)
    intersection_x2 = min(ex2, bx2)
    intersection_y2 = min(ey2, by2)

    return (
        intersection_x2 > intersection_x1
        and intersection_y2 > intersection_y1
    )

# Post Processing
def global_nms_confidence(predictions):
    """
    Remove duplicate detections produced by overlapping patches.

    NMS is class-aware:
    two boxes are considered duplicates only if:
        - they belong to the same class
        - their IoU is sufficiently high

    Among duplicate boxes, the one with the highest confidence
    is kept.
    """

    predictions = sorted(
        predictions,
        key=lambda prediction: prediction["score"],
        reverse=True
    )

    selected = []

    while predictions:

        best = predictions.pop(0)
        selected.append(best)

        remaining = []

        for prediction in predictions:

            # Different classes are not considered duplicates.
            if prediction["class_id"] != best["class_id"]:
                remaining.append(prediction)
                continue

            iou =           calculate_iou(best["bbox"], prediction["bbox"])
            containment =   calculate_containment(best["bbox"], prediction["bbox"])

            if iou < GLOBAL_NMS_IOU_THRESHOLD and containment < GLOBAL_CONTAINMENT_THRESHOLD:
                remaining.append(prediction)

        predictions = remaining

    return selected

def global_nms_patch_aware(predictions, image_width, image_height):
    """
    Remove duplicate detections produced by different patches.

    Predictions coming from the same patch are not suppressed
    by the global NMS.

    When duplicate detections come from different patches,
    detections away from internal patch borders are preferred.
    Confidence is used as a secondary criterion.
    """

    def sorting_key(prediction):
        near_border = is_near_internal_patch_border(prediction, image_width, image_height,)

        # First prefer boxes not close to internal borders,
        # then prefer higher confidence.
        return (not near_border, prediction["score"])

    predictions = sorted(predictions, key=sorting_key, reverse=True)

    selected = []

    while predictions:

        best = predictions.pop(0)
        selected.append(best)

        remaining = []

        best_near_border = is_near_internal_patch_border(best, image_width, image_height)


        for prediction in predictions:

            prediction_near_border = is_near_internal_patch_border(
                prediction,
                image_width,
                image_height
            )

            # Same patch:
            # do NOT use global NMS to suppress them.
            if (prediction["patch_index"] == best["patch_index"]):
                remaining.append(prediction)
                continue

            # Different classes are currently kept separate.
            if (prediction["class_id"] != best["class_id"]):
                containment = calculate_containment(best["bbox"], prediction["bbox"])
                if (containment < GLOBAL_CROSS_CONTAINMENT_THRESHOLD):
                    remaining.append(prediction)
                    continue

            # --------------------------------------------------
            # SPECIAL CASE:
            # same class, different patch
            #
            # best is a reliable non-border detection,
            # prediction is a border fragment close to best.
            # --------------------------------------------------
            if (
                not best_near_border
                and prediction_near_border
                and is_near_box(
                    best["bbox"],
                    prediction["bbox"]
                )
            ):
                # Suppress border fragment
                continue


            # Cross-patch duplicate detection
            iou = calculate_iou(best["bbox"], prediction["bbox"])
            containment = calculate_containment(best["bbox"], prediction["bbox"])

            if ((iou < GLOBAL_NMS_IOU_THRESHOLD) and (containment < GLOBAL_CONTAINMENT_THRESHOLD)):
                remaining.append(prediction)

        predictions = remaining

    return selected    

# Patch-based inference
def predict_image_with_patches(model, image, patch_size=IMAGE_SIZE):
    """
    Runs patch-based inference on an original-resolution RGB image.

    Patch extraction and global NMS use the same logic already used
    by the tested inference pipeline, while model inference is
    performed in batches for better efficiency.

    Returns predictions using detection class IDs 0..3.
    """

    # Validation images are TensorFlow tensors.
    # extract_patch() already works with NumPy arrays.
    if tf.is_tensor(image):
        image = image.numpy()

    image_height, image_width = image.shape[:2]
    patch_height, patch_width = patch_size

    # starting position of each patch
    x_starts = get_patch_starts(
        image_width,
        patch_width,
        PATCH_STRIDE
    )

    y_starts = get_patch_starts(
        image_height,
        patch_height,
        PATCH_STRIDE
    )

    # extract the patches
    patches = []
    patch_metadata = []

    for y_start in y_starts:
        for x_start in x_starts:

            patch, valid_width, valid_height = extract_patch(
                image,
                x_start,
                y_start,
                patch_width,
                patch_height
            )

            patches.append(patch)

            # Store the information needed to convert
            # local coordinates to global coordinates.
            patch_metadata.append(
                (
                    x_start,
                    y_start,
                    valid_width,
                    valid_height
                )
            )

    # All patches have the same size because extract_patch()
    # adds padding when necessary.
    patches = np.stack(patches, axis=0)

    # Batch inference

    predictions = model.predict(
        patches,
        batch_size=PATCH_BATCH_SIZE,
        # verbose=0
    )

    # Move from local predictions to global
    all_predictions = []

    for patch_index, metadata in enumerate(patch_metadata):

        offset_x = metadata[0]
        offset_y = metadata[1]
        valid_width = metadata[2]
        valid_height = metadata[3]

        boxes = predictions["boxes"][patch_index]
        classes = predictions["classes"][patch_index]
        scores = predictions["confidence"][patch_index]

        # If available, keep only actual detections.
        if "num_detections" in predictions:
            num_detections = int(predictions["num_detections"][patch_index])

            boxes = boxes[:num_detections]
            classes = classes[:num_detections]
            scores = scores[:num_detections]

        for bbox, class_id, score in zip(boxes, classes,scores):
            # Remove invalid classes.
            if class_id not in DETECTION_ID_TO_NAME:
                continue

            # Remove low-confidence detections.
            if score < CONFIDENCE_THRESHOLD:
                continue

            x_min, y_min, x_max, y_max = [
                float(value)
                for value in bbox
            ]

            # Clip to the real part of the patch.
            # This removes predictions that fall inside padding.
            x_min = max(0.0, min(x_min, valid_width))
            y_min = max(0.0, min(y_min, valid_height))
            x_max = max(0.0, min(x_max, valid_width))
            y_max = max(0.0, min(y_max, valid_height))

            if x_max <= x_min or y_max <= y_min:
                continue

            local_box = [
                x_min,
                y_min,
                x_max,
                y_max
            ]

            # Convert patch coordinates to coordinates
            # of the original image.
            global_box = [
                x_min + offset_x,
                y_min + offset_y,
                x_max + offset_x,
                y_max + offset_y
            ]

            all_predictions.append({
                "bbox": global_box,
                "local_bbox": local_box,
                "class_id": int(class_id),
                "score": float(score),

                # Patch information
                "patch_index": patch_index,
                "patch_x": offset_x,
                "patch_y": offset_y,
                "patch_valid_width": valid_width,
                "patch_valid_height": valid_height,
            })

    save_predictions_before_nms(image, all_predictions)

    # 5. GLOBAL NMS
    # Reuse the already tested class-aware NMS.
    # predictions = global_nms_confidence(all_predictions)
    predictions = global_nms_patch_aware(all_predictions, image_width, image_height)

    # 6. FORMAT RESULT FOR KERASCV METRICS
    final_boxes = []
    final_classes = []
    final_scores = []

    for prediction in predictions:
        x_min, y_min, x_max, y_max = prediction["bbox"]

        # Final safety clipping to original image dimensions.
        x_min = max(0.0, min(x_min, image_width))
        y_min = max(0.0, min(y_min, image_height))
        x_max = max(0.0, min(x_max, image_width))
        y_max = max(0.0, min(y_max, image_height))

        if x_max <= x_min or y_max <= y_min:
            continue

        final_boxes.append([x_min, y_min, x_max, y_max])
        final_classes.append(prediction["class_id"])
        final_scores.append(prediction["score"])

    # No detections.
    if len(final_boxes) == 0:
        return {
            "boxes": tf.zeros((0, 4), dtype=tf.float32),
            "classes": tf.zeros((0,), dtype=tf.float32),
            "confidence": tf.zeros((0,), dtype=tf.float32)
        }

    return {
        "boxes": tf.constant(
            final_boxes,
            dtype=tf.float32
        ),
        "classes": tf.constant(
            final_classes,
            dtype=tf.float32
        ),
        "confidence": tf.constant(
            final_scores,
            dtype=tf.float32
        ),
    }

# wrapper
def predict_boxes(image_bgr, model) -> list[Box]:
    """
    Run patch-based YOLO inference and return predictions
    using the shared Box representation.
    """

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a NumPy array.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            "image_bgr must have shape [height, width, 3]."
        )

    # Test images are loaded with OpenCV -> BGR.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    prediction = predict_image_with_patches(model, image_rgb)

    result = []

    for bbox, class_id, score in zip(
        prediction["boxes"],
        prediction["classes"],
        prediction["confidence"]
    ):

        box = prediction_to_box(
            bbox.numpy().tolist(),
            int(class_id),
            float(score)
        )

    # No detections.
    if len(final_boxes) == 0:
        return {
            "boxes": tf.zeros((0, 4), dtype=tf.float32),
            "classes": tf.zeros((0,), dtype=tf.float32),
            "confidence": tf.zeros((0,), dtype=tf.float32)
        }

    return result

