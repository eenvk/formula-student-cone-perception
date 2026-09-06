import cv2
import numpy as np
import tensorflow as tf

from dataset.dataset_utils import (
    Box,
    DETECTION_ID_TO_NAME,
    prediction_to_box,
)

from detection.detection_config import IMAGE_SIZE


PATCH_STRIDE = 600
GLOBAL_NMS_IOU_THRESHOLD = 0.5
PATCH_OVERLAP = 0.20
PATCH_BATCH_SIZE = 5
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.5
GLOBAL_CONTAINMENT_THRESHOLD = 0.9

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


def calculate_iou(box_a, box_b):
    """
    Compute IoU between two xyxy bounding boxes.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area

def calculate_containment(box_a, box_b):
    """
    Compute how much of the smaller box is contained
    inside the other box.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    smaller_area = min(area_a, area_b)

    if smaller_area <= 0:
        return 0.0

    return intersection_area / smaller_area

def global_nms(predictions):
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

def predict_patch(patch_rgb, model, offset_x, offset_y, valid_width,valid_height):
    """
    Run YOLO on one patch and convert local bounding boxes
    to coordinates of the original image.

    Returns detections using YOLO class IDs (0..3).
    """

    patch_tensor = tf.convert_to_tensor(
        patch_rgb,
        dtype=tf.uint8
    )

    patch_tensor = tf.expand_dims(
        patch_tensor,
        axis=0
    )

    prediction = model.predict(
        patch_tensor,
        verbose=0
    )

    boxes = prediction["boxes"][0]
    classes = prediction["classes"][0]
    scores = prediction["confidence"][0]

    # Keep only valid detections if YOLO provides their number.
    if "num_detections" in prediction:
        num_detections = int(prediction["num_detections"][0])

        boxes = boxes[:num_detections]
        classes = classes[:num_detections]
        scores = scores[:num_detections]

    results = []

    for bbox, class_id, score in zip(
        boxes,
        classes,
        scores
    ):

        class_id = int(class_id)
        score = float(score)

        if class_id not in DETECTION_ID_TO_NAME:
            continue

        x_min, y_min, x_max, y_max = [
            float(value)
            for value in bbox
        ]

        x_min = max(0.0, min(x_min, valid_width))
        y_min = max(0.0, min(y_min, valid_height))
        x_max = max(0.0, min(x_max, valid_width))
        y_max = max(0.0, min(y_max, valid_height))

        if x_max <= x_min or y_max <= y_min:
            continue

        global_box = [
            x_min + offset_x,
            y_min + offset_y,
            x_max + offset_x,
            y_max + offset_y
        ]

        results.append({
            "bbox": global_box,
            "class_id": class_id,
            "score": score
        })

    return results


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
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    prediction = predict_image_with_patches(
        model,
        image_rgb
    )

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

        result.append(box)

    return result

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

            class_id = class_id
            score = score

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
                "class_id": class_id,
                "score": score
            })

    # 5. GLOBAL NMS
    # Reuse the already tested class-aware NMS.
    predictions = global_nms(all_predictions)

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