import cv2
import keras_cv
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

            iou = calculate_iou(
                best["bbox"],
                prediction["bbox"]
            )

            if iou < GLOBAL_NMS_IOU_THRESHOLD:
                remaining.append(prediction)

        predictions = remaining

    return selected


def predict_patch(
    patch_rgb,
    model,
    offset_x,
    offset_y,
    valid_width,
    valid_height
):
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

    prediction = keras_cv.bounding_box.to_ragged(
        prediction
    )

    boxes = prediction["boxes"][0].numpy()
    classes = prediction["classes"][0].numpy()
    scores = prediction["confidence"][0].numpy()

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

        # A boundary patch may contain black padding.
        # Predictions must therefore be clipped to the real
        # part of the image.
        x_min = max(0.0, min(x_min, valid_width))
        y_min = max(0.0, min(y_min, valid_height))
        x_max = max(0.0, min(x_max, valid_width))
        y_max = max(0.0, min(y_max, valid_height))

        if x_max <= x_min or y_max <= y_min:
            continue

        # Local patch coordinates -> original image coordinates.
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
    Run tiled YOLO inference on an original BGR image.

    The image is divided into overlapping patches.
    Predictions are converted back to original-image coordinates
    and duplicate detections are removed through global NMS.

    Args:
        image_bgr:
            Original OpenCV BGR image.

        model:
            Trained YOLO detector.

    Returns:
        list[Box]:
            Bounding boxes expressed in coordinates of the
            ORIGINAL image.

            Class IDs use the shared convention:
                1 = yellow
                2 = blue
                3 = small orange
                4 = big orange

            Each Box also contains the prediction confidence.
    """

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a NumPy array.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            "image_bgr must have shape [height, width, 3]."
        )

    image_height, image_width = image_bgr.shape[:2]

    # OpenCV uses BGR, while YOLO was trained with RGB images.
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    patch_height = IMAGE_SIZE[0]
    patch_width = IMAGE_SIZE[1]

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

    all_predictions = []

    for y_start in y_starts:

        for x_start in x_starts:

            patch, valid_width, valid_height = extract_patch(
                image_rgb,
                x_start,
                y_start,
                patch_width,
                patch_height
            )

            patch_predictions = predict_patch(
                patch,
                model,
                x_start,
                y_start,
                valid_width,
                valid_height
            )

            all_predictions.extend(
                patch_predictions
            )

    # The same cone can be detected by multiple overlapping patches.
    predictions = global_nms(
        all_predictions
    )

    result = []

    for prediction in predictions:

        x_min, y_min, x_max, y_max = prediction["bbox"]

        # Final clipping to original image boundaries.
        x_min = max(0.0, min(x_min, image_width))
        y_min = max(0.0, min(y_min, image_height))
        x_max = max(0.0, min(x_max, image_width))
        y_max = max(0.0, min(y_max, image_height))

        x_min = int(round(x_min))
        y_min = int(round(y_min))
        x_max = int(round(x_max))
        y_max = int(round(y_max))

        if x_max <= x_min or y_max <= y_min:
            continue

        # prediction_to_box() performs the YOLO 0..3
        # -> shared 1..4 class conversion and stores the score.
        box = prediction_to_box(
            [x_min, y_min, x_max, y_max],
            prediction["class_id"],
            prediction["score"]
        )

        result.append(box)

        print(result)

    return result