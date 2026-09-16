#Granati

import numpy as np
import time

import cv2
import numpy as np
import tensorflow as tf

import time

from timing.timing_utils import elapsed_ms
from detection.data_pipeline import build_inference_metadata, create_combined_views

from dataset.dataset_utils import Box, DETECTION_ID_TO_NAME, prediction_to_box
from detection.detection_config import (CONFIDENCE_THRESHOLD, GLOBAL_NMS_IOU_THRESHOLD, GLOBAL_CONTAINMENT_THRESHOLD, PATCH_BORDER_MARGIN,
                                        GLOBAL_CROSS_CONTAINMENT_THRESHOLD)


def predict_inference_dataset(infer, inference_ds, metadata) -> list[list[Box]]:
    outputs = {}

    for images in inference_ds:
        batch_predictions = infer(images)

        for name, values in batch_predictions.items():
            if name not in outputs:
                outputs[name] = []

            outputs[name].append(values)

    if not outputs:
        raise ValueError("Inference dataset is empty.")

    raw_predictions = {
        name: np.concatenate(parts, axis=0)
        for name, parts in outputs.items()
    }

    return postprocess_inference_dataset(raw_predictions, metadata)

def postprocess_inference_dataset(raw_predictions, metadata) -> list[list[Box]]:
    """
    Convert raw predictions from patch and full-image views into final
    detections grouped by original image.
    The function maps predictions back to original-image coordinates, applies patch-aware and final NMS,
    and returns one list of Box objects
    for each original image

    Returns:
        list: final detections grouped by original image.
    """
    if not metadata:
        return []

    num_views = len(metadata)
    num_predictions = len(raw_predictions["boxes"])

    # debug check: it's important that the number of predictions it's equal to the number of the metadata list
    # 1 prediction for each image = 1 metadata element
    if num_views != num_predictions:
        raise ValueError(
            f"Predictions/metadata mismatch: "
            f"{num_predictions} predictions for {num_views} views."
        )

    num_images = max(item["image_index"] for item in metadata) + 1

    # create list of list predictions
    patch_predictions = [[] for _ in range(num_images)]
    full_predictions = [[] for _ in range(num_images)]

    # Now we want to organized all the views w.r.t. images
    for view_index, view_metadata in enumerate(metadata):
        image_index = view_metadata["image_index"]

        image_width = view_metadata["image_width"]
        image_height = view_metadata["image_height"]

        if view_metadata["is_patch"]:
            # add all elems of a list inside another list in sequential order
            patch_predictions[image_index].extend(
                process_patch_view(raw_predictions, view_index, view_metadata)
            )

        else:
            full_predictions[image_index].extend(
                process_full_view(raw_predictions, view_index, view_metadata, image_width, image_height)
            )

    # NMS independently for every original image
    final_predictions = []

    for image_index in range(num_images):

        # All metadata belonging to the same image have
        # the same original image dimensions.
        # Let's consider only views belonging to image_index
        for item in metadata:
            if item["image_index"] == image_index:
                image_metadata = item
                break

        #now i can recover the image_size information of the image
        image_width = image_metadata["image_width"]
        image_height = image_metadata["image_height"]

        image_patch_predictions = global_nms_patch_aware(patch_predictions[image_index], image_width, image_height)
        image_final_predictions = final_nms(image_patch_predictions, full_predictions[image_index])

        final_predictions.append([
            prediction_to_box(prediction["bbox"], prediction["class_id"], prediction["score"],)
            for prediction in image_final_predictions
        ])

    return final_predictions


# Box Geometry
def compute_areas_and_intersection(box_a, box_b):
    """Compute the areas of two xyxy bounding boxes and their intersection area"""
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

    return intersection_area, area_a, area_b

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
    Compute how much of the smaller box is contained inside the other box.
    """

    intersection_area, area_a, area_b = compute_areas_and_intersection(box_a, box_b)
    smaller_area = min(area_a, area_b)

    if smaller_area <= 0:
        return 0.0

    return intersection_area / smaller_area

def is_near_internal_patch_border(prediction, image_width, image_height, margin=PATCH_BORDER_MARGIN):
    """
    Check whether a detection is close to an INTERNAL patch border.

    local_bbox is stored directly in source-patch coordinates, so the same
    logic works for native patches and letterboxed quadrants.

    Returns true if the detection is near an internal patch border, false otherwise.
    """

    x_min, y_min, x_max, y_max = prediction["local_bbox"]

    patch_x = prediction["patch_x"]
    patch_y = prediction["patch_y"]

    valid_width = prediction["patch_valid_width"]
    valid_height = prediction["patch_valid_height"]

    has_internal_left = patch_x > 0
    has_internal_top = patch_y > 0

    has_internal_right = (patch_x + valid_width < image_width)
    has_internal_bottom = (patch_y + valid_height < image_height)

    near_left = (has_internal_left and x_min <= margin)
    near_top = (has_internal_top and y_min <= margin)

    near_right = (has_internal_right and valid_width - x_max <= margin)
    near_bottom = (has_internal_bottom and valid_height - y_max <= margin)

    return (near_left or near_top or near_right or near_bottom)

def is_near_box(box_a, box_b, expansion_factor=0.30):
    """
    Check whether box_b is close to box_a.
    box_a is expanded by a fraction of its width/height.
    Useful for detecting border fragments that may have
    little or no IoU with the main detection.

    Returns True if box_b intersects the expanded region of box_a, False otherwise.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    width_a = ax2 - ax1
    height_a = ay2 - ay1

    if width_a <= 0 or height_a <= 0:
        return False

    # Expand box_a by a fraction of its width and height
    pad_x = width_a * expansion_factor
    pad_y = height_a * expansion_factor

    # Create an expanded box around box_a
    expanded_a = [ax1 - pad_x, ay1 - pad_y, ax2 + pad_x, ay2 + pad_y]

    ex1, ey1, ex2, ey2 = expanded_a

    # Does box_b intersect the expanded region?
    intersection_x1 = max(ex1, bx1)
    intersection_y1 = max(ey1, by1)
    intersection_x2 = min(ex2, bx2)
    intersection_y2 = min(ey2, by2)

    # Check if the intersection area is positive
    return (intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1)

# Post Processing
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
            prediction_near_border = is_near_internal_patch_border(prediction, image_width, image_height)

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

            # SPECIAL CASE:
            # same class, different patch
            # best is a reliable non-border detection,
            # prediction is a border fragment close to best.
            if (not best_near_border and prediction_near_border and is_near_box(best["bbox"], prediction["bbox"])):
                # Suppress border fragment
                continue

            # Cross-patch duplicate detection
            iou = calculate_iou(best["bbox"], prediction["bbox"])
            containment = calculate_containment(best["bbox"], prediction["bbox"])

            if ((iou < GLOBAL_NMS_IOU_THRESHOLD) and (containment < GLOBAL_CONTAINMENT_THRESHOLD)):
                remaining.append(prediction)

        predictions = remaining

    return selected

def clip_coordinates(image_height, image_width, x_min, y_min, x_max, y_max):
    """Clip bounding box coordinates to be within the image dimensions"""
    x_min = max(0.0, min(x_min, image_width))
    y_min = max(0.0, min(y_min, image_height))
    x_max = max(0.0, min(x_max, image_width))
    y_max = max(0.0, min(y_max, image_height))

    return x_min,y_min,x_max,y_max

def final_nms(patch_predictions, total_predictions):
    """
    Resolve patch-vs-total duplicates using greedy NMS

    Returns:
        list: Final predictions after NMS.
    """

    candidates = []

    for prediction in patch_predictions:
        candidates.append((prediction, "patch"))

    for prediction in total_predictions:
        candidates.append((prediction, "total"))

    candidates.sort(key=lambda x: x[0]["score"], reverse=True)
    selected = []

    for prediction, source in candidates:
        suppress = False

        for selected_prediction, selected_source in selected:
            # Patch-vs-patch and total-vs-total
            # have already been handled before.
            if source == selected_source:
                continue

            iou = calculate_iou(prediction["bbox"], selected_prediction["bbox"])
            containment = calculate_containment(prediction["bbox"], selected_prediction["bbox"])

            # Different classes are kept unless
            # containment is almost complete.
            if prediction["class_id"] != selected_prediction["class_id"]:
                if containment < GLOBAL_CROSS_CONTAINMENT_THRESHOLD:
                    continue

            if (iou < GLOBAL_NMS_IOU_THRESHOLD and containment < GLOBAL_CONTAINMENT_THRESHOLD):
                continue

            # selected_prediction has equal or higher confidence,
            # because candidates are sorted by score.
            suppress = True
            break

        if not suppress:
            selected.append((prediction, source))

    final_predictions = []

    for prediction, source in selected:
        final_predictions.append(prediction)

    return final_predictions

def get_view_detections(predictions, index):
    """
    Extracts detections for a specific view from the model's predictions.
    
    Returns:
        list of tuples: Each tuple contains (bbox, class_id, score) for a detection."""
    boxes = predictions["boxes"][index]
    classes = predictions["classes"][index]
    scores = predictions["confidence"][index]

    if "num_detections" in predictions:
        n = int(predictions["num_detections"][index])
        boxes, classes, scores = boxes[:n], classes[:n], scores[:n]

    result = []

    for bbox, class_id, score in zip(boxes, classes, scores):
        if class_id not in DETECTION_ID_TO_NAME or score < CONFIDENCE_THRESHOLD:
            continue

        result.append(([float(value) for value in bbox], class_id, score))

    return result

def process_patch_view(predictions, index, metadata):
    """
    Convert predictions from a patch view into source-image coordinates.
    
    Returns:
        list of dicts: Each dict contains metadata and coordinates for a detection in the source image."""
    result = []

    # Extract patch metadata
    offset_x = metadata["offset_x"]
    offset_y = metadata["offset_y"]

    valid_width = metadata["valid_width"]
    valid_height = metadata["valid_height"]

    # Extract source image dimensions and letterbox parameters
    source_width = metadata["source_width"]
    source_height = metadata["source_height"]

    scale_x = metadata["scale_x"]
    scale_y = metadata["scale_y"]
    pad_x = metadata["pad_x"]
    pad_y = metadata["pad_y"]

    for bbox, class_id, score in get_view_detections(predictions, index):
        x_min, y_min, x_max, y_max = bbox

        # First clip in the detector input space.
        x_min, y_min, x_max, y_max = clip_coordinates(valid_height, valid_width, x_min, y_min, x_max, y_max)

        if x_max <= x_min or y_max <= y_min:
            continue

        # Convert from letterboxed 800x800 coordinates back to the source patch/quadrant coordinates.
        source_x_min = (x_min - pad_x) / scale_x
        source_y_min = (y_min - pad_y) / scale_y
        source_x_max = (x_max - pad_x) / scale_x
        source_y_max = (y_max - pad_y) / scale_y

        # Clip the coordinates to ensure they are within the source image dimensions.
        source_x_min, source_y_min, source_x_max, source_y_max = clip_coordinates(source_height, source_width, source_x_min,
                                                                                  source_y_min, source_x_max, source_y_max)

        if source_x_max <= source_x_min or source_y_max <= source_y_min:
            continue

        # Store local_bbox in SOURCE-PATCH coordinates.
        # This keeps patch-border checks consistent even when letterbox is used.
        local_box = [source_x_min, source_y_min, source_x_max, source_y_max]

        result.append({
            "bbox": [source_x_min + offset_x, source_y_min + offset_y,
                     source_x_max + offset_x, source_y_max + offset_y],
            "local_bbox": local_box,
            "class_id": class_id,
            "score": score,
            "patch_index": index,
            "patch_x": offset_x,
            "patch_y": offset_y,
            "patch_valid_width": source_width,
            "patch_valid_height": source_height
        })

    return result

def process_full_view(predictions, index, metadata, image_width, image_height):
    """
    Convert predictions from a full-image view into original-image coordinates.
    
    Returns:
        list of dicts: Each dict contains metadata and coordinates for a detection in the original image
    """
    result = []

    # Extract letterbox parameters
    scale_x = metadata["scale_x"]
    scale_y = metadata["scale_y"]
    pad_x = metadata["pad_x"]
    pad_y = metadata["pad_y"]

    for bbox, class_id, score in get_view_detections(predictions, index):
        x_min, y_min, x_max, y_max = bbox

        x_min = (x_min - pad_x) / scale_x
        y_min = (y_min - pad_y) / scale_y
        x_max = (x_max - pad_x) / scale_x
        y_max = (y_max - pad_y) / scale_y

        # Clip the coordinates to ensure they are within the original image dimensions.
        x_min, y_min, x_max, y_max = clip_coordinates(image_height, image_width, x_min, y_min, x_max, y_max)

        if x_max <= x_min or y_max <= y_min:
            continue

        result.append({
            "bbox": [x_min, y_min, x_max, y_max],
            "class_id": class_id,
            "score": score,
        })

    return result

def build_detection_inputs(image_bgr):
    """Create yolo views and metadata for one frame"""
    image_height, image_width = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    views = create_combined_views(image_rgb)

    image_shapes = tf.constant([[image_height, image_width]], dtype=tf.int32)
    metadata = build_inference_metadata(image_shapes)

    if len(views) != len(metadata):
        raise ValueError(f"Views/metadata mismatch: {len(views)} views and {len(metadata)} metadata elements")

    inputs = np.ascontiguousarray(views, dtype=np.float32)

    return inputs, metadata, len(views)


def detect_frame(image_bgr, yolo_infer):
    """Run yolo detection on one frame"""
    timings = {}

    start_time = time.perf_counter()

    inputs, metadata, num_views = build_detection_inputs(image_bgr)

    timings["detection_preprocess"] = elapsed_ms(start_time)

    start_time = time.perf_counter()

    raw_predictions = yolo_infer(inputs)

    timings["yolo_inference"] = elapsed_ms(start_time)

    start_time = time.perf_counter()

    predictions = postprocess_inference_dataset(raw_predictions, metadata)

    timings["detection_postprocess"] = elapsed_ms(start_time)

    if len(predictions) != 1:
        raise ValueError(f"Expected one prediction group, got {len(predictions)}")

    return predictions[0], timings, num_views