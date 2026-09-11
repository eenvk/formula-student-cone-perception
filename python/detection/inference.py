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

def create_patches(image_rgb, patch_size=IMAGE_SIZE):
    image = image_rgb.numpy() if tf.is_tensor(image_rgb) else image_rgb

    image_height, image_width = image_rgb.shape[:2]
    patch_height, patch_width = patch_size

        # starting position of each patch
    x_starts = get_patch_starts(image_width, patch_width, PATCH_STRIDE)
    y_starts = get_patch_starts(image_height, patch_height, PATCH_STRIDE)

    # extract the patches
    patches = []
    patch_metadata = []

    for y_start in y_starts:
        for x_start in x_starts:

            patch, valid_width, valid_height = extract_patch(
                image, x_start, y_start, patch_width, patch_height
                )

            patches.append(patch)

            # Store the information needed to convert
            # local coordinates to global coordinates.
            patch_metadata.append({
                "is_patch": True,
                "offset_x": x_start,
                "offset_y": y_start,
                "valid_width": valid_width,
                "valid_height": valid_height,
            })
            

    patches = np.stack(patches, axis=0)

    return patches, patch_metadata



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
def predict_image_with_patches(model, image, patch_size=IMAGE_SIZE, save_debug=False):
    image = image.numpy() if tf.is_tensor(image) else image
    if image.ndim == 4:
        image = image[0]

    image_height, image_width = image.shape[:2]
    patches, metadata = create_patches(image, patch_size)

    raw_predictions = model.predict(
        patches,
        batch_size=PATCH_BATCH_SIZE,
        # verbose=0
    )

    predictions = []

    for index, patch_metadata in enumerate(metadata):
        predictions.extend(
            process_patch_view(raw_predictions, index, patch_metadata)
        )

    if save_debug:
        save_predictions_before_nms(image, predictions)

    predictions = global_nms_patch_aware(
        predictions, image_width, image_height
    )

    return predictions_to_tensors(predictions)

def resize_with_letterbox(image, target_size):
    target_h, target_w = target_size

    image_size = tf.cast(tf.shape(image)[:2], tf.float32)
    image_height = image_size[0]
    image_width = image_size[1]

    scale = tf.minimum(
        target_w / image_width,
        target_h / image_height
    )

    resized_size_float = tf.round(image_size * scale)
    resized_size = tf.cast(resized_size_float, tf.int32)

    resized_h = resized_size[0]
    resized_w = resized_size[1]

    image = tf.image.resize(image, resized_size)

    scale_y = resized_size_float[0] / image_height
    scale_x = resized_size_float[1] / image_width

    pad_y = (target_h - resized_h) // 2
    pad_x = (target_w - resized_w) // 2

    image = tf.image.pad_to_bounding_box(image, pad_y, pad_x, target_h, target_w)

    return image, scale_x, scale_y, pad_x, pad_y

def predict_total_image(model, image):
    image = image.numpy() if tf.is_tensor(image) else image
    if image.ndim == 4:
        image = image[0]

    image_height, image_width = image.shape[:2]

    resized_image, scale_x, scale_y, pad_x, pad_y = resize_with_letterbox(image, IMAGE_SIZE)

    raw_predictions = model.predict(
        tf.expand_dims(resized_image, axis=0),
        batch_size=1,
        # verbose=0
    )

    metadata = {
        "is_patch": False,
        "scale_x": float(scale_x.numpy()),
        "scale_y": float(scale_y.numpy()),
        "pad_x": float(pad_x.numpy()),
        "pad_y": float(pad_y.numpy()),
    }

    predictions = process_full_view(
        raw_predictions, 0, metadata, image_width, image_height
    )

    return predictions_to_tensors(predictions)

def clip_coordinates(image_height, image_width, x_min, y_min, x_max, y_max):
    x_min = max(0.0, min(x_min, image_width))
    y_min = max(0.0, min(y_min, image_height))
    x_max = max(0.0, min(x_max, image_width))
    y_max = max(0.0, min(y_max, image_height))
    return x_min,y_min,x_max,y_max   

def tensor_predictions_to_list(predictions):
    converted_predictions = []

    for bbox, class_id, score in zip(
        predictions["boxes"],
        predictions["classes"],
        predictions["confidence"],
    ):
        prediction = {
            "bbox": bbox.numpy().tolist(),
            "class_id": int(class_id),
            "score": float(score),
        }

        converted_predictions.append(prediction)

    return converted_predictions

def final_nms(patch_predictions, total_predictions, image_width, image_height):
    """
    Resolve patch-vs-total duplicates using greedy NMS.

    Patch-vs-patch conflicts have already been handled by
    global_nms_patch_aware().

    Total-vs-total conflicts have already been handled by
    the model decoder.

    Therefore only cross-source conflicts are considered.
    """

    patch_predictions = ensure_prediction_list(patch_predictions)
    total_predictions = ensure_prediction_list(total_predictions)

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

            iou = calculate_iou(
                prediction["bbox"],
                selected_prediction["bbox"]
            )

            containment = calculate_containment(
                prediction["bbox"],
                selected_prediction["bbox"]
            )

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

# wrapper
def predict_boxes(image_bgr, model) -> list[Box]:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a NumPy array.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape [height, width, 3].")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    predictions = tensor_predictions_to_list(
        predict_combined(model, image_rgb)
    )

    return [
        prediction_to_box(
            prediction["bbox"],
            prediction["class_id"],
            prediction["score"],
        )
        for prediction in predictions
    ]

def create_combined_views(image):
    """
    This functions create all the views from an image: patches and full_image

    patches: piece of the original image of IMAGE_SIZExIMAGE_SIZE dimension. Their union covers the entire image.
    full_image: it's the original image resize in a IMAGE_SIZExIMAGE_SIZE dimension preserving the ratio.
    """

    # patches is a list with all the patches
    # metadata list of the information related to each single patch
    patches, metadata = create_patches(image)

    full_image, scale_x, scale_y, pad_x, pad_y = resize_with_letterbox(image, IMAGE_SIZE)
    full_image = full_image.numpy() if tf.is_tensor(full_image) else full_image

    patches = patches.astype(np.float32, copy=False)
    full_image = np.asarray(full_image, dtype=np.float32)

    # structure of views: [patch0, patch1, ..., patchN, full_image]
    views = np.concatenate([patches, full_image[None, ...]], axis=0)

    # the last element of views is the full_image, so we just need to append its metadata
    metadata.append({
        "is_patch": False,
        "scale_x": float(scale_x.numpy()),
        "scale_y": float(scale_y.numpy()),
        "pad_x": float(pad_x.numpy()),
        "pad_y": float(pad_y.numpy()),
    })

    return views, metadata

def get_view_detections(predictions, index):
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
    result = []

    offset_x = metadata["offset_x"]
    offset_y = metadata["offset_y"]
    valid_width = metadata["valid_width"]
    valid_height = metadata["valid_height"]

    for bbox, class_id, score in get_view_detections(predictions, index):
        x_min, y_min, x_max, y_max = clip_coordinates(
            valid_height, valid_width, *bbox
        )

        if x_max <= x_min or y_max <= y_min:
            continue

        local_box = [x_min, y_min, x_max, y_max]

        result.append({
            "bbox": [
                x_min + offset_x,
                y_min + offset_y,
                x_max + offset_x,
                y_max + offset_y,
            ],
            "local_bbox": local_box,
            "class_id": class_id,
            "score": score,
            "patch_index": index,
            "patch_x": offset_x,
            "patch_y": offset_y,
            "patch_valid_width": valid_width,
            "patch_valid_height": valid_height,
        })

    return result

def process_full_view(predictions, index, metadata, image_width, image_height):
    result = []

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

        x_min, y_min, x_max, y_max = clip_coordinates(
            image_height, image_width, x_min, y_min, x_max, y_max
        )

        if x_max <= x_min or y_max <= y_min:
            continue

        result.append({
            "bbox": [x_min, y_min, x_max, y_max],
            "class_id": class_id,
            "score": score,
        })

    return result

def predictions_to_tensors(predictions):
    if not predictions:
        return {
            "boxes": tf.zeros((0, 4), dtype=tf.float32),
            "classes": tf.zeros((0,), dtype=tf.float32),
            "confidence": tf.zeros((0,), dtype=tf.float32),
        }

    return {
        "boxes": tf.constant([p["bbox"] for p in predictions], dtype=tf.float32),
        "classes": tf.constant([p["class_id"] for p in predictions], dtype=tf.float32),
        "confidence": tf.constant([p["score"] for p in predictions], dtype=tf.float32),
    }

def predict_views_in_batches(model, images, batch_size):
    batch_predictions = []

    for start in range(0, len(images), batch_size):
        end = start + batch_size

        predictions = model.predict(
            images[start:end],
            # verbose=0
        )

        batch_predictions.append(predictions)

    return {
        key: np.concatenate(
            [predictions[key] for predictions in batch_predictions],
            axis=0
        )
        for key in batch_predictions[0]
    }

def predict_combined(model, image):
    if tf.is_tensor(image):
        image = image.numpy()

    if image.ndim == 4:
        image = image[0]

    image_height, image_width = image.shape[:2]
    images, metadata = create_combined_views(image)

    print(
        "Views:", len(images),
        "| patches:", len(images) - 1,
        "| batch size:", PATCH_BATCH_SIZE
    )

    raw_predictions = predict_views_in_batches(
        model,
        images,
        PATCH_BATCH_SIZE
    )

    patch_predictions = []
    total_predictions = []

    for index, view_metadata in enumerate(metadata):
        if view_metadata["is_patch"]:
            patch_predictions.extend(
                process_patch_view(raw_predictions, index, view_metadata)
            )
        else:
            total_predictions.extend(
                process_full_view(
                    raw_predictions,
                    index,
                    view_metadata,
                    image_width,
                    image_height,
                )
            )

    patch_predictions = global_nms_patch_aware(
        patch_predictions,
        image_width,
        image_height
    )

    raw_predictions = final_nms(
        patch_predictions,
        total_predictions,
        image_width,
        image_height
    )

    return predictions_to_tensors(raw_predictions)

def ensure_prediction_list(predictions):
    if isinstance(predictions, list):
        return predictions
    return tensor_predictions_to_list(predictions)

def postprocess_inference_dataset(raw_predictions, metadata):
    if not metadata:
        return []

    num_views = len(metadata)
    num_predictions = len(raw_predictions["boxes"])

    if num_views != num_predictions:
        raise ValueError(
            f"Predictions/metadata mismatch: "
            f"{num_predictions} predictions for {num_views} views."
        )

    num_images = max(
        item["image_index"] for item in metadata
    ) + 1

    patch_predictions = [
        [] for _ in range(num_images)
    ]

    full_predictions = [
        [] for _ in range(num_images)
    ]

    # --------------------------------------------------
    # 1. Convert every view prediction back to
    #    original-image coordinates
    # --------------------------------------------------

    for view_index, view_metadata in enumerate(metadata):

        image_index = view_metadata["image_index"]

        image_width = view_metadata["image_width"]
        image_height = view_metadata["image_height"]

        if view_metadata["is_patch"]:

            patch_predictions[image_index].extend(
                process_patch_view(raw_predictions, view_index, view_metadata)
            )

        else:

            full_predictions[image_index].extend(
                process_full_view(raw_predictions, view_index, view_metadata, image_width, image_height)
            )

    # --------------------------------------------------
    # 2. NMS independently for every original image
    # --------------------------------------------------

    final_predictions = []

    for image_index in range(num_images):

        # All metadata belonging to the same image have
        # the same original image dimensions.
        image_metadata = next(
            item
            for item in metadata
            if item["image_index"] == image_index
        )

        image_width = image_metadata["image_width"]
        image_height = image_metadata["image_height"]

        image_patch_predictions = global_nms_patch_aware(patch_predictions[image_index],image_width,image_height)

        image_final_predictions = final_nms(image_patch_predictions, full_predictions[image_index], image_width, image_height)

        final_predictions.append([
            prediction_to_box(prediction["bbox"], prediction["class_id"], prediction["score"],)
            for prediction in image_final_predictions
        ])
        
    return final_predictions

def predict_inference_dataset(model, inference_ds, metadata):
    raw_predictions = model.predict(inference_ds)

    return postprocess_inference_dataset(raw_predictions, metadata)