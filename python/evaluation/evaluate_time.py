"""Optimized YOLO + U-Net inference pipeline."""

import importlib
import time
import random

import cv2
import numpy as np
import tensorflow as tf

from dataset.dataset_utils import BACKGROUND_ID
from detection.data_pipeline import build_inference_metadata, create_combined_views
from detection.detection_config import SEED
from detection.inference import postprocess_inference_dataset
from segmentation.segmentation_config import INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, MASK_THRESHOLD, RANDOM_SEED
from segmentation.segmentation_dataset import create_crop_box, letterbox_sample


UNET_MAX_BATCH_SIZE = 32

YOLO_SCORE_THRESHOLD = 0.25
YOLO_PRE_NMS_TOP_K = 100
YOLO_NMS_IOU_THRESHOLD = 0.50
YOLO_NMS_MAX_DETECTIONS = 100


def elapsed_ms(start_time):
    """Return elapsed time in milliseconds."""
    return (time.perf_counter() - start_time) * 1000.0


def tensor_to_numpy(tensor):
    """Convert a TensorFlow tensor to NumPy."""
    return tensor.numpy()


def create_yolo_inference(model):
    """Create the optimized YOLO inference function."""
    implementation = importlib.import_module(type(model).__module__)

    required_functions = ("decode_regression_to_boxes", "get_anchors", "dist2bbox", "bounding_box", "ops")

    if any(not hasattr(implementation, name) for name in required_functions):
        raise RuntimeError("Installed YOLO implementation does not contain the required decoder functions.")

    @tf.function(input_signature=[tf.TensorSpec((None, 800, 800, 3), tf.float32)])
    def graph_predict(images):
        raw_outputs = model(images, training=False)

        distances = implementation.decode_regression_to_boxes(raw_outputs["boxes"])
        anchors, strides = implementation.get_anchors(image_shape=images.shape[1:])
        strides = implementation.ops.expand_dims(strides, axis=-1)

        boxes = implementation.dist2bbox(distances, anchors) * strides

        boxes_xyxy = implementation.bounding_box.convert_format(boxes, source="xyxy", target="xyxy", images=images)

        class_scores = raw_outputs["classes"]

        confidence = tf.reduce_max(class_scores, axis=-1)
        classes = tf.argmax(class_scores, axis=-1, output_type=tf.int32)

        threshold_mask = confidence >= YOLO_SCORE_THRESHOLD

        filtered_scores = tf.where(threshold_mask, confidence, tf.fill(tf.shape(confidence), tf.constant(-1.0, dtype=confidence.dtype)))

        num_candidates = tf.shape(filtered_scores)[1]
        top_k = tf.minimum(num_candidates, YOLO_PRE_NMS_TOP_K)

        top_scores, top_indices = tf.math.top_k(filtered_scores, k=top_k, sorted=True)

        top_boxes = tf.gather(boxes_xyxy, top_indices, batch_dims=1)
        top_classes = tf.gather(classes, top_indices, batch_dims=1)

        top_boxes_yxyx = tf.stack([top_boxes[..., 1], top_boxes[..., 0], top_boxes[..., 3], top_boxes[..., 2]], axis=-1)

        max_coordinate = tf.reduce_max(tf.abs(top_boxes_yxyx), axis=[1, 2], keepdims=True) + 1.0

        class_offsets = tf.cast(top_classes, top_boxes_yxyx.dtype)[..., None] * max_coordinate

        nms_boxes = top_boxes_yxyx + class_offsets

        def nms_single(arguments):
            boxes_one, scores_one = arguments

            return tf.image.non_max_suppression_padded(boxes=boxes_one, scores=scores_one, max_output_size=YOLO_NMS_MAX_DETECTIONS, iou_threshold=YOLO_NMS_IOU_THRESHOLD, score_threshold=YOLO_SCORE_THRESHOLD, pad_to_max_output_size=True)

        selected_indices, num_detections = tf.map_fn(nms_single, (nms_boxes, top_scores), fn_output_signature=(tf.TensorSpec((YOLO_NMS_MAX_DETECTIONS,), tf.int32), tf.TensorSpec((), tf.int32)))

        selected_boxes = tf.gather(top_boxes, selected_indices, batch_dims=1)
        selected_scores = tf.gather(top_scores, selected_indices, batch_dims=1)
        selected_classes = tf.gather(top_classes, selected_indices, batch_dims=1)

        valid_positions = tf.sequence_mask(num_detections, maxlen=YOLO_NMS_MAX_DETECTIONS)

        selected_boxes = tf.where(valid_positions[..., None], selected_boxes, tf.zeros_like(selected_boxes))
        selected_scores = tf.where(valid_positions, selected_scores, tf.zeros_like(selected_scores))
        selected_classes = tf.where(valid_positions, selected_classes, tf.zeros_like(selected_classes))

        selected_boxes = implementation.bounding_box.convert_format(selected_boxes, source="xyxy", target=model.bounding_box_format, images=images)

        return {
            "boxes": selected_boxes,
            "confidence": selected_scores,
            "classes": selected_classes,
            "num_detections": num_detections,
        }

    def infer(inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        outputs = graph_predict(inputs)

        return tf.nest.map_structure(tensor_to_numpy, outputs)

    return infer


def build_detection_inputs(image_bgr):
    """Create YOLO views and metadata for one frame."""
    image_height, image_width = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    views, _ = create_combined_views(image_rgb)

    image_shapes = tf.constant([[image_height, image_width]], dtype=tf.int32)
    metadata = build_inference_metadata(image_shapes)

    if len(views) != len(metadata):
        raise ValueError(f"Views/metadata mismatch: {len(views)} views and {len(metadata)} metadata elements.")

    inputs = np.ascontiguousarray(views, dtype=np.float32)

    return inputs, metadata, len(views)


def detect_frame(image_bgr, yolo_infer):
    """Run YOLO detection on one frame."""
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
        raise ValueError(f"Expected one prediction group, got {len(predictions)}.")

    return predictions[0], timings, num_views


def prepare_segmentation_inputs(image_bgr, boxes):
    """Create U-Net crops from YOLO bounding boxes."""
    image_height, image_width = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    model_inputs = []
    roi_metadata = []

    random_generator = random.Random(RANDOM_SEED)

    for box in boxes:
        x_min, y_min, x_max, y_max = create_crop_box(box, image_height, image_width, random_generator, training=False)

        image_crop = image_rgb[y_min:y_max, x_min:x_max]

        image_crop, _, letterbox_metadata = letterbox_sample(image_crop)

        image_crop = image_crop.astype(np.float32) / 255.0

        model_inputs.append(image_crop)
        roi_metadata.append((box, x_min, y_min, x_max, y_max, letterbox_metadata))

    return np.stack(model_inputs), roi_metadata


def create_unet_inference(model):
    """Create the optimized U-Net inference function."""
    @tf.function(input_signature=[tf.TensorSpec((None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), tf.float32)])
    def graph_predict(inputs):
        return model(inputs, training=False)

    def infer(inputs):
        if len(inputs) == 0:
            return np.empty((0, INPUT_HEIGHT, INPUT_WIDTH, 1), dtype=np.float32)

        outputs = []

        for start_index in range(0, len(inputs), UNET_MAX_BATCH_SIZE):
            batch = inputs[start_index:start_index + UNET_MAX_BATCH_SIZE]
            batch_predictions = graph_predict(batch).numpy()
            outputs.append(batch_predictions)

        if len(outputs) == 1:
            return outputs[0]

        return np.concatenate(outputs, axis=0)

    return infer


def predict_segmentation(image_bgr, boxes, unet_infer):
    """Create the semantic segmentation mask from YOLO detections."""
    timings = {
        "segmentation_preprocess": 0.0,
        "unet_inference": 0.0,
        "segmentation_postprocess": 0.0,
    }

    image_height, image_width = image_bgr.shape[:2]

    semantic_mask = np.full((image_height, image_width), BACKGROUND_ID, dtype=np.uint8)

    if not boxes:
        return semantic_mask, timings

    start_time = time.perf_counter()

    model_inputs, roi_metadata = prepare_segmentation_inputs(image_bgr, boxes)

    timings["segmentation_preprocess"] = elapsed_ms(start_time)

    start_time = time.perf_counter()

    predictions = unet_infer(model_inputs)

    timings["unet_inference"] = elapsed_ms(start_time)

    start_time = time.perf_counter()

    score_mask = np.full((image_height, image_width), -1.0, dtype=np.float32)

    for prediction, metadata in zip(predictions, roi_metadata):
        box, x_min, y_min, x_max, y_max, letterbox_metadata = metadata

        top_padding, left_padding, resized_height, resized_width = letterbox_metadata

        probability_map = prediction[:, :, 0]

        probability_map = probability_map[top_padding:top_padding + resized_height, left_padding:left_padding + resized_width]

        roi_width = x_max - x_min
        roi_height = y_max - y_min

        probability_map = cv2.resize(probability_map, (roi_width, roi_height), interpolation=cv2.INTER_LINEAR)

        binary_mask = probability_map >= MASK_THRESHOLD

        detection_score = box.score if box.score is not None else 1.0

        semantic_region = semantic_mask[y_min:y_max, x_min:x_max]
        score_region = score_mask[y_min:y_max, x_min:x_max]

        update_pixels = binary_mask & (detection_score > score_region)

        semantic_region[update_pixels] = box.class_id
        score_region[update_pixels] = detection_score

    timings["segmentation_postprocess"] = elapsed_ms(start_time)

    return semantic_mask, timings


def run_pipeline_on_frame(image_bgr, yolo_infer, unet_infer):
    """Run YOLO and U-Net on one frame."""
    pipeline_start = time.perf_counter()

    predicted_boxes, detection_timings, num_views = detect_frame(image_bgr, yolo_infer)

    predicted_mask, segmentation_timings = predict_segmentation(image_bgr, predicted_boxes, unet_infer)

    pipeline_total = elapsed_ms(pipeline_start)

    timings = {
        **detection_timings,
        **segmentation_timings,
        "pipeline_total": pipeline_total,
    }

    return predicted_boxes, predicted_mask, timings, num_views


def warmup_pipeline(image_bgr, yolo_infer, unet_infer):
    """Run one complete frame before timing starts."""
    print("Running one pipeline warm-up frame...")

    _, _, _, num_views = run_pipeline_on_frame(image_bgr, yolo_infer, unet_infer)

    print(f"Warm-up completed with {num_views} YOLO views.")
    print("Warm-up time is excluded from FPS.")