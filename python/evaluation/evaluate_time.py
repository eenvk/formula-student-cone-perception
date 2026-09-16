"""Optimized yolo + unet inference pipeline"""

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

def elapsed_ms(start_time):
    """Return elapsed time in milliseconds"""
    return (time.perf_counter() - start_time) * 1000.0



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


def prepare_segmentation_inputs(image_bgr, boxes):
    """Create unet crops from yolo bounding boxes"""
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
    """Create the optimized unet inference function"""
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
    """Create the semantic segmentation mask from yolo detections"""
    timings = {"segmentation_preprocess": 0.0, "unet_inference": 0.0, "segmentation_postprocess": 0.0 }

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
    """Run yolo and unet on one frame"""
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
    """Run one complete frame before timing starts"""
    print("Running one pipeline warm-up frame...")

    _, _, _, num_views = run_pipeline_on_frame(image_bgr, yolo_infer, unet_infer)

    print(f"Warm-up completed with {num_views} yolo views")
    print("Warm-up time is excluded from fps")