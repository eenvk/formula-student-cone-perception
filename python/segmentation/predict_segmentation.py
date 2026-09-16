#Novkovic
"""U-Net inference and semantic mask reconstruction."""

import random
import time

import cv2
import numpy as np
import tensorflow as tf

from timing.timing_utils import elapsed_ms
from dataset.dataset_utils import BACKGROUND_ID
from segmentation.segmentation_config import INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, MASK_THRESHOLD, RANDOM_SEED
from segmentation.segmentation_dataset import create_crop_box, letterbox_sample


UNET_MAX_BATCH_SIZE = 32

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