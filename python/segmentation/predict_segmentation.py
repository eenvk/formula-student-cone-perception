#novkovic

"""Cone segmentation inference from detected bounding boxes."""

import random
from typing import Sequence

import cv2
import numpy as np
import tensorflow as tf

from dataset.dataset_utils import BACKGROUND_ID, Box, PROJECT_ROOT
from segmentation.segmentation_config import MASK_THRESHOLD, RANDOM_SEED
from segmentation.segmentation_dataset import create_crop_box, letterbox_sample


MODEL_DIR = PROJECT_ROOT / "models"
BEST_WEIGHTS_PATH = MODEL_DIR / "unet_best.weights.h5"


def predict_segmentation(image_bgr: np.ndarray, boxes: Sequence[Box], model: tf.keras.Model) -> np.ndarray:
    """Predict a full-image semantic mask from detector bounding boxes."""

    image_height, image_width = image_bgr.shape[:2]
    semantic_mask = np.full((image_height, image_width), BACKGROUND_ID, dtype=np.uint8)

    if not boxes:
        return semantic_mask

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    model_inputs = []
    roi_metadata = []

    rng = random.Random(RANDOM_SEED)

    for box in boxes:
        x_min, y_min, x_max, y_max = create_crop_box(box, image_height, image_width, rng, training=False)

        image_crop = image_rgb[y_min:y_max, x_min:x_max]

        image_crop, _, letterbox_metadata = letterbox_sample(image_crop)

        image_crop = image_crop.astype(np.float32) / 255.0

        model_inputs.append(image_crop)
        roi_metadata.append((box, x_min, y_min, x_max, y_max, letterbox_metadata))

    predictions = model.predict(np.stack(model_inputs), verbose=0)

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

    return semantic_mask