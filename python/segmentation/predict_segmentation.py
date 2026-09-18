#Novkovic

"""U-Net inference and semantic mask reconstruction."""

import time

import cv2
import numpy as np
import tensorflow as tf

from timing.timing_utils import elapsed_ms
from dataset.dataset_utils import BACKGROUND_ID
from segmentation.segmentation_config import INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, MASK_THRESHOLD
from segmentation.segmentation_dataset import create_crop_box, letterbox_sample


UNET_BATCH_SIZE = 16


def prepare_segmentation_inputs(image_bgr, boxes):
    """Create U-Net crops from YOLO bounding boxes."""
    image_height, image_width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    model_inputs = []
    roi_metadata = []

    for box in boxes:
        # Enlarge the YOLO box to include some context around the cone.
        x_min, y_min, x_max, y_max = create_crop_box(box, image_height, image_width, training=False)

        image_crop = image_rgb[y_min:y_max, x_min:x_max]

        # Resize the crop to the U-Net input size while preserving its aspect ratio.
        image_crop, _, letterbox_metadata = letterbox_sample(image_crop)

        image_crop = image_crop.astype(np.float32) / 255.0

        model_inputs.append(image_crop)

        # Store the information needed to map the predicted mask back to the original image.
        roi_metadata.append((box, x_min, y_min, x_max, y_max, letterbox_metadata))

    return np.stack(model_inputs), roi_metadata




def create_unet_inference(model):
    """Create the optimized U-Net inference function with a fixed batch size."""

    @tf.function(input_signature=[tf.TensorSpec(shape=(UNET_BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), dtype=tf.float32)])
    def graph_predict(inputs):
        return model(inputs, training=False)

    def infer(inputs):
        outputs = []

        for start_index in range(0, len(inputs), UNET_BATCH_SIZE):
            batch = inputs[start_index:start_index + UNET_BATCH_SIZE]

            real_batch_size = len(batch)

            # Pad the batch to keep a fixed input shape.
            if real_batch_size < UNET_BATCH_SIZE:
                padding_size = UNET_BATCH_SIZE - real_batch_size
                padding = np.zeros((padding_size, *batch.shape[1:]), dtype=batch.dtype)
                batch = np.concatenate((batch, padding), axis=0)

            batch_predictions = graph_predict(batch).numpy()

            # Keep only predictions corresponding to real cone crops.
            outputs.append(batch_predictions[:real_batch_size])

        return np.concatenate(outputs, axis=0)

    return infer

def predict_segmentation(image_bgr, boxes, unet_infer):
    """Create the semantic segmentation mask from YOLO detections."""
    timings = {"segmentation_preprocess": 0.0, "unet_inference": 0.0, "segmentation_postprocess": 0.0}

    image_height, image_width = image_bgr.shape[:2]

    # Start with an image containing only background pixels.
    semantic_mask = np.full((image_height, image_width), BACKGROUND_ID, dtype=np.uint8)

    if not boxes:
        return semantic_mask, timings

    start_time = time.perf_counter()

    # Prepare one U-Net crop for each YOLO detection.
    model_inputs, roi_metadata = prepare_segmentation_inputs(image_bgr, boxes)

    timings["segmentation_preprocess"] = elapsed_ms(start_time)

    start_time = time.perf_counter()

    # Predict a binary cone mask for every detected ROI.
    predictions = unet_infer(model_inputs)

    timings["unet_inference"] = elapsed_ms(start_time)

    start_time = time.perf_counter()

    # Store the YOLO confidence assigned to each pixel to resolve overlapping detections.
    score_mask = np.full((image_height, image_width), -1.0, dtype=np.float32)

    for prediction, metadata in zip(predictions, roi_metadata):
        box, x_min, y_min, x_max, y_max, letterbox_metadata = metadata

        top_padding, left_padding, resized_height, resized_width = letterbox_metadata

        probability_map = prediction[:, :, 0]

        # Remove the padding previously added.
        probability_map = probability_map[top_padding:top_padding + resized_height, left_padding:left_padding + resized_width]

        roi_width = x_max - x_min
        roi_height = y_max - y_min

        # Restore the predicted mask to the original ROI size.
        probability_map = cv2.resize(probability_map, (roi_width, roi_height), interpolation=cv2.INTER_LINEAR)

        # Convert U-Net probabilities into a binary cone/background mask.
        binary_mask = probability_map >= MASK_THRESHOLD

        detection_score = box.score if box.score is not None else 1.0

        semantic_region = semantic_mask[y_min:y_max, x_min:x_max]
        score_region = score_mask[y_min:y_max, x_min:x_max]

        # In overlapping regions, keep the class from the detection with higher confidence.
        update_pixels = binary_mask & (detection_score > score_region)

        # U-Net determines the cone pixels, while YOLO provides their semantic class.
        semantic_region[update_pixels] = box.class_id
        score_region[update_pixels] = detection_score

    timings["segmentation_postprocess"] = elapsed_ms(start_time)

    return semantic_mask, timings