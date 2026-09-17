#Novkovic
"""Unet inference and semantic mask reconstruction"""

import time

import cv2
import numpy as np

from timing.timing_utils import elapsed_ms
from dataset.dataset_utils import BACKGROUND_ID
from segmentation.segmentation_config import MASK_THRESHOLD
from segmentation.segmentation_dataset import create_crop_box, letterbox_sample


UNET_MAX_BATCH_SIZE = 32

def prepare_segmentation_inputs(image_bgr, boxes):
    """Create unet crops from yolo bounding boxes"""
    image_height, image_width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    model_inputs = []
    roi_metadata = []

    for box in boxes:
        #enlarge yolo box to include some context arount the cone
        x_min, y_min, x_max, y_max = create_crop_box(box, image_height, image_width, training=False)

        image_crop = image_rgb[y_min:y_max, x_min:x_max]

        #resize crop to the unet input size, preserving ratio
        image_crop, _, letterbox_metadata = letterbox_sample(image_crop)

        image_crop = image_crop.astype(np.float32) / 255.0

        model_inputs.append(image_crop)

        #store information needed to map the predicted mask back to the original image
        roi_metadata.append((box, x_min, y_min, x_max, y_max, letterbox_metadata))

    return np.stack(model_inputs), roi_metadata


def create_unet_inference(model):
    """Create the optimized unet inference function"""

    def infer(inputs):

        outputs = []

        #process cone crops in batches to limit memory usage
        for start_index in range(0, len(inputs), UNET_MAX_BATCH_SIZE):
            batch = inputs[start_index:start_index + UNET_MAX_BATCH_SIZE]
            batch_predictions = model(batch,training=False).numpy()
            outputs.append(batch_predictions)

        return np.concatenate(outputs, axis=0)

    return infer


def predict_segmentation(image_bgr, boxes, unet_infer):
    """Create the semantic segmentation mask from yolo detections"""

    timings = {"segmentation_preprocess": 0.0, "unet_inference": 0.0, "segmentation_postprocess": 0.0 }

    image_height, image_width = image_bgr.shape[:2]

    #start with an image containing only background pixels
    semantic_mask = np.full((image_height, image_width), BACKGROUND_ID, dtype=np.uint8)

    if not boxes:
        return semantic_mask, timings

    start_time = time.perf_counter()

    #prepare one unet crop for each yolo detection
    model_inputs, roi_metadata = prepare_segmentation_inputs(image_bgr, boxes)

    timings["segmentation_preprocess"] = elapsed_ms(start_time)

    start_time = time.perf_counter()

    #predict a binary cone mask for every detected roi
    predictions = unet_infer(model_inputs)

    timings["unet_inference"] = elapsed_ms(start_time)

    start_time = time.perf_counter()

    #store yolo confidence assigned to each pixel to resolve overlapping detections
    score_mask = np.full((image_height, image_width), -1.0, dtype=np.float32)

    for prediction, metadata in zip(predictions, roi_metadata):
        box, x_min, y_min, x_max, y_max, letterbox_metadata = metadata

        top_padding, left_padding, resized_height, resized_width = letterbox_metadata

        probability_map = prediction[:, :, 0]

        #remove the padding previously added
        probability_map = probability_map[top_padding:top_padding + resized_height, left_padding:left_padding + resized_width]

        roi_width = x_max - x_min
        roi_height = y_max - y_min

        #restore predicted mask to original roi sizw
        probability_map = cv2.resize(probability_map, (roi_width, roi_height), interpolation=cv2.INTER_LINEAR)

        #convert unet probabilities into a binary cone/background mask
        binary_mask = probability_map >= MASK_THRESHOLD

        detection_score = box.score if box.score is not None else 1.0

        semantic_region = semantic_mask[y_min:y_max, x_min:x_max]
        score_region = score_mask[y_min:y_max, x_min:x_max]

        #in overlapping regions keep the class from the detection with higher confidence
        update_pixels = binary_mask & (detection_score > score_region)

        #unet determines the cone pixels, while YOLO provides their semantic class.
        semantic_region[update_pixels] = box.class_id
        score_region[update_pixels] = detection_score

    timings["segmentation_postprocess"] = elapsed_ms(start_time)

    return semantic_mask, timings