# Novkovic

"""Run YOLO and U-Net inference and export predictions for the C++ pipeline."""

import csv
import time

import cv2
import tensorflow as tf

from dataset.dataset_utils import PROJECT_ROOT, get_segmentation_test_pairs, load_annotation, load_image
from detection.data_pipeline import build_inference_dataset, build_inference_metadata
from detection.detection_config import CHECKPOINT_PATH, SEED
from detection.inference import predict_inference_dataset
from detection.model_utils import create_model
from segmentation.predict_segmentation import predict_segmentation
from segmentation.segmentation_model import build_unet


MODEL_DIR = PROJECT_ROOT / "models"

DETECTION_WEIGHTS_PATH = CHECKPOINT_PATH
SEGMENTATION_WEIGHTS_PATH = MODEL_DIR / "unet_best.weights.h5"

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "cpp_predictions"
DETECTIONS_PATH = OUTPUT_DIR / "detections.csv"
MASKS_DIR = OUTPUT_DIR / "masks"
TIMING_PATH = OUTPUT_DIR / "timing.csv"


def prepare_inference_data(pairs):
    """Prepare image paths and original image sizes for YOLO inference."""

    image_paths = []
    image_shapes = []

    for image_path, annotation_path in pairs:
        annotation = load_annotation(annotation_path)

        image_height = annotation["size"]["height"]
        image_width = annotation["size"]["width"]

        image_paths.append(str(image_path))
        image_shapes.append([image_height, image_width])

    image_paths = tf.constant(image_paths, dtype=tf.string)
    image_shapes = tf.constant(image_shapes, dtype=tf.int32)

    return image_paths, image_shapes


def save_detection_predictions(pairs, predictions):
    """Save final YOLO detections in CSV format."""

    with DETECTIONS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["image_name", "x_min", "y_min", "x_max", "y_max", "class_id", "score"])

        for (image_path, _), boxes in zip(pairs, predictions):
            for box in boxes:
                writer.writerow([image_path.name, box.x_min, box.y_min, box.x_max, box.y_max, box.class_id, box.score])


def save_timing(num_images, detection_seconds, segmentation_seconds):
    """Save total inference time and FPS."""

    total_seconds = detection_seconds + segmentation_seconds

    if total_seconds > 0.0:
        fps = num_images / total_seconds
    else:
        fps = 0.0

    with TIMING_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["num_images", "detection_seconds", "segmentation_seconds", "total_seconds", "fps"])
        writer.writerow([num_images, detection_seconds, segmentation_seconds, total_seconds, fps])


def main():
    tf.keras.utils.set_random_seed(SEED)

    if not DETECTION_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Detection weights not found: {DETECTION_WEIGHTS_PATH}")

    if not SEGMENTATION_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Segmentation weights not found: {SEGMENTATION_WEIGHTS_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = get_segmentation_test_pairs()

    print("Test images:", len(pairs))

    detector_model = create_model()
    detector_model.load_weights(str(DETECTION_WEIGHTS_PATH))

    segmentation_model = build_unet()
    segmentation_model.load_weights(str(SEGMENTATION_WEIGHTS_PATH))

    image_paths, image_shapes = prepare_inference_data(pairs)

    inference_dataset = build_inference_dataset(image_paths)
    inference_metadata = build_inference_metadata(image_shapes)

    detection_start = time.perf_counter()

    predicted_boxes = predict_inference_dataset(detector_model, inference_dataset, inference_metadata)

    detection_seconds = time.perf_counter() - detection_start

    if len(predicted_boxes) != len(pairs):
        raise ValueError(f"Predictions/images mismatch: {len(predicted_boxes)} predictions for {len(pairs)} images.")

    save_detection_predictions(pairs, predicted_boxes)

    segmentation_seconds = 0.0

    for image_index, ((image_path, _), boxes) in enumerate(zip(pairs, predicted_boxes)):
        image_bgr = load_image(image_path)

        segmentation_start = time.perf_counter()

        predicted_mask = predict_segmentation(image_bgr, boxes, segmentation_model)

        segmentation_seconds += time.perf_counter() - segmentation_start

        mask_path = MASKS_DIR / f"{image_path.stem}.png"

        if not cv2.imwrite(str(mask_path), predicted_mask):
            raise RuntimeError(f"Could not save mask: {mask_path}")

        print(f"Processed {image_index + 1}/{len(pairs)} images")

    save_timing(len(pairs), detection_seconds, segmentation_seconds)

    print()
    print("Predictions exported successfully.")
    print("Detections:", DETECTIONS_PATH)
    print("Masks:", MASKS_DIR)
    print("Timing:", TIMING_PATH)


if __name__ == "__main__":
    main()