# Granati

"""Detection inference and prediction export for the C++ evaluation pipeline."""

import csv

import tensorflow as tf

from dataset.dataset_utils import load_annotation
from detection.data_pipeline import build_inference_dataset, build_inference_metadata
from detection.detection_config import CHECKPOINT_PATH
from detection.inference import predict_inference_dataset
from detection.model_utils import create_model


DETECTION_WEIGHTS_PATH = CHECKPOINT_PATH


def load_detection_model():
    """Build the YOLO model and load the trained weights."""

    if not DETECTION_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Detection weights not found: {DETECTION_WEIGHTS_PATH}")

    model = create_model()
    model.load_weights(str(DETECTION_WEIGHTS_PATH))

    return model


def prepare_detection_inference(pairs):
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

    inference_dataset = build_inference_dataset(image_paths)
    inference_metadata = build_inference_metadata(image_shapes)

    return inference_dataset, inference_metadata


def run_detection_inference(model, inference_dataset, inference_metadata):
    """Run YOLO inference and return the final predicted bounding boxes."""

    return predict_inference_dataset(model, inference_dataset, inference_metadata)


def save_detection_predictions(pairs, predictions, output_path):
    """Save final YOLO detections in CSV format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["image_name", "x_min", "y_min", "x_max", "y_max", "class_id", "score"])

        for (image_path, _), boxes in zip(pairs, predictions):
            for box in boxes:
                writer.writerow([image_path.name, box.x_min, box.y_min, box.x_max, box.y_max, box.class_id, box.score])