#

"""Run the optimized frame-by-frame YOLO + U-Net inference pipeline and export predictions for the C++ application."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from dataset.dataset_utils import PROJECT_ROOT, load_image
from detection.detection_config import DETECTION_WEIGHTS_PATH, SEED
from detection.model_utils import configure_gpu, create_model
from evaluation.evaluate_time import create_unet_inference, create_yolo_threshold_nms_inference, run_pipeline_on_frame, warmup_unet, warmup_yolo
from segmentation.segmentation_model import build_unet


SEGMENTATION_WEIGHTS_PATH = PROJECT_ROOT / "models" / "unet_best.weights.h5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "cpp_predictions"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_arguments():
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description="Run optimized YOLO and U-Net inference for the C++ pipeline.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where predictions are saved.")
    return parser.parse_args()


def find_images(image_dir):
    """Return all supported images in deterministic order."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    image_paths.sort(key=lambda path: path.name.lower())

    if not image_paths:
        raise RuntimeError(f"No supported images found in: {image_dir}")

    return image_paths


def prepare_output_directory(output_dir):
    """Create output directories and remove old prediction masks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for mask_path in mask_dir.glob("*.png"):
        mask_path.unlink()

    return mask_dir


def load_models():
    """Build YOLO and U-Net and load their trained weights."""
    if not DETECTION_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {DETECTION_WEIGHTS_PATH}")

    if not SEGMENTATION_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"U-Net weights not found: {SEGMENTATION_WEIGHTS_PATH}")

    print("Loading YOLO model...")
    detector_model = create_model()
    detector_model.load_weights(str(DETECTION_WEIGHTS_PATH))

    print("Loading U-Net model...")
    segmentation_model = build_unet()
    segmentation_model.load_weights(str(SEGMENTATION_WEIGHTS_PATH))

    return detector_model, segmentation_model


def save_detections(image_paths, all_predicted_boxes, output_path):
    """Save YOLO predictions in a CSV file readable by C++."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "x_min", "y_min", "x_max", "y_max", "class_id", "score"])

        for image_path, predicted_boxes in zip(image_paths, all_predicted_boxes):
            for box in predicted_boxes:
                if box.score is None:
                    raise ValueError(f"Prediction without confidence score in image: {image_path.name}")

                writer.writerow([image_path.name, box.x_min, box.y_min, box.x_max, box.y_max, box.class_id, box.score])


def save_prediction_mask(image_path, image_bgr, predicted_mask, mask_dir):
    """Validate and save one semantic prediction mask."""
    predicted_mask = np.asarray(predicted_mask)

    if predicted_mask.ndim != 2:
        raise ValueError(f"Invalid segmentation mask shape for {image_path.name}: {predicted_mask.shape}")

    if predicted_mask.shape != image_bgr.shape[:2]:
        raise ValueError(f"Segmentation mask size mismatch for {image_path.name}: mask={predicted_mask.shape}, image={image_bgr.shape[:2]}")

    predicted_mask = predicted_mask.astype(np.uint8, copy=False)
    output_path = mask_dir / f"{image_path.stem}.png"

    if not cv2.imwrite(str(output_path), predicted_mask):
        raise RuntimeError(f"Could not save segmentation mask: {output_path}")


def run_optimized_inference(image_paths, yolo_infer, unet_infer, mask_dir):
    """Run the same optimized frame-by-frame pipeline used by evaluate_time.py."""
    all_predicted_boxes = []

    detection_seconds = 0.0
    segmentation_seconds = 0.0
    total_seconds = 0.0

    for image_index, image_path in enumerate(image_paths):
        image_bgr = load_image(image_path)

        predicted_boxes, predicted_mask, frame_timings, num_views = run_pipeline_on_frame(image_bgr, yolo_infer, unet_infer)

        all_predicted_boxes.append(predicted_boxes)
        save_prediction_mask(image_path, image_bgr, predicted_mask, mask_dir)

        detection_ms = frame_timings["detection_preprocess"] + frame_timings["yolo_inference"] + frame_timings["detection_postprocess"]
        segmentation_ms = frame_timings["segmentation_preprocess"] + frame_timings["unet_inference"] + frame_timings["segmentation_postprocess"]

        detection_seconds += detection_ms / 1000.0
        segmentation_seconds += segmentation_ms / 1000.0
        total_seconds += frame_timings["pipeline_total"] / 1000.0

        print(f"[{image_index + 1}/{len(image_paths)}] {image_path.name} | views: {num_views} | boxes: {len(predicted_boxes)} | pipeline: {frame_timings['pipeline_total']:.2f} ms")

    return all_predicted_boxes, detection_seconds, segmentation_seconds, total_seconds


def save_timing(num_images, detection_seconds, segmentation_seconds, total_seconds, output_path):
    """Save optimized pipeline timing and FPS."""
    fps = num_images / total_seconds if total_seconds > 0.0 else 0.0

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["num_images", "detection_seconds", "segmentation_seconds", "total_seconds", "fps"])
        writer.writerow([num_images, detection_seconds, segmentation_seconds, total_seconds, fps])

    return fps


def main():
    """Run the optimized inference export pipeline."""
    arguments = parse_arguments()

    tf.keras.utils.set_random_seed(SEED)
    configure_gpu()

    image_dir = arguments.image_dir.resolve()
    output_dir = arguments.output_dir.resolve()

    image_paths = find_images(image_dir)
    mask_dir = prepare_output_directory(output_dir)

    print("Images:", len(image_paths))
    print("Input directory:", image_dir)
    print("Output directory:", output_dir)

    detector_model, segmentation_model = load_models()

    yolo_infer, _ = create_yolo_threshold_nms_inference(detector_model)
    unet_infer, _ = create_unet_inference(segmentation_model)

    print()
    print("Warming up optimized inference graphs...")

    warmup_yolo(yolo_infer)
    warmup_unet(unet_infer)

    print()
    print("Running optimized frame-by-frame inference...")

    all_predicted_boxes, detection_seconds, segmentation_seconds, total_seconds = run_optimized_inference(image_paths, yolo_infer, unet_infer, mask_dir)

    print("Saving detection predictions...")
    save_detections(image_paths, all_predicted_boxes, output_dir / "detections.csv")

    fps = save_timing(len(image_paths), detection_seconds, segmentation_seconds, total_seconds, output_dir / "timing.csv")

    print()
    print("Inference completed.")
    print(f"Detection time: {detection_seconds:.6f} s")
    print(f"Segmentation time: {segmentation_seconds:.6f} s")
    print(f"Total pipeline time: {total_seconds:.6f} s")
    print(f"FPS: {fps:.6f}")
    print("Detections:", output_dir / "detections.csv")
    print("Masks:", mask_dir)
    print("Timing:", output_dir / "timing.csv")


if __name__ == "__main__":
    main()
