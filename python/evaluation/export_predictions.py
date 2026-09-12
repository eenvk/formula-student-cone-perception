#

"""Run the complete inference pipeline and export predictions for C++ evaluation."""

import tensorflow as tf

from dataset.dataset_utils import PROJECT_ROOT, get_segmentation_test_pairs
from detection.detection_config import SEED
from evaluation.detection_export import load_detection_model, prepare_detection_inference, run_detection_inference, save_detection_predictions
from evaluation.inference_timing import measure_execution, save_timing
from evaluation.segmentation_export import load_segmentation_image, load_segmentation_model, run_segmentation_inference, save_segmentation_prediction


OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "cpp_predictions"
DETECTIONS_PATH = OUTPUT_DIR / "detections.csv"
MASKS_DIR = OUTPUT_DIR / "masks"
TIMING_PATH = OUTPUT_DIR / "timing.csv"


def main():
    """Run detection and segmentation inference and export all results."""

    tf.keras.utils.set_random_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = get_segmentation_test_pairs()

    print("Test images:", len(pairs))

    detector_model = load_detection_model()
    segmentation_model = load_segmentation_model()

    inference_dataset, inference_metadata = prepare_detection_inference(pairs)

    predicted_boxes, detection_seconds = measure_execution(run_detection_inference, detector_model, inference_dataset, inference_metadata)

    if len(predicted_boxes) != len(pairs):
        raise ValueError(f"Predictions/images mismatch: {len(predicted_boxes)} predictions for {len(pairs)} images.")

    save_detection_predictions(pairs, predicted_boxes, DETECTIONS_PATH)

    segmentation_seconds = 0.0

    for image_index, ((image_path, _), boxes) in enumerate(zip(pairs, predicted_boxes)):
        image_bgr = load_segmentation_image(image_path)

        predicted_mask, image_segmentation_seconds = measure_execution(run_segmentation_inference, segmentation_model, image_bgr, boxes)

        segmentation_seconds += image_segmentation_seconds

        save_segmentation_prediction(image_path, predicted_mask, MASKS_DIR)

        print(f"Processed {image_index + 1}/{len(pairs)} images")

    save_timing(TIMING_PATH, len(pairs), detection_seconds, segmentation_seconds)

    print()
    print("Predictions exported successfully.")
    print("Detections:", DETECTIONS_PATH)
    print("Masks:", MASKS_DIR)
    print("Timing:", TIMING_PATH)


if __name__ == "__main__":
    main()