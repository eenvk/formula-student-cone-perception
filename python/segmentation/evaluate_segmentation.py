#novkovic

"""Evaluate cone segmentation on the validation set using ground-truth bounding boxes."""

from pathlib import Path

import cv2
import numpy as np

from dataset.dataset_utils import PROJECT_ROOT, annotation_to_instances, annotation_to_semantic_mask, create_overlay, get_segmentation_train_pairs, load_image, train_validation_split
from evaluation.evaluation_utils import SegmentationEvaluator
from segmentation.predict_segmentation import predict_segmentation
from segmentation.segmentation_config import RANDOM_SEED, VALIDATION_FRACTION
from segmentation.segmentation_model import build_unet


MODEL_DIR = PROJECT_ROOT / "models"
BEST_WEIGHTS_PATH = MODEL_DIR / "unet_best.weights.h5"

VISUALIZATION_DIR = MODEL_DIR / "segmentation_validation"
NUM_VISUALIZATIONS = 20


def main():
    """Evaluate the best segmentation model on the validation set."""

    if not BEST_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Segmentation weights not found: {BEST_WEIGHTS_PATH}")

    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    pairs = get_segmentation_train_pairs()
    _, validation_pairs = train_validation_split(pairs, validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED)

    print(f"Validation images: {len(validation_pairs)}")
    print(f"Loading weights from: {BEST_WEIGHTS_PATH}")

    model = build_unet()
    model.load_weights(BEST_WEIGHTS_PATH)

    evaluator = SegmentationEvaluator()

    for image_index, (image_path, annotation_path) in enumerate(validation_pairs):
        image_bgr = load_image(image_path)
        image_height, image_width = image_bgr.shape[:2]

        gt_boxes = annotation_to_instances(annotation_path, image_height, image_width, bitmap_only=True)
        gt_mask = annotation_to_semantic_mask(annotation_path, image_height, image_width)

        predicted_mask = predict_segmentation(image_bgr, gt_boxes, model)

        evaluator.update(gt_mask, predicted_mask)

        if image_index < NUM_VISUALIZATIONS:
            gt_overlay = create_overlay(image_bgr, gt_mask)
            predicted_overlay = create_overlay(image_bgr, predicted_mask)
            comparison = np.hstack((image_bgr, gt_overlay, predicted_overlay))
            output_path = VISUALIZATION_DIR / f"{image_index:03d}_{image_path.stem}.jpg"
            cv2.imwrite(str(output_path), comparison)

        if (image_index + 1) % 50 == 0 or image_index + 1 == len(validation_pairs):
            print(f"Processed {image_index + 1}/{len(validation_pairs)} images")

    report = evaluator.report()

    print()
    print("Segmentation validation results")
    print("-------------------------------")

    for class_name, iou in report["iou_per_class"].items():
        print(f"{class_name}: IoU = {iou:.4f}")

    print("-------------------------------")
    print(f"Cone mIoU: {report['mIoU']:.4f}")
    print(f"Visualizations saved in: {VISUALIZATION_DIR}")


if __name__ == "__main__":
    main()