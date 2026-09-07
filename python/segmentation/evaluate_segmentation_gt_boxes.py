# novkovic

"""Evaluate segmentation on the test set using ground-truth bounding boxes."""

import cv2
import numpy as np

from dataset.dataset_utils import PROJECT_ROOT, annotation_to_instances, annotation_to_semantic_mask, create_overlay, get_segmentation_test_pairs, load_image
from evaluation.evaluation_utils import SegmentationEvaluator
from segmentation.predict_segmentation import predict_segmentation
from segmentation.segmentation_model import build_unet


MODEL_DIR = PROJECT_ROOT / "models"
SEGMENTATION_WEIGHTS_PATH = MODEL_DIR / "unet_best.weights.h5"

VISUALIZATION_DIR = MODEL_DIR / "segmentation_test_gt_boxes"
NUM_VISUALIZATIONS = 24


def main():

    if not SEGMENTATION_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Segmentation weights not found: {SEGMENTATION_WEIGHTS_PATH}")

    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    segmentation_model = build_unet()
    segmentation_model.load_weights(SEGMENTATION_WEIGHTS_PATH)

    evaluator = SegmentationEvaluator()

    pairs = get_segmentation_test_pairs()

    print(f"Test images: {len(pairs)}")
    print(f"Loading segmentation weights from: {SEGMENTATION_WEIGHTS_PATH}")

    for image_index, (image_path, annotation_path) in enumerate(pairs):
        image_bgr = load_image(image_path)
        image_height, image_width = image_bgr.shape[:2]

        gt_boxes = annotation_to_instances(annotation_path, image_height, image_width, bitmap_only=True)
        gt_mask = annotation_to_semantic_mask(annotation_path, image_height, image_width)

        predicted_mask = predict_segmentation(image_bgr, gt_boxes, segmentation_model)

        evaluator.update(gt_mask, predicted_mask)

        if image_index < NUM_VISUALIZATIONS:
            gt_overlay = create_overlay(image_bgr, gt_mask)
            predicted_overlay = create_overlay(image_bgr, predicted_mask)

            comparison = np.hstack((image_bgr, gt_overlay, predicted_overlay))

            output_path = VISUALIZATION_DIR / f"{image_index:03d}_{image_path.stem}.jpg"
            cv2.imwrite(str(output_path), comparison)

        print(f"Processed {image_index + 1}/{len(pairs)} images")

    report = evaluator.report()

    print()
    print("Segmentation results with GT bounding boxes")
    print("-------------------------------------------")

    for class_name, iou in report["iou_per_class"].items():
        print(f"{class_name}: IoU = {iou:.4f}")

    print("-------------------------------------------")
    print(f"Cone mIoU: {report['mIoU']:.4f}")

    print()
    print("Pixel confusion matrix")
    print(evaluator.confusion_matrix)

    print()
    print(f"Visualizations saved in: {VISUALIZATION_DIR}")


if __name__ == "__main__":
    main()