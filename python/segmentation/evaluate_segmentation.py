#Novkovic

"""
Evaluate the complete end-to-end segmentation pipeline on the test set.

For each test image, the file runs YOLO detection, uses the predicted bounding boxes as input for the U-Net segmentation model,
compares the predicted semantic mask with the ground-truth mask,
computes the IoU metrics, and saves a number of visual comparisons.
"""

import cv2
import numpy as np
import tensorflow as tf

from dataset.dataset_utils import PROJECT_ROOT, annotation_to_semantic_mask, create_overlay, get_segmentation_test_pairs, load_annotation, load_image
from detection.data_pipeline import build_inference_dataset, build_inference_metadata
from detection.detection_config import CHECKPOINT_PATH, SEED
from detection.inference import predict_inference_dataset
from detection.model_utils import create_model
from evaluation.evaluation_utils import SegmentationEvaluator
from segmentation.predict_segmentation import predict_segmentation
from segmentation.segmentation_model import build_unet


MODEL_DIR = PROJECT_ROOT / "models"
SEGMENTATION_WEIGHTS_PATH = MODEL_DIR / "unet_best.weights.h5"

VISUALIZATION_DIR = MODEL_DIR / "segmentation_test"



def prepare_inference_data(pairs):
    """Prepares image paths and original image sizes for batched detector inference."""
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

def main():

    tf.keras.utils.set_random_seed(SEED)

    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    detector_model = create_model()
    detector_model.load_weights(str(CHECKPOINT_PATH))

    segmentation_model = build_unet()
    segmentation_model.load_weights(SEGMENTATION_WEIGHTS_PATH)

    evaluator = SegmentationEvaluator()

    pairs = get_segmentation_test_pairs()

    print(f"Test images: {len(pairs)}")

    image_paths, image_shapes = prepare_inference_data(pairs)

    inference_ds = build_inference_dataset(image_paths)
    inference_metadata = build_inference_metadata(image_shapes)

    all_predicted_boxes = predict_inference_dataset(detector_model, inference_ds, inference_metadata)

    for image_index, (image_path, annotation_path) in enumerate(pairs):
        image_bgr = load_image(image_path)
        image_height, image_width = image_bgr.shape[:2]

        gt_mask = annotation_to_semantic_mask(annotation_path, image_height, image_width)

        predicted_boxes = all_predicted_boxes[image_index]
        predicted_mask = predict_segmentation(image_bgr, predicted_boxes, segmentation_model)

        evaluator.update(gt_mask, predicted_mask)


        gt_overlay = create_overlay(image_bgr, gt_mask)
        predicted_overlay = create_overlay(image_bgr, predicted_mask)

        comparison = np.hstack((image_bgr, gt_overlay, predicted_overlay))

        output_path = VISUALIZATION_DIR / f"{image_index:03d}_{image_path.stem}.jpg"
        cv2.imwrite(str(output_path), comparison)

        if (image_index + 1) % 50 == 0 or image_index + 1 == len(pairs):
            print(f"Processed {image_index + 1}/{len(pairs)} images")

    report = evaluator.report()

    print()
    print("End-to-end segmentation results")
    print("--------------------------------")

    for class_name, iou in report["iou_per_class"].items():
        print(f"{class_name}: IoU = {iou:.4f}")

    print("--------------------------------")
    print(f"Cone mIoU: {report['mIoU']:.4f}")
    print(f"Visualizations saved in: {VISUALIZATION_DIR}")

if __name__ == "__main__":
    main()