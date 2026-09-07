import tensorflow as tf

from dataset.dataset_utils import (
    get_segmentation_test_pairs,
    annotation_to_instances,
    load_image as load_cv_image,
)

from detection.inference import predict_boxes

from detection.detection_config import (
    CHECKPOINT_PATH,
    SEED,
)

from detection.model_utils import create_model

from evaluation.evaluation_utils import (
    ClassificationEvaluator,
    DetectionEvaluator,
)


def evaluate_model(model, pairs):

    detection_evaluator = DetectionEvaluator()
    classification_evaluator = ClassificationEvaluator(
        iou_threshold=0.5
    )

    total_gt = 0
    total_predictions = 0

    for index, (image_path, annotation_path) in enumerate(pairs):

        print(
            f"[{index + 1}/{len(pairs)}] "
            f"{image_path.name}"
        )

        # Load original image
        image_bgr = load_cv_image(image_path)

        image_height, image_width = image_bgr.shape[:2]

        # Ground truth boxes from segmentation masks
        gt_boxes = annotation_to_instances(
            annotation_path,
            image_height,
            image_width,
            bitmap_only=True
        )

        # Predictions using patch-based inference
        pred_boxes = predict_boxes(image_bgr, model)

        total_gt += len(gt_boxes)
        total_predictions += len(pred_boxes)

        print(
            f"  GT: {len(gt_boxes)} | "
            f"Predictions: {len(pred_boxes)}"
        )

        detection_evaluator.update(
            gt_boxes,
            pred_boxes
        )

        classification_evaluator.update(
            gt_boxes,
            pred_boxes
        )

    print()
    print("Total GT:", total_gt)
    print("Total predictions:", total_predictions)

    return (
        detection_evaluator.report(),
        classification_evaluator.report()
    )


def main():

    tf.keras.utils.set_random_seed(SEED)

    pairs = get_segmentation_test_pairs()

    print()
    print(
        "Test image/annotation pairs found:",
        len(pairs)
    )

    model = create_model()
    model.load_weights(
        str(CHECKPOINT_PATH)
    )

    detection_report, classification_report = evaluate_model(
        model,
        pairs
    )

    print()
    print("==============================")
    print("DETECTION EVALUATION")
    print("==============================")

    for class_name, ap in detection_report[
        "AP@0.5:0.95_per_class"
    ].items():

        print(
            f"{class_name:20s}: "
            f"{ap:.4f}"
        )

    print(
        "mAP@0.5:0.95:",
        f"{detection_report['mAP@0.5:0.95']:.4f}"
    )

    print()
    print("==============================")
    print("CLASSIFICATION EVALUATION")
    print("==============================")

    for class_name in classification_report["f1_per_class"]:

        precision = classification_report["precision_per_class"][class_name]
        recall = classification_report["recall_per_class"][class_name]
        f1 = classification_report["f1_per_class"][class_name]

        print(
            f"{class_name:20s}: "
            f"Precision={precision:.4f}  "
            f"Recall={recall:.4f}  "
            f"F1={f1:.4f}"
        )

    print("\nMacro F1:", f"{classification_report['macro_f1']:.4f}")

if __name__ == "__main__":
    main()