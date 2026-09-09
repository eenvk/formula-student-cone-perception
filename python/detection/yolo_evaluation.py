import argparse
import tensorflow as tf

from dataset.dataset_utils import (
    get_segmentation_test_pairs,
    annotation_to_instances,
    load_image as load_cv_image,
    prediction_to_box,
)

from detection.inference import (
    predict_boxes,
    predict_image_with_patches,
    predict_total_image,
    final_nms,
)

from detection.detection_config import (
    CHECKPOINT_PATH,
    SEED,
)

from detection.model_utils import create_model

from evaluation.evaluation_utils import (
    ClassificationEvaluator,
    DetectionEvaluator,
)


def evaluate_model(model, pairs, mode):

    detection_evaluator = DetectionEvaluator()
    classification_evaluator = ClassificationEvaluator(iou_threshold=0.5)

    total_gt = 0
    total_predictions = 0

    for index, (image_path, annotation_path) in enumerate(pairs):

        print(f"[{index + 1}/{len(pairs)}] {image_path.name}")

        image_bgr = load_cv_image(image_path)
        image_height, image_width = image_bgr.shape[:2]

        gt_boxes = annotation_to_instances(
            annotation_path,
            image_height,
            image_width,
            bitmap_only=True
        )

        if mode == "combined":
            pred_boxes = predict_boxes(image_bgr, model)

        elif mode == "separate":
            image_rgb = image_bgr[:, :, ::-1]

            patch_predictions = predict_image_with_patches(model, image_rgb)
            total_predictions_image = predict_total_image(model, image_rgb)

            predictions = final_nms(
                patch_predictions,
                total_predictions_image,
                image_width,
                image_height
            )

            pred_boxes = [
                prediction_to_box(
                    prediction["bbox"],
                    prediction["class_id"],
                    prediction["score"],
                )
                for prediction in predictions
            ]

        else:
            raise ValueError(f"Unknown inference mode: {mode}")

        total_gt += len(gt_boxes)
        total_predictions += len(pred_boxes)

        print(
            f"  GT: {len(gt_boxes)} | "
            f"Predictions: {len(pred_boxes)}"
        )

        detection_evaluator.update(gt_boxes, pred_boxes)
        classification_evaluator.update(gt_boxes, pred_boxes)

    print()
    print("Total GT:", total_gt)
    print("Total predictions:", total_predictions)

    return (
        detection_evaluator.report(),
        classification_evaluator.report()
    )


def print_report(detection_report, classification_report, mode):

    print()
    print("=" * 40)
    print(f"INFERENCE MODE: {mode.upper()}")
    print("=" * 40)

    print()
    print("DETECTION EVALUATION")
    print("--------------------")

    for class_name, ap in detection_report["AP@0.5:0.95_per_class"].items():
        print(f"{class_name:20s}: {ap:.4f}")

    print(
        "mAP@0.5:0.95:",
        f"{detection_report['mAP@0.5:0.95']:.4f}"
    )

    print()
    print("CLASSIFICATION EVALUATION")
    print("-------------------------")

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

    print(
        "\nMacro F1:",
        f"{classification_report['macro_f1']:.4f}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["combined", "separate", "both"],
        default="combined",
        help=(
            "combined: patch e full image nella stessa model.predict(); "
            "separate: due inference separate; "
            "both: esegue entrambe"
        )
    )

    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)

    pairs = get_segmentation_test_pairs()

    print()
    print("Test image/annotation pairs found:", len(pairs))

    model = create_model()
    model.load_weights(str(CHECKPOINT_PATH))

    modes = (
        ["separate", "combined"]
        if args.mode == "both"
        else [args.mode]
    )

    for mode in modes:

        print()
        print("=" * 40)
        print(f"STARTING {mode.upper()} INFERENCE")
        print("=" * 40)

        detection_report, classification_report = evaluate_model(
            model,
            pairs,
            mode
        )

        print_report(
            detection_report,
            classification_report,
            mode
        )


if __name__ == "__main__":
    main()