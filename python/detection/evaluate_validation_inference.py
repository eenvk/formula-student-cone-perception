import argparse
import time
import tensorflow as tf

from dataset.dataset_utils import (
    get_bounding_boxes_train_pairs,
    train_validation_split,
    annotation_to_instances,
    load_image,
)

from detection.detection_config import (
    CHECKPOINT_PATH,
    SEED,
    SPLIT_RATIO,
)

from detection.model_utils import create_model
from detection.inference import predict_boxes

from evaluation.evaluation_utils import (
    DetectionEvaluator,
    ClassificationEvaluator,
)


def evaluate_model(model, validation_pairs, limit=None):
    """
    Evaluate the complete inference pipeline on the validation split.

    Pipeline:
        original image
        -> create patch views + full-image view
        -> single model.predict()
        -> patch post-processing
        -> full-image post-processing
        -> patch NMS
        -> final patch-vs-full-image NMS
        -> detection/classification evaluation
    """

    detection_evaluator = DetectionEvaluator()

    classification_evaluator = ClassificationEvaluator(
        iou_threshold=0.5
    )

    if limit is not None:
        validation_pairs = validation_pairs[:limit]

    total_gt = 0
    total_predictions = 0

    for index, (image_path, annotation_path) in enumerate(validation_pairs):

        print(
            f"\n[{index + 1}/{len(validation_pairs)}] "
            f"{image_path.name}"
        )

        # Original image
        image_bgr = load_image(image_path)

        image_height, image_width = image_bgr.shape[:2]

        # Ground truth in original-image coordinates
        gt_boxes = annotation_to_instances(
            annotation_path,
            image_height,
            image_width,
        )

        # Complete combined inference.
        #
        # predict_boxes() performs:
        # 1. BGR -> RGB conversion
        # 2. patch creation
        # 3. full-image letterbox creation
        # 4. one model.predict() over all views
        # 5. patch post-processing
        # 6. full-image post-processing
        # 7. patch-aware NMS
        # 8. final patch-vs-full-image NMS
        # 9. conversion to shared Box representation
        pred_boxes = predict_boxes(
            image_bgr,
            model
        )

        total_gt += len(gt_boxes)
        total_predictions += len(pred_boxes)

        print(
            f"GT: {len(gt_boxes)} | "
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

    return (
        detection_evaluator.report(),
        classification_evaluator.report(),
        total_gt,
        total_predictions,
    )


def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a YOLO model on the validation split "
            "using the combined patch + full-image inference pipeline."
        )
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=str(CHECKPOINT_PATH),
        help="Path to model weights."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N validation images."
    )

    args = parser.parse_args()

    # Reproduce the validation split used during training.
    tf.keras.utils.set_random_seed(SEED)

    print("\n===================================")
    print("BUILDING VALIDATION SPLIT")
    print("===================================")

    pairs = get_bounding_boxes_train_pairs()

    train_pairs, validation_pairs = train_validation_split(
        pairs,
        SPLIT_RATIO,
        SEED
    )

    print("Total dataset images:", len(pairs))
    print("Training images:", len(train_pairs))
    print("Validation images:", len(validation_pairs))

    if args.limit is not None:
        print("Evaluation limited to:", args.limit)

    # ------------------------------------------------------
    # MODEL
    # ------------------------------------------------------

    print("\n===================================")
    print("LOADING MODEL")
    print("===================================")

    model = create_model()
    model.load_weights(args.weights)

    print("Weights:", args.weights)

    # ------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------

    print("\n===================================")
    print("STARTING COMBINED VALIDATION INFERENCE")
    print("===================================")

    (
        detection_report,
        classification_report,
        total_gt,
        total_predictions,
    ) = evaluate_model(
        model,
        validation_pairs,
        limit=args.limit,
    )

    # ------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------

    print("\n===================================")
    print("DETECTION RESULTS")
    print("===================================")

    for class_name, ap in detection_report[
        "AP@0.5:0.95_per_class"
    ].items():

        print(
            f"{class_name:20s}: "
            f"{ap:.4f}"
        )

    print(
        "\nmAP@0.5:0.95:",
        f"{detection_report['mAP@0.5:0.95']:.4f}"
    )

    print("\n===================================")
    print("CLASSIFICATION RESULTS")
    print("===================================")

    for class_name, f1 in classification_report[
        "f1_per_class"
    ].items():

        print(
            f"{class_name:20s}: "
            f"{f1:.4f}"
        )

    print(
        "\nMacro F1:",
        f"{classification_report['macro_f1']:.4f}"
    )

    # ------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------

    elapsed_time = time.perf_counter() - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print("\n===================================")
    print("SUMMARY")
    print("===================================")

    print("Ground-truth cones:", total_gt)
    print("Predicted cones:", total_predictions)

    print(
        f"Total time: "
        f"{int(minutes)} min {seconds:.2f} s"
    )


if __name__ == "__main__":
    main()