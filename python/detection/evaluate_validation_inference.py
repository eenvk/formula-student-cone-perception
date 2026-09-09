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

from detection.data_pipeline import (
    prepare_dataset_data,
    build_inference_dataset,
    build_inference_metadata,
)

from detection.inference import (
    predict_inference_dataset,
)

from detection.model_utils import create_model
from detection.inference import predict_boxes

from evaluation.evaluation_utils import (
    DetectionEvaluator,
    ClassificationEvaluator,
)


def evaluate_model(model, validation_pairs, limit=None):

    detection_evaluator = DetectionEvaluator()

    classification_evaluator = ClassificationEvaluator(
        iou_threshold=0.5
    )

    if limit is not None:
        validation_pairs = validation_pairs[:limit]

    total_gt = 0
    total_predictions = 0

    # ======================================================
    # 1. BUILD DATASET
    # ======================================================

    (
        image_paths,
        _,
        _,
        image_shapes,
    ) = prepare_dataset_data(validation_pairs)

    inference_ds = build_inference_dataset(
        image_paths
    )

    metadata = build_inference_metadata(
        image_shapes
    )

    print("Images:", len(validation_pairs))
    print("Views:", len(metadata))

    # ======================================================
    # 2. INFERENCE - UNA SOLA VOLTA
    # ======================================================

    predictions = predict_inference_dataset(
        model,
        inference_ds,
        metadata
    )

    # ======================================================
    # 3. EVALUATION
    # ======================================================

    for index, (image_path, annotation_path) in enumerate(validation_pairs):

        print(
            f"\n[{index + 1}/{len(validation_pairs)}] "
            f"{image_path.name}"
        )

        image_bgr = load_image(image_path)

        image_height, image_width = image_bgr.shape[:2]

        gt_boxes = annotation_to_instances(
            annotation_path,
            image_height,
            image_width,
        )

        # NON facciamo più predict qui
        pred_boxes = predictions[index]

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