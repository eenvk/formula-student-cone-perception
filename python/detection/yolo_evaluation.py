import tensorflow as tf

from dataset.dataset_utils import (
    get_segmentation_test_pairs,
    annotation_to_instances,
    load_image as load_cv_image,
    prediction_to_box,
)

from detection.data_pipeline import (
    build_inference_dataset,
    build_inference_metadata,
)

from detection.inference import (
    postprocess_inference_dataset,
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


def prepare_test_data(pairs):
    image_paths = []
    image_shapes = []
    y_true = []

    for image_path, annotation_path in pairs:
        image = load_cv_image(image_path)

        image_height, image_width = image.shape[:2]

        gt_boxes = annotation_to_instances(
            annotation_path,
            image_height,
            image_width,
            bitmap_only=True
        )

        image_paths.append(str(image_path))
        image_shapes.append([
            image_height,
            image_width
        ])

        y_true.append(gt_boxes)

    image_paths = tf.constant(
        image_paths,
        dtype=tf.string
    )

    image_shapes = tf.constant(
        image_shapes,
        dtype=tf.int32
    )

    return image_paths, image_shapes, y_true


def build_test_dataset():
    pairs = get_segmentation_test_pairs()

    print()
    print("Test image/annotation pairs found:", len(pairs))

    image_paths, image_shapes, y_true = prepare_test_data(
        pairs
    )

    inference_ds = build_inference_dataset(
        image_paths
    )

    inference_metadata = build_inference_metadata(
        image_shapes
    )

    print("Inference metadata:", len(inference_metadata))

    return (
        pairs,
        inference_ds,
        inference_metadata,
        y_true
    )


def run_inference(
    model,
    inference_ds,
    inference_metadata
):
    raw_predictions = model.predict(
        inference_ds
    )

    predictions = postprocess_inference_dataset(
        raw_predictions,
        inference_metadata,
    )

    return predictions


def evaluate_model(
    predictions,
    y_true,
    pairs
):
    detection_evaluator = DetectionEvaluator()

    classification_evaluator = ClassificationEvaluator(
        iou_threshold=0.5
    )

    total_gt = 0
    total_predictions = 0

    num_images = len(predictions)

    if num_images != len(y_true):
        raise ValueError(
            f"Predictions/GT mismatch: "
            f"{num_images} prediction groups for "
            f"{len(y_true)} images."
        )

    for image_index in range(num_images):
        image_path, _ = pairs[image_index]

        gt_boxes = y_true[image_index]
        image_predictions = predictions[image_index]

        pred_boxes = [
            prediction_to_box(
                prediction["bbox"],
                prediction["class_id"],
                prediction["score"],
            )
            for prediction in image_predictions
        ]

        total_gt += len(gt_boxes)
        total_predictions += len(pred_boxes)

        print(
            f"[{image_index + 1}/{num_images}] "
            f"{image_path.name} | "
            f"GT: {len(gt_boxes)} | "
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


def print_report(detection_report, classification_report):

    print()
    print("=" * 40)
    print("TEST EVALUATION")
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
    tf.keras.utils.set_random_seed(SEED)

    (
        pairs,
        inference_ds,
        inference_metadata,
        y_true
    ) = build_test_dataset()

    model = create_model()

    model.load_weights(
        str(CHECKPOINT_PATH)
    )

    predictions = run_inference(
        model,
        inference_ds,
        inference_metadata
    )

    detection_report, classification_report = evaluate_model(
        predictions,
        y_true,
        pairs
    )

    print_report(
        detection_report,
        classification_report
    )


if __name__ == "__main__":
    main()