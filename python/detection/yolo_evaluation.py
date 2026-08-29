import tensorflow as tf
from keras_cv import bounding_box

from detection_config import CHECKPOINT_PATH, SEED
from data_pipeline import build_datasets
from model_utils import create_model

from dataset.dataset_utils import (
    Box,
    detection_id_to_segmentation_id,
    prediction_to_box
)

from evaluation.evaluation_utils import (
    DetectionEvaluator,
    ClassificationEvaluator,
)

def predictions_to_boxes(boxes, classes, scores):
    return [
        prediction_to_box(bbox, class_id, score)
        for bbox, class_id, score in zip(
            boxes,
            classes,
            scores,
        )
    ]

def ground_truth_to_boxes(boxes, classes):
    result = []

    for bbox, class_id in zip(boxes, classes):
        detection_id = int(class_id)

        shared_id = detection_id_to_segmentation_id(
            detection_id
        )

        x_min, y_min, x_max, y_max = [
            float(value)
            for value in bbox
        ]

        result.append(
            Box(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                class_id=shared_id,
                score=None,
            )
        )

    return result

def ground_truth_to_boxes(boxes, classes):
    result = []

    for bbox, class_id in zip(boxes, classes):
        detection_id = int(class_id)

        shared_id = detection_id_to_segmentation_id(
            detection_id
        )

        x_min, y_min, x_max, y_max = [
            float(value)
            for value in bbox
        ]

        result.append(
            Box(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                class_id=shared_id,
                score=None,
            )
        )

    return result

def evaluate_model(model, val_ds):
    detection_evaluator = DetectionEvaluator()

    classification_evaluator = ClassificationEvaluator(
        iou_threshold=0.5
    )

    for images, y_true in val_ds:

        y_pred = model.predict(
            images,
            verbose=0,
        )

        y_true = bounding_box.to_ragged(
            y_true
        )

        y_pred = bounding_box.to_ragged(
            y_pred
        )

        batch_size = images.shape[0]

        for i in range(batch_size):

            gt_boxes = ground_truth_to_boxes(
                y_true["boxes"][i].numpy(),
                y_true["classes"][i].numpy(),
            )

            pred_boxes = predictions_to_boxes(
                y_pred["boxes"][i].numpy(),
                y_pred["classes"][i].numpy(),
                y_pred["confidence"][i].numpy(),
            )

            detection_evaluator.update(
                gt_boxes,
                pred_boxes,
            )

            classification_evaluator.update(
                gt_boxes,
                pred_boxes,
            )

    return (
        detection_evaluator.report(),
        classification_evaluator.report(),
    )

def main():

    tf.keras.utils.set_random_seed(SEED)

    _, val_ds = build_datasets()

    model = create_model()

    model.load_weights(
        str(CHECKPOINT_PATH)
    )

    detection_report, classification_report = (
        evaluate_model(
            model,
            val_ds,
        )
    )

    print("\n==============================")
    print("DETECTION EVALUATION")
    print("==============================")

    for class_name, ap in detection_report[
        "AP@0.5:0.95_per_class"
    ].items():
        print(
            f"{class_name:20s}: {ap:.4f}"
        )

    print(
        "\nmAP@0.5:0.95:",
        f"{detection_report['mAP@0.5:0.95']:.4f}",
    )

    print("\n==============================")
    print("CLASSIFICATION EVALUATION")
    print("==============================")

    for class_name, f1 in classification_report[
        "f1_per_class"
    ].items():
        print(
            f"{class_name:20s}: {f1:.4f}"
        )

    print(
        "\nMacro F1:",
        f"{classification_report['macro_f1']:.4f}",
    )


if __name__ == "__main__":
    main()