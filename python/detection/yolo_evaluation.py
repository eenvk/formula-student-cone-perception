import tensorflow as tf
from keras_cv import bounding_box

from detection.detection_config import CHECKPOINT_PATH, SEED
from detection.data_pipeline import build_datasets
from detection.model_utils import create_model

from dataset.dataset_utils import (
    Box,
    detection_id_to_segmentation_id,
    prediction_to_box
)

from evaluation.evaluation_utils import (
    DetectionEvaluator,
    ClassificationEvaluator,
)

def predictions_to_boxes(
    boxes,
    classes,
    scores,
    image_width,
    image_height,
):
    result = []

    for bbox, class_id, score in zip(
        boxes,
        classes,
        scores,
    ):
        class_id = int(class_id)

        if class_id == 1:
            class_id = 0
        elif class_id == 0:
            class_id = 1
        
        # Checkpoint has 5 outputs, but only
        # detection classes 0-3 are evaluated.
        if class_id not in (0, 1, 2, 3):
            continue

        x_min, y_min, x_max, y_max = [
            float(value)
            for value in bbox
        ]

        # Clip predictions to image boundaries.
        x_min = max(0.0, min(x_min, image_width))
        y_min = max(0.0, min(y_min, image_height))
        x_max = max(0.0, min(x_max, image_width))
        y_max = max(0.0, min(y_max, image_height))

        # prediction_to_box() rounds to integers.
        # Check the rounded box first to avoid zero-area boxes.
        rx_min = int(round(x_min))
        ry_min = int(round(y_min))
        rx_max = int(round(x_max))
        ry_max = int(round(y_max))

        if rx_max <= rx_min or ry_max <= ry_min:
            continue

        result.append(
            prediction_to_box(
                [rx_min, ry_min, rx_max, ry_max],
                class_id,
                score,
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

        image_height = int(images.shape[1])
        image_width = int(images.shape[2])

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
                image_width,
                image_height
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

    model = create_model(num_classes=5)

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