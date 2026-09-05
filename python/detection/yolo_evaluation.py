import keras_cv
import tensorflow as tf

from dataset.dataset_utils import (
    Box,
    DETECTION_ID_TO_NAME,
    detection_id_to_segmentation_id,
    get_segmentation_test_pairs,
    prediction_to_box,
    load_cv_image,
    annotation_to_instances,
    segmentation_id_to_detection_id,
)
from detection.data_pipeline import (
    dict_to_tuple,
    load_dataset,
)
from detection.detection_config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    IMAGE_SIZE,
    SEED,
)
from detection.model_utils import create_model
from evaluation.evaluation_utils import (
    ClassificationEvaluator,
    DetectionEvaluator,
)


def predictions_to_boxes(boxes, classes, scores, image_width, image_height):
    """Convert model output into validated shared Box instances."""

    result = []

    for bbox, class_id, score in zip(boxes, classes, scores):
        class_id = int(class_id)

        if class_id not in DETECTION_ID_TO_NAME:
            continue

        x_min, y_min, x_max, y_max = [float(value) for value in bbox]
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
    """Convert resized detection targets into shared Box instances."""

    result = []

    for bbox, detection_class_id in zip(boxes, classes):
        detection_class_id = int(detection_class_id)

        if detection_class_id not in DETECTION_ID_TO_NAME:
            continue

        x_min, y_min, x_max, y_max = [
            int(round(float(value)))
            for value in bbox
        ]

        if x_max <= x_min or y_max <= y_min:
            continue

        result.append(
            Box(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                class_id=detection_id_to_segmentation_id(detection_class_id),
            )
        )

    return result


def evaluate_model(model, test_ds):
    """Evaluate detection and classification metrics over a test dataset."""

    detection_evaluator = DetectionEvaluator()
    classification_evaluator = ClassificationEvaluator(iou_threshold=0.5)

    # the model takes each image from the set and it predicts the classification
    # and the detection
    for images, y_true in test_ds:
        y_pred = model.predict(images, verbose=0)
        image_height = int(images.shape[1])
        image_width = int(images.shape[2])

        y_true = keras_cv.bounding_box.to_ragged(y_true)
        y_pred = keras_cv.bounding_box.to_ragged(y_pred)
        # y_true and y_pred are ragged tensors with the following structure:
        # {
        #     "boxes": tensor of shape (num_boxes, 4) with [x_min, y_min, x_max, y_max],
        #     "classes": tensor of shape (num_boxes,) with class IDs,
        #     "confidence": tensor of shape (num_boxes,) with confidence scores (y_pred only)
        # }
        # each element in the batch can be accessed by index: y_true["boxes"][i]
        # each image have its number of cones with corrisponding boxes and classes

        for i in range(images.shape[0]):
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

            detection_evaluator.update(gt_boxes, pred_boxes)
            classification_evaluator.update(gt_boxes, pred_boxes)

    return (
        detection_evaluator.report(),
        classification_evaluator.report(),
    )


def build_test_dataset():
    """Build test targets from segmentation bitmaps via dataset_utils."""

    pairs = get_segmentation_test_pairs()
    print()
    print("Test image/annotation pairs found:", len(pairs))

    image_paths = []
    all_boxes = []
    all_classes = []
    total_objects = 0
    print("Reading test annotations and deriving bounding boxes...")

    for image_path, annotation_path in pairs:

        # Load original image only to know its dimensions
        image = load_cv_image(image_path)

        image_height, image_width = image.shape[:2]

        # This function:
        # bitmap -> binary mask -> tight bounding box
        #
        # class IDs are already the SHARED segmentation IDs:
        #
        # 1 = yellow
        # 2 = blue
        # 3 = small orange
        # 4 = big orange
        #
        # unknown (255) is ignored automatically.
        instances = annotation_to_instances(
            annotation_path,
            image_height,
            image_width,
            bitmap_only=True,
        )

        boxes = [
            [
                instance.x_min,
                instance.y_min,
                instance.x_max,
                instance.y_max,
            ]
            for instance in instances
        ]

        classes = [
            segmentation_id_to_detection_id(instance.class_id)
            for instance in instances
        ]

        image_paths.append(str(image_path))
        all_boxes.append(boxes)
        all_classes.append(classes)

        total_objects += len(instances)

    print("Test images:", len(image_paths))
    print("GT cones derived from masks:", total_objects)

    bbox = tf.ragged.constant(
        all_boxes,
        dtype=tf.float32,
        ragged_rank=1,
    )

    classes = tf.ragged.constant(
        all_classes,
        dtype=tf.float32,
        ragged_rank=1,
    )

    image_paths = tf.constant(
        image_paths,
        dtype=tf.string,
    )

    data = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            classes,
            bbox,
        )
    )

    # Same image loading used for YOLO training/validation.
    data = data.map(
        load_dataset,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    data = data.ragged_batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    # IMPORTANT:
    # exactly the same deterministic resizing used for validation.
    #
    # KerasCV also transforms the GT bounding boxes accordingly.
    test_resizing = keras_cv.layers.JitteredResize(
        target_size=IMAGE_SIZE,
        scale_factor=(1.0, 1.0),
        bounding_box_format="xyxy",
    )
    data = data.map(
        test_resizing,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    data = data.map(
        dict_to_tuple,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return data.prefetch(tf.data.AUTOTUNE)


def main():
    tf.keras.utils.set_random_seed(SEED)
    test_ds = build_test_dataset()

    model = create_model()
    model.load_weights(str(CHECKPOINT_PATH))

    detection_report, classification_report = evaluate_model(
        model,
        test_ds,
    )

    print()
    print("==============================")
    print("DETECTION EVALUATION")
    print("==============================")

    for class_name, ap in detection_report[
        "AP@0.5:0.95_per_class"
    ].items():
        print(f"{class_name:20s}: {ap:.4f}")

    print(
        "mAP@0.5:0.95:",
        f"{detection_report['mAP@0.5:0.95']:.4f}",
    )

    print()
    print("==============================")
    print("CLASSIFICATION EVALUATION")
    print("==============================")

    for class_name, f1 in classification_report[
        "f1_per_class"
    ].items():
        print(f"{class_name:20s}: {f1:.4f}")

    print(
        "Macro F1:",
        f"{classification_report['macro_f1']:.4f}",
    )


if __name__ == "__main__":
    main()
