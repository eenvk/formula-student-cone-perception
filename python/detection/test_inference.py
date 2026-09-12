from pathlib import Path

import cv2
import tensorflow as tf

from dataset.dataset_utils import (
    CLASS_ID_TO_NAME,
    get_segmentation_test_pairs,
)

from detection.data_pipeline import (
    build_inference_dataset,
    build_inference_metadata,
)

from detection.detection_config import (
    CHECKPOINT_PATH,
    SEED,
)

from detection.model_utils import create_model
from detection.inference import postprocess_inference_dataset



def draw_boxes(image, boxes):
    result = image.copy()

    for box in boxes:
        label = CLASS_ID_TO_NAME[box.class_id]

        if box.score is not None:
            label += f" {box.score:.2f}"

        cv2.rectangle(
            result,
            (box.x_min, box.y_min),
            (box.x_max, box.y_max),
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            label,
            (box.x_min, max(20, box.y_min - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return result


def prepare_inference_data(pairs):
    image_paths = []
    image_shapes = []

    for image_path, _ in pairs:
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Could not load image: {image_path}")

        image_height, image_width = image.shape[:2]

        image_paths.append(str(image_path))
        image_shapes.append([image_height, image_width])

    image_paths = tf.constant(image_paths, dtype=tf.string)
    image_shapes = tf.constant(image_shapes, dtype=tf.int32)

    return image_paths, image_shapes


def main():
    tf.keras.utils.set_random_seed(SEED)

    # Load model
    model = create_model()
    model.load_weights(str(CHECKPOINT_PATH))

    # Get test images
    pairs = get_segmentation_test_pairs()

    print("Test images:", len(pairs))

    image_paths, image_shapes = prepare_inference_data(pairs)

    # Build inference dataset and metadata
    inference_ds = build_inference_dataset(image_paths)
    inference_metadata = build_inference_metadata(image_shapes)

    print("Inference views:", len(inference_metadata))

    # Single model.predict() call
    raw_predictions = model.predict(inference_ds)

    predictions = postprocess_inference_dataset(
        raw_predictions,
        inference_metadata
    )

    if len(predictions) != len(pairs):
        raise ValueError(
            f"Predictions/images mismatch: "
            f"{len(predictions)} prediction groups for "
            f"{len(pairs)} images."
        )

    # Draw and save predictions
    output_dir = Path("evaluation/patch_inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_index, (image_path, _) in enumerate(pairs):
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Could not load image: {image_path}")

        boxes = predictions[image_index]

        print(
            f"[{image_index + 1}/{len(pairs)}] "
            f"{image_path.name} | "
            f"Predictions: {len(boxes)}"
        )

        output = draw_boxes(image, boxes)

        output_path = output_dir / f"{image_path.stem}_prediction.jpg"
        cv2.imwrite(str(output_path), output)

    print()
    print("Prediction images saved in:", output_dir)


if __name__ == "__main__":
    main()