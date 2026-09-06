from pathlib import Path

import cv2
import tensorflow as tf

from dataset.dataset_utils import (
    CLASS_ID_TO_NAME,
    get_segmentation_test_pairs,
)

from detection.detection_config import (
    CHECKPOINT_PATH,
    SEED,
)

from detection.model_utils import create_model
from detection.inference import predict_boxes


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


def main():
    tf.keras.utils.set_random_seed(SEED)

    # Load model
    model = create_model()
    model.load_weights(str(CHECKPOINT_PATH))

    # Get test images
    pairs = get_segmentation_test_pairs()

    output_dir = Path("evaluation/patch_inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, (image_path, _) in enumerate(pairs):

        print(f"\n[{index + 1}/{len(pairs)}] {image_path.name}")

        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not load image, skipping.")
            continue

        boxes = predict_boxes(image, model)

        print("Predictions:", len(boxes))

        output = draw_boxes(image, boxes)

        output_path = output_dir / f"{image_path.stem}_prediction.jpg"

        cv2.imwrite(str(output_path), output)

    print("Testing image:")
    print(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Cannot load image: {image_path}")

    print("Original image shape:", image.shape)

    # Patch-based inference
    boxes = predict_boxes(image, model)

    print()
    print("Predictions:", len(boxes))

    for box in boxes:
        print(
            CLASS_ID_TO_NAME[box.class_id],
            (box.x_min, box.y_min, box.x_max, box.y_max),
            f"score={box.score:.3f}"
        )

    # Draw predictions on ORIGINAL image
    output = draw_boxes(image, boxes)


    output_path = output_dir / f"{image_path.stem}_prediction.jpg"

    cv2.imwrite(str(output_path), output)

    print()
    print("Saved prediction image:")
    print(output_path)


if __name__ == "__main__":
    main()