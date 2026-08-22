from pathlib import Path

import cv2
import numpy as np

from dataset.dataset_utils import annotation_to_semantic_mask, create_overlay, find_all_dataset_pairs, train_validation_split
from segmentation.segmentation_model import build_unet


IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

SAMPLE_INDEX = 0


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TRAIN_DATASET_DIR = PROJECT_ROOT / "dataset" / "fsoco_segmentation_train"

WEIGHTS_PATH = PROJECT_ROOT / "models" / "unet_best.weights.h5"

OUTPUT_DIR = PROJECT_ROOT / "prediction_results"


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """Prepare one image for U-Net inference."""

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_rgb = cv2.resize(image_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

    image_rgb = image_rgb.astype(np.float32) / 255.0

    return np.expand_dims(image_rgb, axis=0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Recreate the same validation split used during training.
    pairs = find_all_dataset_pairs(TRAIN_DATASET_DIR)

    _, validation_pairs = train_validation_split(pairs, validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED)

    image_path, annotation_path = validation_pairs[SAMPLE_INDEX]

    print(f"Image: {image_path.name}")

    # Load the original image.
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    original_height, original_width = image.shape[:2]

    # Create the ground-truth semantic mask.
    ground_truth_mask = annotation_to_semantic_mask(annotation_path, original_height, original_width)

    # Prepare the image for the network.
    model_input = preprocess_image(image)

    # Build the same U-Net architecture used during training.
    model = build_unet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    # Load the trained weights.
    model.load_weights(str(WEIGHTS_PATH))

    # Predict class probabilities for every pixel.
    prediction = model.predict(model_input, verbose=0)[0]

    # Convert probabilities into class identifiers.
    predicted_mask = np.argmax(prediction, axis=-1).astype(np.uint8)

    # Restore the predicted mask to the original image resolution.
    predicted_mask = cv2.resize(predicted_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    # Create visualization overlays.
    ground_truth_overlay = create_overlay(image, ground_truth_mask)
    prediction_overlay = create_overlay(image, predicted_mask)

    # Save the results.
    cv2.imwrite(str(OUTPUT_DIR / "original.png"), image)
    cv2.imwrite(str(OUTPUT_DIR / "ground_truth_overlay.png"), ground_truth_overlay)
    cv2.imwrite(str(OUTPUT_DIR / "prediction_overlay.png"), prediction_overlay)

    print("Prediction completed.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()