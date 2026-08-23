from pathlib import Path

import cv2
import numpy as np

from segmentation.segmentation_config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    VALIDATION_FRACTION,
    RANDOM_SEED,
)

from dataset.dataset_utils import (
    NUM_CLASSES,
    SMALL_ORANGE_CONE_ID,
    BIG_ORANGE_CONE_ID,
    annotation_to_semantic_mask,
    create_overlay,
    find_all_dataset_pairs,
    train_validation_split,
)

from segmentation.segmentation_model import build_unet


NUM_SAMPLES = 5


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TRAIN_DATASET_DIR = PROJECT_ROOT / "dataset" / "fsoco_segmentation_train"

WEIGHTS_PATH = PROJECT_ROOT / "models" / "unet_best.weights.h5"

OUTPUT_DIR = PROJECT_ROOT / "prediction_results"


def preprocess_image(image_bgr: np.ndarray):
    """Resize an image with padding while preserving its aspect ratio."""

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    original_height, original_width = image_rgb.shape[:2]

    scale = min(IMAGE_WIDTH / original_width, IMAGE_HEIGHT / original_height)

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = cv2.resize(image_rgb,(new_width, new_height),interpolation=cv2.INTER_LINEAR,)

    padded_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3),dtype=np.uint8,)

    x_offset = (IMAGE_WIDTH - new_width) // 2
    y_offset = (IMAGE_HEIGHT - new_height) // 2

    padded_image[
    y_offset:y_offset + new_height,
    x_offset:x_offset + new_width
    ] = resized_image

    padded_image = padded_image.astype(np.float32) / 255.0

    model_input = np.expand_dims(padded_image, axis=0)

    return model_input, x_offset, y_offset, new_width, new_height

def make_orange_cones_consistent(predicted_mask: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Assign one orange class to each connected orange cone region."""

    orange_mask = np.logical_or(
        predicted_mask == SMALL_ORANGE_CONE_ID,
        predicted_mask == BIG_ORANGE_CONE_ID,
        ).astype(np.uint8)

    num_components, component_labels = cv2.connectedComponents(orange_mask, connectivity=8)

    corrected_mask = predicted_mask.copy()

    for component_id in range(1, num_components):
        component = component_labels == component_id

        small_score = np.mean(probabilities[..., SMALL_ORANGE_CONE_ID][component])
        big_score = np.mean(probabilities[..., BIG_ORANGE_CONE_ID][component])

        if small_score >= big_score:
            corrected_mask[component] = SMALL_ORANGE_CONE_ID
        else:
            corrected_mask[component] = BIG_ORANGE_CONE_ID

    return corrected_mask

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Recreate the same validation split used during training.
    pairs = find_all_dataset_pairs(TRAIN_DATASET_DIR)
    _, validation_pairs = train_validation_split(pairs, validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED)

    # Build the same U-Net architecture used during training.
    model = build_unet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=NUM_CLASSES)

    # Load the trained weights only once.
    model.load_weights(str(WEIGHTS_PATH))

    # Run prediction on multiple validation samples.
    for sample_index in range(NUM_SAMPLES):
        image_path, annotation_path = validation_pairs[sample_index]

        print(f"Sample {sample_index + 1}/{NUM_SAMPLES}: {image_path.name}")

        # Load the original image.
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        original_height, original_width = image.shape[:2]

        # Create the ground-truth semantic mask.
        ground_truth_mask = annotation_to_semantic_mask(annotation_path, original_height, original_width)

        # Prepare the image using the same preprocessing used during training.
        model_input, x_offset, y_offset, new_width, new_height = preprocess_image(image)

        prediction = model.predict(model_input, verbose=0)[0]

        # Remove padding from the probability maps.
        prediction = prediction[
             y_offset:y_offset + new_height,
             x_offset:x_offset + new_width,
        ]

        # Convert probabilities into class identifiers.
        predicted_mask = np.argmax(prediction, axis=-1).astype(np.uint8)

        # Force each orange cone region to have one consistent orange class.
        predicted_mask = make_orange_cones_consistent(predicted_mask, prediction)

        # Restore the predicted mask to the original image resolution.
        predicted_mask = cv2.resize(
            predicted_mask,(original_width, original_height),interpolation=cv2.INTER_NEAREST,)

        # Create visualization overlays.
        ground_truth_overlay = create_overlay(image, ground_truth_mask)
        prediction_overlay = create_overlay(image, predicted_mask)

        # Create a separate folder for each sample.
        sample_output_dir = OUTPUT_DIR / f"sample_{sample_index + 1}"
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        # Save the results.
        cv2.imwrite(str(sample_output_dir / "original.png"), image)
        cv2.imwrite(str(sample_output_dir / "ground_truth_overlay.png"), ground_truth_overlay)
        cv2.imwrite(str(sample_output_dir / "prediction_overlay.png"), prediction_overlay)

    print("Predictions completed.")
    print(f"Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()