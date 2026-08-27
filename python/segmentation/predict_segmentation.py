#novkovic

from pathlib import Path

import cv2
import numpy as np

from segmentation.segmentation_config import PATCH_WIDTH,PATCH_HEIGHT,BATCH_SIZE,VALIDATION_FRACTION,RANDOM_SEED,PATCH_OVERLAP,PATCH_WIDTH,PATCH_HEIGHT
from segmentation.segmentation_config import GEOMETRY_MIN_AREA_FOR_SHAPE, MAX_CONE_WIDTH_HEIGHT_RATIO, MAX_CONE_EXTENT, MAX_COMPONENT_AREA_RATIO

from dataset.dataset_utils import NUM_CLASSES,BACKGROUND_ID,SMALL_ORANGE_CONE_ID,BIG_ORANGE_CONE_ID,annotation_to_semantic_mask,create_overlay,find_all_dataset_pairs,train_validation_split

from segmentation.segmentation_model import build_unet

NUM_SAMPLES = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TRAIN_DATASET_DIR = PROJECT_ROOT / "dataset" / "fsoco_segmentation_train"

WEIGHTS_PATH = PROJECT_ROOT / "models" / "unet_best.weights.h5"

OUTPUT_DIR = PROJECT_ROOT / "prediction_results"


def get_patch_positions(length: int, patch_size: int, overlap: int) -> list[int]:
    """Return patch starting positions that cover an entire image dimension."""

    if length <= patch_size:
        return [0]

    stride = patch_size - overlap

    positions = list(range(0,length - patch_size + 1,stride,))

    last_position = length - patch_size

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions

def predict_with_patches(model, image_bgr: np.ndarray) -> np.ndarray:
    """Predict full-resolution probabilities using overlapping patches."""

    original_height, original_width = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    padded_height = max(original_height, PATCH_HEIGHT)
    padded_width = max(original_width, PATCH_WIDTH)

    padded_image = np.zeros((padded_height, padded_width, 3),dtype=np.uint8,)

    padded_image[:original_height, :original_width] = image_rgb

    padded_image = padded_image.astype(np.float32) / 255.0

    x_positions = get_patch_positions(padded_width,PATCH_WIDTH,PATCH_OVERLAP,)

    y_positions = get_patch_positions(padded_height,PATCH_HEIGHT,PATCH_OVERLAP,)

    patches = []
    patch_coordinates = []

    for y_start in y_positions:
        for x_start in x_positions:
            patch = padded_image[y_start:y_start + PATCH_HEIGHT,x_start:x_start + PATCH_WIDTH,]

            patches.append(patch)
            patch_coordinates.append((x_start, y_start))

    patches = np.stack(patches, axis=0)

    predictions = model.predict(patches,batch_size=BATCH_SIZE,verbose=0,)

    probability_sum = np.zeros((padded_height, padded_width, NUM_CLASSES),dtype=np.float32,)

    prediction_count = np.zeros((padded_height, padded_width, 1),dtype=np.float32,)

    for prediction, (x_start, y_start) in zip(predictions, patch_coordinates):
        probability_sum[
        y_start:y_start + PATCH_HEIGHT,
        x_start:x_start + PATCH_WIDTH,
        ] += prediction

        prediction_count[
        y_start:y_start + PATCH_HEIGHT,
        x_start:x_start + PATCH_WIDTH,
        ] += 1.0

    probabilities = probability_sum / prediction_count

    probabilities = probabilities[:original_height,:original_width,]

    return probabilities

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

def filter_geometric_components(predicted_mask: np.ndarray) -> np.ndarray:
    """Remove cone components with clearly implausible geometry."""

    filtered_mask = predicted_mask.copy()

    image_height, image_width = predicted_mask.shape
    image_area = image_height * image_width

    for class_id in range(1, NUM_CLASSES):
        class_mask = (predicted_mask == class_id).astype(np.uint8)

        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for component_id in range(1, num_components):
            x = stats[component_id, cv2.CC_STAT_LEFT]
            y = stats[component_id, cv2.CC_STAT_TOP]
            width = stats[component_id, cv2.CC_STAT_WIDTH]
            height = stats[component_id, cv2.CC_STAT_HEIGHT]
            area = stats[component_id, cv2.CC_STAT_AREA]

            area_ratio = area / image_area

            # Remove extremely large components.
            if area_ratio > MAX_COMPONENT_AREA_RATIO:
                filtered_mask[labels == component_id] = BACKGROUND_ID
                continue

            # Preserve very small components because they may be distant cones.
            if area < GEOMETRY_MIN_AREA_FOR_SHAPE:
                continue

            width_height_ratio = width / max(height, 1)
            bounding_box_area = width * height
            extent = area / max(bounding_box_area, 1)

            too_wide = width_height_ratio > MAX_CONE_WIDTH_HEIGHT_RATIO
            too_rectangular = extent > MAX_CONE_EXTENT

            if too_wide or too_rectangular:
                filtered_mask[labels == component_id] = BACKGROUND_ID

    return filtered_mask

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Recreate the same validation split used during training.
    pairs = find_all_dataset_pairs(TRAIN_DATASET_DIR)
    _, validation_pairs = train_validation_split(pairs, validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED)

    # Build the same U-Net architecture used during patch training.
    model = build_unet(input_shape=(PATCH_HEIGHT, PATCH_WIDTH, 3), num_classes=NUM_CLASSES)

    # Load the trained patch-model weights.
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

        # Create the ground-truth semantic mask at original resolution.
        ground_truth_mask = annotation_to_semantic_mask(annotation_path, original_height, original_width)

        # Predict full-resolution class probabilities using overlapping patches.
        prediction = predict_with_patches(model, image)

        # Convert class probabilities into class identifiers.
        predicted_mask = np.argmax(prediction, axis=-1).astype(np.uint8)

        # Force each orange cone region to have one consistent orange class.
        predicted_mask = make_orange_cones_consistent(predicted_mask, prediction)

        predicted_mask = filter_geometric_components(predicted_mask)

        # Create visualization overlays.
        ground_truth_overlay = create_overlay(image, ground_truth_mask)
        prediction_overlay = create_overlay(image, predicted_mask)

        # Create a separate output folder for each sample.
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