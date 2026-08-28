#novkovic
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

from dataset.dataset_utils import BACKGROUND_ID,IGNORE_ID, NUM_CLASSES, annotation_to_semantic_mask, find_all_dataset_pairs, resize_mask, train_validation_split
from segmentation.segmentation_model import build_unet
from segmentation.segmentation_config import PATCH_WIDTH,PATCH_HEIGHT,CONE_PATCH_PROBABILITY,BATCH_SIZE,EPOCHS,LEARNING_RATE,VALIDATION_FRACTION,RANDOM_SEED

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATASET_DIR = SEGMENTATION_TRAIN_ROOT

MODEL_DIR = PROJECT_ROOT / "models"
BEST_WEIGHTS_PATH = MODEL_DIR / "unet_best.weights.h5"
BACKUP_DIR = MODEL_DIR / "training_backup"

def pad_to_patch_size(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad image and mask if they are smaller than the patch size."""

    original_height, original_width = image.shape[:2]

    padded_height = max(original_height, PATCH_HEIGHT)
    padded_width = max(original_width, PATCH_WIDTH)

    padded_image = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
    padded_mask = np.full((padded_height, padded_width), IGNORE_ID, dtype=np.uint8)

    padded_image[:original_height, :original_width] = image
    padded_mask[:original_height, :original_width] = mask

    return padded_image, padded_mask



def extract_patch(image: np.ndarray, mask: np.ndarray, training: bool) -> tuple[np.ndarray, np.ndarray]:
    """Extract one fixed-size patch from an image-mask pair."""

    image, mask = pad_to_patch_size(image, mask)

    image_height, image_width = image.shape[:2]

    max_x = image_width - PATCH_WIDTH
    max_y = image_height - PATCH_HEIGHT

    cone_pixels = np.logical_and(mask > BACKGROUND_ID, mask < NUM_CLASSES)

    if training and np.any(cone_pixels) and np.random.rand() < CONE_PATCH_PROBABILITY:
        y_coordinates, x_coordinates = np.where(cone_pixels)

        selected_pixel = np.random.randint(len(x_coordinates))

        center_x = int(x_coordinates[selected_pixel])
        center_y = int(y_coordinates[selected_pixel])

        jitter_x = np.random.randint(-PATCH_WIDTH // 4, PATCH_WIDTH // 4 + 1)
        jitter_y = np.random.randint(-PATCH_HEIGHT // 4, PATCH_HEIGHT // 4 + 1)

        x_start = center_x - PATCH_WIDTH // 2 + jitter_x
        y_start = center_y - PATCH_HEIGHT // 2 + jitter_y

        x_start = int(np.clip(x_start, 0, max_x))
        y_start = int(np.clip(y_start, 0, max_y))

    elif training:
        x_start = np.random.randint(0, max_x + 1)
        y_start = np.random.randint(0, max_y + 1)

    else:
        if np.any(cone_pixels):
            y_coordinates, x_coordinates = np.where(cone_pixels)

            center_x = int(np.median(x_coordinates))
            center_y = int(np.median(y_coordinates))

            x_start = int(np.clip(center_x - PATCH_WIDTH // 2, 0, max_x))
            y_start = int(np.clip(center_y - PATCH_HEIGHT // 2, 0, max_y))

        else:
            x_start = max_x // 2
            y_start = max_y // 2

    image_patch = image[y_start:y_start + PATCH_HEIGHT, x_start:x_start + PATCH_WIDTH]
    mask_patch = mask[y_start:y_start + PATCH_HEIGHT, x_start:x_start + PATCH_WIDTH]

    return image_patch, mask_patch

def extract_patch_around_point(
        image: np.ndarray,
        mask: np.ndarray,
        center_x: int,
        center_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a fixed-size patch around a selected point."""

    image_height, image_width = image.shape[:2]

    max_x = image_width - PATCH_WIDTH
    max_y = image_height - PATCH_HEIGHT

    x_start = center_x - PATCH_WIDTH // 2
    y_start = center_y - PATCH_HEIGHT // 2

    x_start = int(np.clip(x_start, 0, max_x))
    y_start = int(np.clip(y_start, 0, max_y))

    image_patch = image[
                  y_start:y_start + PATCH_HEIGHT,
                  x_start:x_start + PATCH_WIDTH,
                  ]

    mask_patch = mask[
                 y_start:y_start + PATCH_HEIGHT,
                 x_start:x_start + PATCH_WIDTH,
                 ]

    return image_patch, mask_patch

def load_sample(image_path: Path, annotation_path: Path, training: bool) -> tuple[np.ndarray, np.ndarray]:
    """Load one image-mask pair and extract a fixed-size patch."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    original_height, original_width = image.shape[:2]

    mask = annotation_to_semantic_mask(annotation_path, original_height, original_width)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image, mask = extract_patch(image, mask, training)

    image = image.astype(np.float32) / 255.0
    mask = mask.astype(np.int32)

    return image, mask

def sample_generator(pairs, training: bool):
    """Load dataset patches one at a time."""

    for image_path, annotation_path in pairs:
        image, mask = load_sample(image_path, annotation_path, training)

        yield image, mask

def create_dataset(pairs, training: bool):
    """Create a TensorFlow dataset that loads patches on demand."""

    dataset = tf.data.Dataset.from_generator(
        lambda: sample_generator(pairs, training),
        output_signature=(
            tf.TensorSpec(shape=(PATCH_HEIGHT, PATCH_WIDTH, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(PATCH_HEIGHT, PATCH_WIDTH), dtype=tf.int32),
        ),
    )

    if training:
        dataset = dataset.shuffle(buffer_size=16, seed=RANDOM_SEED, reshuffle_each_iteration=True)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset


def masked_pixel_accuracy(y_true, y_pred):
    """
    Compute pixel accuracy while ignoring IGNORE_ID pixels.
    """

    y_true = tf.cast(y_true, tf.int32)

    predicted_classes = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    valid_pixels = tf.not_equal(y_true, IGNORE_ID)

    correct_pixels = tf.equal(y_true, predicted_classes)
    correct_pixels = tf.logical_and(correct_pixels, valid_pixels)

    correct_pixels = tf.cast(correct_pixels, tf.float32)
    valid_pixels = tf.cast(valid_pixels, tf.float32)

    correct_pixel_count = tf.reduce_sum(correct_pixels)
    valid_pixel_count = tf.reduce_sum(valid_pixels)

    return correct_pixel_count / tf.maximum(valid_pixel_count, 1.0)

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """
    Compute sparse categorical cross-entropy while ignoring
    pixels whose class identifier is IGNORE_ID.
    """

    y_true = tf.cast(y_true, tf.int32)

    valid_pixels = tf.not_equal(y_true, IGNORE_ID)

    safe_y_true = tf.where(valid_pixels, y_true, 0)

    pixel_loss = tf.keras.losses.sparse_categorical_crossentropy(safe_y_true,y_pred,)

    valid_pixels = tf.cast(valid_pixels, pixel_loss.dtype)

    pixel_loss = pixel_loss * valid_pixels

    total_loss = tf.reduce_sum(pixel_loss)
    valid_pixel_count = tf.reduce_sum(valid_pixels)

    return total_loss / tf.maximum(valid_pixel_count, 1.0)

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Find all image-annotation pairs in the segmentation training dataset.
    pairs = find_all_dataset_pairs(TRAIN_DATASET_DIR)

    if len(pairs) == 0:
        raise RuntimeError(f"No training samples found in: {TRAIN_DATASET_DIR}")

    print(f"Total samples: {len(pairs)}")

    # Split the training dataset into training and validation sets.
    training_pairs, validation_pairs = train_validation_split(pairs, validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED)


    print(f"Training samples: {len(training_pairs)}")
    print(f"Validation samples: {len(validation_pairs)}")

    print("\nCreating training dataset")
    training_dataset = create_dataset(training_pairs,training=True)
    print("Creating validation dataset")
    validation_dataset = create_dataset(validation_pairs, training=False)

    # Build the U-Net model.
    model = build_unet(input_shape=(PATCH_HEIGHT, PATCH_WIDTH, 3),num_classes=NUM_CLASSES,)

    model.summary()

    # Configure the optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Configure the model for training.
    model.compile(optimizer=optimizer,loss=masked_sparse_categorical_crossentropy,metrics=[masked_pixel_accuracy],)
    # Save the model weights whenever the validation loss improves.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=str(BEST_WEIGHTS_PATH), monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)

    # Stop training if the validation loss does not improve for several epochs.
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

    backup = tf.keras.callbacks.BackupAndRestore(backup_dir=str(BACKUP_DIR),save_freq="epoch",delete_checkpoint=False,)

    # Train the model.
    model.fit(training_dataset,validation_data=validation_dataset,epochs=EPOCHS,callbacks=[checkpoint, early_stopping, backup],)
    print(f"\nBest weights saved to: {BEST_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()