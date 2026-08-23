from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from dataset.dataset_utils import IGNORE_ID, NUM_CLASSES, annotation_to_semantic_mask, find_all_dataset_pairs, resize_mask, train_validation_split
from segmentation.segmentation_model import build_unet

from segmentation.segmentation_config import IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_FRACTION, RANDOM_SEED


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TRAIN_DATASET_DIR = PROJECT_ROOT / "dataset" / "fsoco_segmentation_train"

MODEL_DIR = PROJECT_ROOT / "models"

BEST_WEIGHTS_PATH = MODEL_DIR / "unet_best.weights.h5"


def resize_with_padding(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize image and mask while preserving their aspect ratio."""

    original_height, original_width = image.shape[:2]

    scale = min(IMAGE_WIDTH / original_width, IMAGE_HEIGHT / original_height)

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    image_output = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    mask_output = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), IGNORE_ID, dtype=np.uint8)

    x_offset = (IMAGE_WIDTH - new_width) // 2
    y_offset = (IMAGE_HEIGHT - new_height) // 2

    image_output[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    mask_output[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_mask

    return image_output, mask_output

def load_sample(image_path: Path, annotation_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess one image-mask pair.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    original_height, original_width = image.shape[:2]

    mask = annotation_to_semantic_mask(annotation_path, original_height, original_width)

    # OpenCV loads images in BGR format.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image, mask = resize_with_padding(image, mask)

    image = image.astype(np.float32) / 255.0
    mask = mask.astype(np.int32)

    return image, mask


def load_dataset(pairs: list[tuple[Path, Path]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all image-mask pairs into NumPy arrays.
    """

    images = []
    masks = []

    for index, (image_path, annotation_path) in enumerate(pairs):
        image, mask = load_sample(image_path, annotation_path)

        images.append(image)
        masks.append(mask)

        if (index + 1) % 100 == 0:
            print(f"Loaded {index + 1}/{len(pairs)} samples.")

    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)

    return images, masks


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """
    Compute sparse categorical cross-entropy while ignoring
    pixels whose class identifier is IGNORE_ID.
    """

    y_true = tf.cast(y_true, tf.int32)

    valid_pixels = tf.not_equal(y_true, IGNORE_ID)

    # Replace ignored pixels with a temporary valid class.
    # Their contribution will be removed from the loss afterwards.
    safe_y_true = tf.where(valid_pixels, y_true, 0)

    pixel_loss = tf.keras.losses.sparse_categorical_crossentropy(safe_y_true, y_pred)

    valid_pixels = tf.cast(valid_pixels, pixel_loss.dtype)

    pixel_loss = pixel_loss * valid_pixels

    total_loss = tf.reduce_sum(pixel_loss)
    valid_pixel_count = tf.reduce_sum(valid_pixels)

    return total_loss / tf.maximum(valid_pixel_count, 1.0)


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


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("TensorFlow version:", tf.__version__)
    print("Available GPUs:", tf.config.list_physical_devices("GPU"))

    # Find all image-annotation pairs in the segmentation training dataset.
    pairs = find_all_dataset_pairs(TRAIN_DATASET_DIR)

    if len(pairs) == 0:
        raise RuntimeError(f"No training samples found in: {TRAIN_DATASET_DIR}")

    print(f"Total samples: {len(pairs)}")

    # Split the training dataset into training and validation sets.
    training_pairs, validation_pairs = train_validation_split(pairs, validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED)

    print(f"Training samples: {len(training_pairs)}")
    print(f"Validation samples: {len(validation_pairs)}")

    # Load the training dataset.
    print("\nLoading training dataset...")
    x_train, y_train = load_dataset(training_pairs)

    # Load the validation dataset.
    print("\nLoading validation dataset...")
    x_validation, y_validation = load_dataset(validation_pairs)

    print("\nDataset shapes:")
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_validation:", x_validation.shape)
    print("y_validation:", y_validation.shape)

    # Build the U-Net model.
    model = build_unet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=NUM_CLASSES)

    model.summary()

    # Configure the optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Configure the model for training.
    model.compile(optimizer=optimizer, loss=masked_sparse_categorical_crossentropy, metrics=[masked_pixel_accuracy])

    # Save the model weights whenever the validation loss improves.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=str(BEST_WEIGHTS_PATH), monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)

    # Stop training if the validation loss does not improve for several epochs.
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

    # Train the model.
    model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[checkpoint, early_stopping], shuffle=True)

    print(f"\nBest weights saved to: {BEST_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()