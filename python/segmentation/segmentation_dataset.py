#novkovic

"""Dataset pipeline for cone segmentation."""

from pathlib import Path
import random
from typing import Iterator, Sequence

import cv2
import numpy as np
import tensorflow as tf

from dataset.dataset_utils import Box, annotation_to_segmentation_instances, get_segmentation_train_pairs, load_image, train_validation_split
from segmentation.segmentation_config import BATCH_SIZE, BBOX_CENTER_JITTER, BBOX_PADDING_MAX, BBOX_PADDING_MIN, BRIGHTNESS_JITTER, CONTRAST_JITTER, HORIZONTAL_FLIP_PROBABILITY, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, MAX_ROTATION_DEGREES, RANDOM_SEED, SATURATION_JITTER, VALIDATION_FRACTION,ROTATION_PROBABILITY,COLOR_AUGMENTATION_PROBABILITY


def create_crop_box(box: Box, image_height: int, image_width: int, rng: random.Random, training: bool) -> tuple[int, int, int, int]:
    """Expand a ground-truth box and optionally perturb its center during training."""

    box_width = box.x_max - box.x_min
    box_height = box.y_max - box.y_min

    if training:
        padding_factor = rng.uniform(BBOX_PADDING_MIN, BBOX_PADDING_MAX)
        center_offset_x = rng.uniform(-BBOX_CENTER_JITTER, BBOX_CENTER_JITTER) * box_width
        center_offset_y = rng.uniform(-BBOX_CENTER_JITTER, BBOX_CENTER_JITTER) * box_height
    else:
        padding_factor = (BBOX_PADDING_MIN + BBOX_PADDING_MAX) / 2.0
        center_offset_x = 0.0
        center_offset_y = 0.0

    center_x = (box.x_min + box.x_max) / 2.0 + center_offset_x
    center_y = (box.y_min + box.y_max) / 2.0 + center_offset_y

    crop_width = box_width * (1.0 + 2.0 * padding_factor)
    crop_height = box_height * (1.0 + 2.0 * padding_factor)

    x_min = max(0, int(np.floor(center_x - crop_width / 2.0)))
    y_min = max(0, int(np.floor(center_y - crop_height / 2.0)))
    x_max = min(image_width, int(np.ceil(center_x + crop_width / 2.0)))
    y_max = min(image_height, int(np.ceil(center_y + crop_height / 2.0)))

    return x_min, y_min, x_max, y_max


def letterbox_sample(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int]]:
    """Resize an image and optional mask while preserving their aspect ratio."""

    image_height, image_width = image.shape[:2]

    scale = min(INPUT_WIDTH / image_width, INPUT_HEIGHT / image_height)

    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))

    resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    horizontal_padding = INPUT_WIDTH - resized_width
    vertical_padding = INPUT_HEIGHT - resized_height

    left_padding = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding
    top_padding = vertical_padding // 2
    bottom_padding = vertical_padding - top_padding

    padded_image = cv2.copyMakeBorder(resized_image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_REPLICATE)

    padded_mask = None

    if mask is not None:
        resized_mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        padded_mask = cv2.copyMakeBorder(resized_mask, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=0)

    metadata = (top_padding, left_padding, resized_height, resized_width)

    return padded_image, padded_mask, metadata

def apply_geometric_augmentation(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Apply synchronized geometric augmentation to an image and its mask."""

    if rng.random() < HORIZONTAL_FLIP_PROBABILITY:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    if rng.random() < ROTATION_PROBABILITY:
        angle = rng.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES)

        image_height, image_width = image.shape[:2]
        center = (image_width / 2.0, image_height / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        image = cv2.warpAffine(image, rotation_matrix, (image_width, image_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(mask, rotation_matrix, (image_width, image_height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image, mask


def apply_color_augmentation(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply moderate brightness, contrast, and saturation augmentation."""

    image_float = image.astype(np.float32)

    contrast_factor = rng.uniform(1.0 - CONTRAST_JITTER, 1.0 + CONTRAST_JITTER)
    brightness_offset = rng.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER) * 255.0

    channel_mean = np.mean(image_float, axis=(0, 1), keepdims=True)
    image_float = (image_float - channel_mean) * contrast_factor + channel_mean
    image_float += brightness_offset
    image = np.clip(image_float, 0.0, 255.0).astype(np.uint8)

    saturation_factor = rng.uniform(1.0 - SATURATION_JITTER, 1.0 + SATURATION_JITTER)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_image[:, :, 1] *= saturation_factor
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0.0, 255.0)

    return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)


def prepare_instance_sample(image_rgb: np.ndarray, instance, rng: random.Random, training: bool) -> tuple[np.ndarray, np.ndarray]:
    """Convert one cone instance into a model-ready image crop and binary mask."""

    image_height, image_width = image_rgb.shape[:2]

    x_min, y_min, x_max, y_max = create_crop_box(instance.bbox, image_height, image_width, rng, training)

    image_crop = image_rgb[y_min:y_max, x_min:x_max]
    mask_crop = instance.mask[y_min:y_max, x_min:x_max].astype(np.uint8)

    image_crop, mask_crop, _ = letterbox_sample(image_crop, mask_crop)

    if training:
        image_crop, mask_crop = apply_geometric_augmentation(image_crop, mask_crop, rng)

        if rng.random() < COLOR_AUGMENTATION_PROBABILITY:
            image_crop = apply_color_augmentation(image_crop, rng)

    image_crop = image_crop.astype(np.float32) / 255.0
    mask_crop = mask_crop.astype(np.float32)
    mask_crop = np.expand_dims(mask_crop, axis=-1)

    return image_crop, mask_crop


def segmentation_sample_generator(pairs: Sequence[tuple[Path, Path]], training: bool) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Generate segmentation samples from image and annotation pairs."""

    rng = random.Random(RANDOM_SEED)
    ordered_pairs = list(pairs)

    if training:
        rng.shuffle(ordered_pairs)

    for image_path, annotation_path in ordered_pairs:
        image_bgr = load_image(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        image_height, image_width = image_rgb.shape[:2]
        instances = annotation_to_segmentation_instances(annotation_path, image_height, image_width)

        if training:
            rng.shuffle(instances)

        for instance in instances:
            yield prepare_instance_sample(image_rgb, instance, rng, training)


def create_segmentation_dataset(pairs: Sequence[tuple[Path, Path]], training: bool) -> tf.data.Dataset:
    """Create a batched TensorFlow dataset for segmentation."""

    output_signature = (
        tf.TensorSpec(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(INPUT_HEIGHT, INPUT_WIDTH, 1), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(lambda: segmentation_sample_generator(pairs, training), output_signature=output_signature)

    if training:
        dataset = dataset.shuffle(buffer_size=512, seed=RANDOM_SEED, reshuffle_each_iteration=True)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_train_validation_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create segmentation training and validation datasets."""

    pairs = get_segmentation_train_pairs()
    train_pairs, validation_pairs = train_validation_split(pairs, validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED)

    train_dataset = create_segmentation_dataset(train_pairs, training=True)
    validation_dataset = create_segmentation_dataset(validation_pairs, training=False)

    return train_dataset, validation_dataset