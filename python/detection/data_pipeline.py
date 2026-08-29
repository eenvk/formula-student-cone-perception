# ============================================================
# DATASET AND TF.DATA PIPELINE
# ============================================================

import json
from pathlib import Path

import tensorflow as tf
import keras_cv

from detection_config import (
    BATCH_SIZE,
    CLASS_TO_ID,
    DATASET_DIR,
    IMAGE_SIZE,
    NUM_PARALLEL_CALLS,
    PREFETCH_BUFFER,
    SEED,
    SPLIT_RATIO,
    VALID_EXTENSIONS,
)


def collect_image_annotation_pairs(dataset_dir: Path):
    """
    Collects pairs of images and their corresponding annotation files.
    
    This function searches the dataset directory for images and matches them
    with their JSON annotation files. It validates that all images have 
    corresponding annotations and raises an error if any are missing.
    
    Args:
        dataset_dir (Path): Root directory containing subdirectories with 
                           'img' and 'ann' folders
    
    Returns:
        list: List of tuples containing (image_path, annotation_path) pairs
    
    Raises:
        FileNotFoundError: If directory not found or images missing annotations
        RuntimeError: If no valid images are found in the directory
    """
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{dataset_dir}"
        )

    #sort all the images path found
    image_paths = sorted(
        path
        for path in dataset_dir.glob("*/img/*")
        if (
            path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            f"No images found inside:\n{dataset_dir}"
        )

    pairs = []
    missing_annotations = []

    for image_path in image_paths:
        annotation_path = (
            image_path.parent.parent
            / "ann"
            / f"{image_path.name}.json"
        )

        if annotation_path.is_file():
            pairs.append((image_path, annotation_path))
        else:
            missing_annotations.append(
                (image_path, annotation_path)
            )

    if missing_annotations:
        message = [
            "Some images do not have the expected annotation.",
            "",
        ]

        for image_path, annotation_path in missing_annotations[:10]:
            message.append(
                f"Image: {image_path}\n"
                f"Expected JSON: {annotation_path}\n"
            )

        if len(missing_annotations) > 10:
            message.append(
                f"... and {len(missing_annotations) - 10} others."
            )

        raise FileNotFoundError("\n".join(message))

    return pairs


def parse_annotation(json_path: Path):
    """
    Parses a JSON annotation file to extract bounding boxes and class labels.
    
    Reads annotation data from JSON files and converts bounding box coordinates
    and class information into lists. Validates that all classes exist in the
    CLASS_TO_ID mapping and that bounding box coordinates are valid.
    
    Args:
        json_path (Path): Path to the JSON annotation file
    
    Returns:
        tuple: (boxes, classes) where:
               - boxes: List of [x_min, y_min, x_max, y_max] coordinates
               - classes: List of class IDs corresponding to each box
    
    Raises:
        ValueError: If unknown class found or invalid bounding box coordinates
    """
    with json_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        annotation = json.load(file)

    boxes = []
    classes = []

    objects = annotation.get("objects", [])

    for obj in objects:
        class_name = obj["classTitle"]

        if class_name == "unknown_cone":
            continue

        if class_name not in CLASS_TO_ID:
            raise ValueError(
                f"Unknown class '{class_name}' in:\n"
                f"{json_path}"
            )

        exterior = obj["points"]["exterior"]

        if len(exterior) != 2:
            raise ValueError(
                f"Invalid bounding box in:\n{json_path}"
            )

        (x1, y1), (x2, y2) = exterior

        x_min = float(min(x1, x2))
        y_min = float(min(y1, y2))
        x_max = float(max(x1, x2))
        y_max = float(max(y1, y2))

        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"Invalid bounding box coordinates in:\n"
                f"{json_path}\n"
                f"Box: {(x_min, y_min, x_max, y_max)}"
            )

        boxes.append(
            [x_min, y_min, x_max, y_max]
        )

        classes.append(
            CLASS_TO_ID[class_name]
        )

    return boxes, classes


@tf.function
def load_image(image_path):
    """
    Loads an image file and decodes it into a tensor.
    
    Reads an image from disk (PNG or JPEG) and decodes it into a 3-channel
    RGB tensor. Automatically detects the image format based on file extension.
    
    Args:
        image_path: Path to the image file (PNG or JPEG)
    
    Returns:
        Tensor: Decoded image with shape [height, width, 3]
    """
    image_bytes = tf.io.read_file(image_path)

    lower_path = tf.strings.lower(image_path)

    is_png = tf.strings.regex_full_match(
        lower_path,
        r".*\.png"
    )

    image = tf.cond(
        is_png,
        lambda: tf.io.decode_png(
            image_bytes,
            channels=3,
        ),
        lambda: tf.io.decode_jpeg(
            image_bytes,
            channels=3,
        ),
    )

    image.set_shape(
        [None, None, 3]
    )

    return image


def load_dataset(image_path, classes, bbox):
    """
    Loads an image and packages it with its bounding box annotations.
    
    Combines an image with its associated bounding box and class information
    into a dictionary format. Used as a mapping function in the TensorFlow
    data pipeline.
    
    Args:
        image_path: Path to the image file
        classes: List of class IDs for each bounding box
        bbox: List of bounding box coordinates
    
    Returns:
        dict: Dictionary with 'images' and 'bounding_boxes' keys
    """
    image = load_image(image_path)

    bounding_boxes = {
        "classes": tf.cast(
            classes,
            tf.float32,
        ),
        "boxes": tf.cast(
            bbox,
            tf.float32,
        ),
    }

    return {
        "images": image,
        "bounding_boxes": bounding_boxes,
    }


def dict_to_tuple(inputs):
    """
    Converts dataset samples from dictionary format to tuple format.
    
    Transforms the batched dictionary representation into a tuple of
    (images, bounding_boxes) which is the expected format for training.
    
    Args:
        inputs (dict): Dictionary with 'images' and 'bounding_boxes' keys
    
    Returns:
        tuple: (images, bounding_boxes) tuple
    """
    return (
        inputs["images"],
        inputs["bounding_boxes"],
    )


def build_datasets():
    """
    Builds training and validation datasets from the complete dataset.
    
    Creates TensorFlow datasets by:
    1. Collecting all image-annotation pairs
    2. Parsing annotations to extract bounding boxes and classes
    3. Splitting data into training and validation sets
    4. Applying data augmentation and preprocessing to training data
    5. Batching and prefetching for efficient training
    
    Returns:
        tuple: (train_ds, val_ds) TensorFlow datasets ready for training
    """
    pairs = collect_image_annotation_pairs(DATASET_DIR)

    print(f"\nImage/annotation pairs found: {len(pairs)}")

    image_paths = []
    all_boxes = []
    all_classes = []

    print("\nReading annotations...")

    for image_path, annotation_path in pairs:
        boxes, classes = parse_annotation(annotation_path)

        image_paths.append(str(image_path))
        all_boxes.append(boxes)
        all_classes.append(classes)

    # transforming list in ragged list, in this way different sizes of cones can be captured properly
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

    print("\nDataset loaded.")
    print("Images:", image_paths.shape)
    print("Bounding boxes:", bbox.shape)
    print("Classes:", classes.shape)

    # Create a TensorFlow dataset by combining image paths, classes, and bounding boxes
    # from_tensor_slices creates a dataset where each element is a slice of the inputs
    data = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            classes,
            bbox,
        )
    )

    # Calculate total number of samples in the dataset
    num_samples = len(pairs)

    # Calculate the number of validation samples based on SPLIT_RATIO
    # Use at least 1 sample for validation set
    num_val = int(num_samples * SPLIT_RATIO)

    # Calculate the remaining samples for training
    num_train = num_samples - num_val

    # Display the split results
    print("\nTraining samples:", num_train)
    print("Validation samples:", num_val)

    # First shuffle is deterministic so that train/validation split is stable.
    # This ensures the same images are always in train/validation across runs
    data = data.shuffle(
        buffer_size=num_samples,
        seed=SEED,
        reshuffle_each_iteration=False,
    )

    # Extract validation data: takes the first num_val samples from the shuffled dataset
    val_data = data.take(num_val)
    
    # Extract training data: skips the first num_val samples, takes the rest
    train_data = data.skip(num_val)

    # Training order changes at every epoch.
    train_data = train_data.shuffle(
        buffer_size=num_train,
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    # Jittered Resize is used to resize with a casual variability
    train_resizing = keras_cv.layers.JitteredResize(
        target_size=IMAGE_SIZE,
        scale_factor=(0.75, 1.30), #range of rescaling
        bounding_box_format="xyxy",
    )

    val_resizing = keras_cv.layers.JitteredResize(
        target_size=IMAGE_SIZE,
        scale_factor=(1.0, 1.0),
        bounding_box_format="xyxy",
    )

    train_ds = train_data.map(
        load_dataset,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        deterministic=False,
    )

    train_ds = train_ds.ragged_batch(BATCH_SIZE,drop_remainder=True)

    # Raggruppa gli elementi in batch dopo aver caricato le immagini
    # I batch permettono di processare più immagini insieme sulla GPU (più efficiente)
    # drop_remainder=True: scarta gli ultimi elementi se non fanno un batch completo

    # Applica resizing e data augmentation DOPO batching
    # Questo è più efficiente: resizziamo interi batch, non singole immagini
    # La variabilità JitteredResize (0.75-1.30) aggiunge diversità ai dati di training
    train_ds = train_ds.map(
        train_resizing,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        deterministic=False,
    )

    for images, y_true in train_ds.take(1):
        print(y_true["classes"])

    # ============================================================
    # VALIDATION DATASET PROCESSING
    # ============================================================
    
    val_ds = val_data.map(load_dataset, num_parallel_calls=1)
    
    # Batch senza scartare elementi (drop_remainder=False)
    # Vogliamo valutare su TUTTI i dati di validazione
    val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=False)
    
    # Applica resizing DETERMINISTICO (1.0-1.0 = nessuna variabilità)
    # La validazione deve essere coerente e ripetibile
    val_ds = val_ds.map(
        val_resizing,
        num_parallel_calls=1,
    )

    # ============================================================
    # FORMAT CONVERSION
    # ============================================================
    # Converte il formato da dict a tuple per compatibilità con il modello
    # Il modello si aspetta input come (images, bounding_boxes), non come dizionario

    train_ds = train_ds.map(
        dict_to_tuple,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        deterministic=False,
    )

    val_ds = val_ds.map(
        dict_to_tuple,
        num_parallel_calls=1,
    )

    # ============================================================
    # PREFETCHING FOR GPU OPTIMIZATION
    # ============================================================
    # Prefetch prepara i dati in anticipo mentre il modello si allena
    # Riduce i tempi di attesa: mentre la GPU allena batch N, prefetch carica batch N+1
    # PREFETCH_BUFFER è grande per training, piccolo (1) per validazione
    
    train_ds = train_ds.prefetch(PREFETCH_BUFFER)
    val_ds = val_ds.prefetch(1)

    train_options = tf.data.Options()
    train_options.experimental_deterministic = False

    train_ds = train_ds.with_options(train_options)

    return train_ds, val_ds
