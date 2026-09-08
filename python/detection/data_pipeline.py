#tino

import tensorflow as tf
import keras_cv
import cv2

from detection.inference import (
    resize_with_letterbox
)

from dataset.dataset_utils import (
    get_bounding_boxes_train_pairs,
    train_validation_split,
    annotation_to_instances,
    segmentation_id_to_detection_id,
    load_image,
    load_annotation
)

from detection.detection_config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    NUM_PARALLEL_CALLS,
    PREFETCH_BUFFER,
    SEED,
    SPLIT_RATIO,
)

def prepare_dataset_data(pairs):
    """
    Reads the annotations using dataset_utils and converts them
    into the format needed to create a TensorFlow dataset.
    """

    image_paths = []
    all_boxes = []
    all_classes = []

    for image_path, annotation_path in pairs:

        # We load the image here only to know width and height.
        annotation = load_annotation(annotation_path)

        image_height = annotation["size"]["height"]
        image_width = annotation["size"]["width"]

        # dataset_utils parses the JSON, removes unknown_cone
        # and returns the bounding boxes.
        instances = annotation_to_instances(
            annotation_path,
            image_height,
            image_width
        )

        boxes = []
        classes = []

        for box in instances:

            boxes.append([
                float(box.x_min),
                float(box.y_min),
                float(box.x_max),
                float(box.y_max)
            ])

            # dataset_utils uses IDs 1-4.
            # Detection uses IDs 0-3.
            detection_class_id = segmentation_id_to_detection_id(
                box.class_id
            )

            classes.append(detection_class_id)

        image_paths.append(str(image_path))
        all_boxes.append(boxes)
        all_classes.append(classes)

    image_paths = tf.constant(
        image_paths,
        dtype=tf.string
    )

    bbox = tf.ragged.constant(
        all_boxes,
        dtype=tf.float32,
        ragged_rank=1
    )

    classes = tf.ragged.constant(
        all_classes,
        dtype=tf.float32,
        ragged_rank=1
    )

    return image_paths, classes, bbox


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

    # decode_image automatically handles common image formats
    # such as JPEG, PNG and BMP.
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
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
        "classes": tf.cast(classes, tf.float32),
        "boxes": tf.cast(bbox, tf.float32)
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


def build_train_val_datasets():
    """
    Builds training and validation datasets from the complete dataset.
    
    Creates TensorFlow datasets by:
    1. Collecting all image-annotation pairs
    2. Splitting data into training and validation sets
    3. Parsing annotations to extract bounding boxes and classes
    4. Applying data augmentation and preprocessing to training data
    5. Batching and prefetching for efficient training
    
    Returns:
        tuple: (train_ds, val_ds) TensorFlow datasets ready for training
    """

    # 1. pairs of the type: [image_dir, annotation_dir]
    pairs = get_bounding_boxes_train_pairs()
    num_samples = len(pairs)

    print(f"\nImage/annotation pairs found: {num_samples}")

    # 2.
    train_pairs, val_pairs = train_validation_split(pairs, SPLIT_RATIO, SEED)
    print("Train pairs: ", len(train_pairs))
    print("Val pairs: ", len(val_pairs))

    num_train = len(train_pairs)
    num_val = len(val_pairs)

    # 3.
    print("Reading annotations ...")
    train_image_paths, train_classes, train_bboxes = prepare_dataset_data(train_pairs)
    val_image_paths, val_classes, val_bboxes = prepare_dataset_data(val_pairs)

    print("\nDataset loaded.")
    print("Images:", train_image_paths.shape)
    print("Classes:", train_classes.shape)
    print("Bounding boxes:", train_bboxes.shape)

    # Create a TensorFlow dataset by combining image paths, classes, and bounding boxes
    # from_tensor_slices creates a dataset where each element is a slice of the inputs
    # now the structure of the dataset is:
    # elem 0: (path, list_of_class, list_of_boxes)
    train_data = tf.data.Dataset.from_tensor_slices(
        (
            train_image_paths,
            train_classes,
            train_bboxes,
        )
    )

    val_data = tf.data.Dataset.from_tensor_slices(
        (
            val_image_paths,
            val_classes,
            val_bboxes
        )
    )

    # Display the split results
    print("\nTraining samples:", num_train)
    print("Validation samples:", num_val)

    train_ds = build_train_dataset(train_data, num_train)
    val_loss_ds = build_val_loss_dataset(val_data)
    val_inference_ds = build_val_inference_dataset(val_data)



    return train_ds, val_loss_ds, val_inference_ds


def build_train_dataset(train_data, num_train):
    # Shuffle the training samples at each epoch.
    train_data = train_data.shuffle(
        buffer_size=num_train,
        seed=SEED,
        reshuffle_each_iteration=True
    )

    # Jittered Resize is used to resize with a casual variability
    train_resizing = keras_cv.layers.JitteredResize(
        target_size=IMAGE_SIZE,
        scale_factor=(0.75, 1.30), #range of rescaling
        bounding_box_format="xyxy",
    )

    train_ds = train_data.map(
        load_dataset,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        deterministic=False,
    )

    # Raggruppa gli elementi in batch dopo aver caricato le immagini
    # I batch permettono di processare più immagini insieme sulla GPU (più efficiente)
    # drop_remainder=True: scarta gli ultimi elementi se non fanno un batch completo
    train_ds = train_ds.ragged_batch(BATCH_SIZE,drop_remainder=True)


    # Applica resizing e data augmentation DOPO batching
    # Questo è più efficiente: resizziamo interi batch, non singole immagini
    # La variabilità JitteredResize (0.75-1.30) aggiunge diversità ai dati di training
    train_ds = train_ds.map(
        train_resizing,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        deterministic=False,
    )

    # FORMAT CONVERSION
    # Converte il formato da dict a tuple per compatibilità con il modello
    # Il modello si aspetta input come (images, bounding_boxes), non come dizionario
    train_ds = train_ds.map(
        dict_to_tuple,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        deterministic=False,
    )
        # Prefetch prepares data in advance while the model is training
    # it reduces the waiting time
    # PREFETCH_BUFFER è grande per training, piccolo (1) per validazione
    
    train_ds = train_ds.prefetch(PREFETCH_BUFFER)

    train_options = tf.data.Options()
    train_options.experimental_deterministic = False

    train_ds = train_ds.with_options(train_options)


    return train_ds

def resize_validation_sample(inputs):
    image = inputs["images"]
    boxes = inputs["bounding_boxes"]["boxes"]
    classes = inputs["bounding_boxes"]["classes"]

    image, scale, pad_x, pad_y = resize_with_letterbox(
        image,
        IMAGE_SIZE,
    )

    boxes = tf.cast(boxes, tf.float32)

    x_min = boxes[:, 0] * scale + tf.cast(pad_x, tf.float32)
    y_min = boxes[:, 1] * scale + tf.cast(pad_y, tf.float32)
    x_max = boxes[:, 2] * scale + tf.cast(pad_x, tf.float32)
    y_max = boxes[:, 3] * scale + tf.cast(pad_y, tf.float32)

    boxes = tf.stack(
        [x_min, y_min, x_max, y_max],
        axis=-1,
    )

    return {
        "images": image,
        "bounding_boxes": {
            "classes": classes,
            "boxes": boxes,
        }
    }

def build_val_loss_dataset(val_data):

    val_ds = val_data.map(
        load_dataset,
        num_parallel_calls=1,
    )

    val_ds = val_ds.map(resize_validation_sample, num_parallel_calls=1)

    val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=False)
    val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=1)

    val_ds = val_ds.prefetch(1)

    return val_ds

def build_val_inference_dataset(val_data):
    val_ds = val_data.map(
        load_dataset,
        num_parallel_calls = 1
    )

    val_ds = val_ds.prefetch(1)
    return val_ds

