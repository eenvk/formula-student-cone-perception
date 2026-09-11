#tino

import tensorflow as tf
import keras_cv
import cv2
import numpy as np
import random

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

from detection.inference import (
    resize_with_letterbox,
    create_combined_views,
    get_patch_starts
)

from detection.detection_config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    NUM_PARALLEL_CALLS,
    PREFETCH_BUFFER,
    SEED,
    SPLIT_RATIO,
    PATCH_BATCH_SIZE,
    PATCH_STRIDE,
    MIN_RETAINED_AREA,
    NUM_NEGATIVE_CROP_ATTEMPTS
)

def build_train_val_datasets():
    """
    Builds training and validation datasets from the complete dataset.
    
    Creates TensorFlow datasets by:
    1. Collecting all image-annotation pairs
    2. Splitting data into training and validation sets
    3. Parsing annotations to extract bounding boxes and classes
    4. Build the single element of the dataset
    5. Build train and validation set with images and prepares batching
    6. Build the inference validation set
    
    Returns:
        tuple: (train_ds, val_ds) TensorFlow datasets ready for training
    """

    # 1. pairs of the type: [image_dir, annotation_dir]
    pairs = get_bounding_boxes_train_pairs()
    num_samples = len(pairs)

    print(f"\nImage/annotation pairs found: {num_samples}")

    # 2.
    train_pairs, val_pairs = train_validation_split(pairs, SPLIT_RATIO, SEED)

    num_train = len(train_pairs)
    num_val = len(val_pairs)

    # Display the split results
    print("\nTraining samples:", num_train)
    print("Validation samples:", num_val)

    # 3.
    print("Reading annotations and extract bounding boxes ground truth and classes...")
    train_image_paths, train_classes, train_bboxes, _  = prepare_dataset_data(train_pairs)
    val_image_paths, val_classes, val_bboxes, val_image_shapes = prepare_dataset_data(val_pairs)

    print("\nDataset loaded with image paths.")
    print("Training set features:")
    print("- Images:", train_image_paths.shape)
    print("- Classes:", train_classes.shape)
    print("- Bounding boxes:", train_bboxes.shape)
    print("Validation set features:")
    print("- Images:", val_image_paths.shape)
    print("- Classes:", val_classes.shape)
    print("- Bounding bobuild_data_structurexes:", val_bboxes.shape)

    # 4.
    # Create a TensorFlow dataset by combining image paths, classes, and bounding boxes
    # from_tensor_slices creates a dataset where each element is a slice of the inputs
    # now the structure of the dataset is :
    # 
    # elem 0: (path, list_of_class, list_of_boxes), this will be then transform in
    # elem 0: (image, list_of_class, list_of boxes)
    train_data = build_data_structure(train_image_paths, train_classes, train_bboxes)
    val_data = build_data_structure(val_image_paths, val_classes, val_bboxes)

    # 5.
    train_ds = build_train_dataset(train_data, num_train)
    val_loss_ds = build_val_loss_dataset(val_data)

    # 6.
    val_inference_ds = build_inference_dataset(val_image_paths)
    val_inference_metadata = build_inference_metadata(val_image_shapes)

    print("Inference metadata:", len(val_inference_metadata))

    # we build also the solutions of the validation set because this information are not present in the inference_ds
    val_y_true = {"boxes": val_bboxes, "classes": val_classes}

    return train_ds, val_loss_ds, val_inference_ds, val_inference_metadata, val_y_true

def prepare_dataset_data(pairs):
    """
    Reads the annotations using dataset_utils and converts them
    into the format needed to create a TensorFlow dataset.
    """

    image_paths = []
    all_boxes = []
    all_classes = []
    image_shapes = []

    for image_path, annotation_path in pairs:

        # We load the image here only to know width and height.
        annotation = load_annotation(annotation_path)

        image_height = annotation["size"]["height"]
        image_width = annotation["size"]["width"]

        # dataset_utils parses the JSON, removes unknown_cone
        # and returns the bounding boxes.
        instances = annotation_to_instances(annotation_path, image_height, image_width)

        boxes = []
        classes = []

        for box in instances:

            boxes.append([float(box.x_min), float(box.y_min), float(box.x_max), float(box.y_max)])

            # dataset_utils uses IDs 1-4.
            # Detection uses IDs 0-3.
            detection_class_id = segmentation_id_to_detection_id(box.class_id)

            classes.append(detection_class_id)

        image_paths.append(str(image_path))
        all_boxes.append(boxes)
        all_classes.append(classes)
        image_shapes.append([image_height, image_width])

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

    image_shapes = tf.constant(image_shapes, dtype= tf.int32)

    return image_paths, classes, bbox, image_shapes


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

    image.set_shape([None, None, 3])

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



def build_data_structure(image_paths, classes, bboxes):
    """
    build the element of the dataset. Each element is formed by its
    image_path, class, bbox
    """
    return tf.data.Dataset.from_tensor_slices(
        (image_paths, classes, bboxes)
    )

def positive_crop(image_path, classes, bbox):
    """
    it applies the positive crop: crop the image to IMAGE_SIZExIMAGE_SIZE with at least 1 cone.
    the cone should be in a random position w.r.t. the crop and it needs to appear with at least a
    MIN_RETAINED_AREA, otherwise another crop should be considered.
    """
    image = load_image(image_path)

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    crop_height = IMAGE_SIZE[0]
    crop_width = IMAGE_SIZE[1]

    can_crop = tf.logical_and(
        image_height >= crop_height,
        image_width >= crop_width
    )

    # [all rows, selected coloumn: 0 is x_min, 1 is y_min, etc.]
    box_widths = bbox[:, 2] - bbox[:, 0]
    box_heights = bbox[:, 3] - bbox[:, 1]

    can_fit = tf.logical_and(box_widths <= crop_width, box_heights <= crop_height)

    valid_indices = tf.where(can_fit)[:, 0]
    n_valid = tf.shape(valid_indices)[0]

    # if it's not possible to apply any positive_crop, then use a full_image
    def fallback():
        return load_dataset(image_path, classes, bbox)

    # if it's possible, build a positive crop
    def make_positive_crop():
        # select a random cone of the image
        random_index = tf.random.uniform([], 0, n_valid, dtype=tf.int32)
        selected_cone = valid_indices[random_index]

        coord_cone = bbox[selected_cone]
        x1 = coord_cone[0]
        y1 = coord_cone[1]
        x2 = coord_cone[2]
        y2 = coord_cone[3]

        # Valid interval for the top-left corner of the crop.
        # Any point chosen in these intervals guarantees that
        # the selected cone is completely inside the crop.
        min_crop_x = tf.maximum(0, tf.cast(tf.math.ceil(x2 - crop_width), tf.int32))
        max_crop_x = tf.minimum(tf.cast(tf.math.floor(x1), tf.int32), image_width - crop_width)

        min_crop_y = tf.maximum(0, tf.cast(tf.math.ceil(y2 - crop_height), tf.int32))
        max_crop_y = tf.minimum(tf.cast(tf.math.floor(y1), tf.int32), image_height - crop_height)

        # crop in this range that guarantees that the cone is inside the crop
        crop_x = tf.random.uniform([], min_crop_x, max_crop_x + 1, dtype=tf.int32)
        crop_y = tf.random.uniform([], min_crop_y, max_crop_y + 1, dtype=tf.int32)

        # apply the crop
        image_crop = tf.image.crop_to_bounding_box(image, crop_y, crop_x, crop_height, crop_width)

        # now we need to move the bounding box w.r.t. the crop we applied
        offset = tf.cast([crop_x, crop_y, crop_x, crop_y], tf.float32)
        crop_bbox = bbox - offset

        clipped_bbox, retained_area = clip_boxes_and_compute_retained_area(bbox, crop_bbox, crop_width, crop_height)
        valid = retained_area >= MIN_RETAINED_AREA

        clipped_bbox = tf.boolean_mask(clipped_bbox, valid)
        cropped_classes = tf.boolean_mask(classes, valid)

        return {
            "images": image_crop,
            "bounding_boxes": {
                "boxes": clipped_bbox,
                "classes": cropped_classes
            }
        }

    return tf.cond(
        tf.logical_and(can_crop, n_valid > 0),
        make_positive_crop,
        fallback
        )

def clip_boxes_and_compute_retained_area(bbox, crop_bbox, crop_width, crop_height):
    bx1 = tf.clip_by_value(crop_bbox[:, 0], 0.0, crop_width)
    by1 = tf.clip_by_value(crop_bbox[:, 1], 0.0, crop_height)
    bx2 = tf.clip_by_value(crop_bbox[:, 2], 0.0, crop_width)
    by2 = tf.clip_by_value(crop_bbox[:, 3], 0.0, crop_height)

    clipped_bbox = tf.stack([bx1, by1, bx2, by2], axis=-1)

    original_area = ((bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]))

    cropped_area = (tf.maximum(0.0, bx2 - bx1) * tf.maximum(0.0, by2 - by1))
    retained_area = (cropped_area / tf.maximum(original_area, 1e-6))

    return clipped_bbox, retained_area

def negative_crop(image_path, classes, bbox):
    """
    It crops an image such that no cone is present
    """
    image = load_image(image_path)

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    crop_height = IMAGE_SIZE[0]
    crop_width = IMAGE_SIZE[1]

    can_crop = tf.logical_and(
        image_height >= crop_height,
        image_width >= crop_width
    )

    def make_negative_crop():
        # Generate several random candidate crops.
        crop_xs = tf.random.uniform([NUM_NEGATIVE_CROP_ATTEMPTS], 0, image_width - crop_width + 1, dtype=tf.int32)
        crop_ys = tf.random.uniform([NUM_NEGATIVE_CROP_ATTEMPTS], 0, image_height - crop_height + 1, dtype=tf.int32)

        crop_xs_float = tf.cast(crop_xs, bbox.dtype)
        crop_ys_float = tf.cast(crop_ys, bbox.dtype)

        crop_x2s = crop_xs_float + crop_width
        crop_y2s = crop_ys_float + crop_height

        # Intersection between every candidate crop and every bbox.
        ix1 = tf.maximum(crop_xs_float[:, None], bbox[None, :, 0])
        iy1 = tf.maximum(crop_ys_float[:, None], bbox[None, :, 1])
        ix2 = tf.minimum(crop_x2s[:, None], bbox[None, :, 2])
        iy2 = tf.minimum(crop_y2s[:, None], bbox[None, :, 3])

        intersection_width = tf.maximum(0.0, ix2 - ix1)
        intersection_height = tf.maximum(0.0, iy2 - iy1)

        intersection_area = intersection_width * intersection_height

        # A candidate is negative only if it intersects no bbox.
        valid_candidates = tf.reduce_all(intersection_area == 0.0, axis=1)

        valid_indices = tf.where(valid_candidates)[:, 0]
        n_valid = tf.shape(valid_indices)[0]

        def use_negative_crop():
            random_index = tf.random.uniform([], 0, n_valid, dtype=tf.int32)

            selected = valid_indices[random_index]

            crop_x = crop_xs[selected]
            crop_y = crop_ys[selected]

            image_crop = tf.image.crop_to_bounding_box(image, crop_y, crop_x, crop_height, crop_width)

            return {
                "images": image_crop,
                "bounding_boxes": {
                    "boxes": tf.zeros([0, 4], dtype=bbox.dtype), #no bboxes
                    "classes": tf.zeros([0], dtype=classes.dtype) #no classes
                }
            }

        def fallback():
            return load_dataset(image_path, classes, bbox)

        return tf.cond(n_valid > 0,
            use_negative_crop,
            fallback
        )

    def fallback():
        return load_dataset(image_path, classes, bbox)

    return tf.cond(can_crop,
        make_negative_crop,
        fallback
    )

@tf.function
def load_train_dataset(image_path, classes, bbox):
    n = tf.random.uniform([])
    if n < 0.5:
        return load_dataset(image_path, classes, bbox)
    elif n < 0.90:
        return positive_crop(image_path, classes, bbox)
    else:
        return negative_crop(image_path, classes, bbox)


def build_train_dataset(train_data, num_train):
    """
    It describes the train dataset.
    It substitutes the paths in the dataset with the corrisponding images.
    Images in the training set can be of three types:
    1) full_image resize from its dimension to IMAGE_SIZE
    2) positive_crop: we crop an image to IMAGE_SIZE with at least 1 cone (the cone is in a random position of the crop)
    3) negative_crop: we crop an image to IMAGE_SIZE without any cone.

    We also specify charateristics of the set during the training:
    reshuffle, ragged_batch, prefetch.
    """
    
    # Shuffle the training samples at each epoch.
    train_data = train_data.shuffle(
        buffer_size=num_train,
        seed=SEED,
        reshuffle_each_iteration=True
    )

    #load full_image or crop_image
    train_ds = train_data.map(
        load_train_dataset,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        deterministic=False,
    )


    # we fix the image in 800x800
    # NOTe: crop image are already 800x800, while the full_image are not
    train_ds = train_ds.map(
        resize_sample,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        deterministic=False,
    )

    # group elems in batch
    # drop_remainder=True: discard the last elems if they do not fill a complete batch
    train_ds = train_ds.ragged_batch(BATCH_SIZE,drop_remainder=True)

    # FORMAT CONVERSION:
    # the model expects a tuple format (images, bounding_boxes), not a dictionary
    train_ds = train_ds.map(
        dict_to_tuple,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        deterministic=False,
    )

    # Prefetch prepares data in advance while the model is training
    # it reduces the waiting time
    train_ds = train_ds.prefetch(PREFETCH_BUFFER)

    train_options = tf.data.Options()
    train_options.experimental_deterministic = False

    train_ds = train_ds.with_options(train_options)

    return train_ds

def resize_sample(sample):
    """
    it resizes the sample from its original size to IMAGE_SIZExIMAGE_SIZE
    """
    image = sample["images"]
    boxes = sample["bounding_boxes"]["boxes"]
    classes = sample["bounding_boxes"]["classes"]

    image, scale_x, scale_y, pad_x, pad_y = resize_with_letterbox(image, IMAGE_SIZE)

    boxes = tf.cast(boxes, tf.float32)

    x_min = boxes[:, 0] * scale_x + tf.cast(pad_x, tf.float32)
    y_min = boxes[:, 1] * scale_y + tf.cast(pad_y, tf.float32)
    x_max = boxes[:, 2] * scale_x + tf.cast(pad_x, tf.float32)
    y_max = boxes[:, 3] * scale_y + tf.cast(pad_y, tf.float32)

    boxes = tf.stack([x_min, y_min, x_max, y_max], axis=-1)

    return {
        "images": image,
        "bounding_boxes": {
            "classes": classes,
            "boxes": boxes,
        }
    }

def build_val_loss_dataset(val_data):
    """
    validation set built in order to check whether overfitting may occur. This set will be test during the training just by
    looking at the loss function that it will generate the problem
    """
    # Upload the images and fix the size to IMAGE_SIZE
    val_ds = val_data.map(load_dataset, num_parallel_calls=1)
    val_ds = val_ds.map(resize_sample, num_parallel_calls=1)

    # group in batch and bring the format in tuple form.
    val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=False)
    val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=1)

    val_ds = val_ds.prefetch(1)

    return val_ds

def _create_combined_views_numpy(image):
    image = image.numpy()
    views, _ = create_combined_views(image)

    return views.astype(np.float32)

def create_inference_views(image_path):
    """
    here we substitute the image_path with the corresponding image.
    then we create al
    """
    image = load_image(image_path)

    views = tf.py_function(
        func=_create_combined_views_numpy,
        inp=[image],
        Tout=tf.float32,
    )

    views.set_shape([
        None,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3,
    ])

    return views

def build_inference_dataset(image_paths):
    """
    the validation inference set is used to verify the real behaviour of the model. this dataset will be composed
    by patches of IMAGE_SIZE dimension and the original full image resize to IMAGE_SIZE.
    This set is processed by the model as the test_set will be evaluated.
    """

    ds = tf.data.Dataset.from_tensor_slices(image_paths)

    ds = ds.map(create_inference_views,num_parallel_calls=1)

    # prima:
    # elemento 0 -> [N0, 800, 800, 3]
    # elemento 1 -> [N1, 800, 800, 3]
    #
    # dopo:
    # view 0 -> [800, 800, 3]
    # view 1 -> [800, 800, 3]
    # ...

    ds = ds.unbatch()

    ds = ds.batch(PATCH_BATCH_SIZE, drop_remainder=False) #validation to all the images of the inference_dataset
    ds = ds.prefetch(1)

    return ds

def build_inference_metadata(image_shapes):
    """
    list of informations about the inference dataset.
    It's needed in order to obtain information about the predictions of the images.
    """
    all_metadata = []

    patch_height, patch_width = IMAGE_SIZE
    target_height, target_width = IMAGE_SIZE

    for image_index, shape in enumerate(image_shapes.numpy()):

        image_height = int(shape[0])
        image_width = int(shape[1])

        # we obtain a list of starting position of all the patches w.r.t. a single image
        x_starts = get_patch_starts(image_width, patch_width, PATCH_STRIDE)
        y_starts = get_patch_starts(image_height, patch_height, PATCH_STRIDE)

        local_view_index = 0

        # PATCHES metadata
        for y_start in y_starts:
            for x_start in x_starts:
                valid_width = min(patch_width, image_width - x_start)
                valid_height = min(patch_height, image_height - y_start)

                all_metadata.append({
                    "view_index": len(all_metadata),
                    "image_index": image_index,
                    "local_view_index": local_view_index,

                    "is_patch": True,

                    "offset_x": x_start,
                    "offset_y": y_start,

                    "valid_width": valid_width,
                    "valid_height": valid_height,

                    "image_width": image_width,
                    "image_height": image_height,
                })

                local_view_index += 1

        # FULL IMAGE metadata
        scale = min(target_width / image_width, target_height / image_height)

        resized_width = int(np.round(image_width * scale))
        resized_height = int(np.round(image_height * scale))

        scale_x = resized_width / image_width
        scale_y = resized_height / image_height

        pad_x = (target_width - resized_width) // 2
        pad_y = (target_height - resized_height) // 2

        all_metadata.append({
            "view_index": len(all_metadata),
            "image_index": image_index,
            "local_view_index": local_view_index,

            "is_patch": False,

            "scale_x": scale_x,
            "scale_y": scale_y,

            "pad_x": pad_x,
            "pad_y": pad_y,

            "image_width": image_width,
            "image_height": image_height,
        })

    return all_metadata