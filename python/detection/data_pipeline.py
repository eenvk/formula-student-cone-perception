#granati

import tensorflow as tf
import keras_cv
import cv2
import numpy as np
import random

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

    # 6. select a random number of indices, that are the 20% of the validation set
    selected_inference_indices = select_inference_subset_indices(tf.shape(val_image_paths)[0], 0.2)
    print("Number of images used during training to see inference the model", len(selected_inference_indices))

    # take the selected images
    selected_inference_paths = tf.gather(val_image_paths, selected_inference_indices)
    selected_inference_images_shapes = tf.gather(val_image_shapes, selected_inference_indices)

    # define the real inference dataset and taking also the metadata related to it
    val_inference_ds = build_inference_dataset(selected_inference_paths)
    val_inference_metadata = build_inference_metadata(selected_inference_images_shapes)

    print("Inference metadata:", len(val_inference_metadata))

    selected_val_bboxes = tf.gather(val_bboxes,selected_inference_indices)
    selected_val_classes = tf.gather(val_classes, selected_inference_indices)

    # we build also the solutions of the validation set because this information are not present in the inference_ds
    inference_y_true = {
        "boxes": selected_val_bboxes,
        "classes": selected_val_classes
        }

    return train_ds, val_loss_ds, val_inference_ds, val_inference_metadata, inference_y_true

def prepare_dataset_data(pairs):
    """
    Reads the annotations using dataset_utils and converts them
    into the format needed to create a TensorFlow dataset.

    Returns:
        tuple: (image_paths, classes, bboxes, image_shapes)
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

    image_paths = tf.constant(image_paths, dtype=tf.string)
    bbox = tf.ragged.constant(all_boxes, dtype=tf.float32, ragged_rank=1)
    classes = tf.ragged.constant(all_classes, dtype=tf.float32, ragged_rank=1)

    image_shapes = tf.constant(image_shapes, dtype= tf.int32)

    return image_paths, classes, bbox, image_shapes


@tf.function
def load_image(image_path):
    """
    Loads an image file and decodes it into a tensor.
    
    Reads an image from disk and decodes it into a 3-channel
    RGB tensor. Automatically detects the image format based on file extension.

    Returns:
        tf.Tensor: Decoded image tensor with shape [height, width, 3]
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
    
    Returns:
        dict: Dictionary containing the image tensor and bounding box annotations.
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
    """
    return (
        inputs["images"],
        inputs["bounding_boxes"],
    )



def build_data_structure(image_paths, classes, bboxes):
    """
    build the element of the dataset. Each element is formed by its
    image_path, class, bbox

    Returns:
        tf.data.Dataset: A TensorFlow dataset where each element is a tuple of (image_path, classes, bboxes)
    """
    return tf.data.Dataset.from_tensor_slices((image_paths, classes, bboxes))

def positive_crop(image_path, classes, bbox):
    """
    it applies the positive crop: crop the image to IMAGE_SIZExIMAGE_SIZE with at least 1 cone.
    the cone should be in a random position w.r.t. the crop and it needs to appear with at least a
    MIN_RETAINED_AREA, otherwise another crop should be considered.
    If it's not possible to apply a positive crop, then the full image is used instead.

    Returns:
        dict: Dictionary containing the cropped image tensor and adjusted bounding box annotations.
    """
    image = load_image(image_path)

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    crop_height = IMAGE_SIZE[0]
    crop_width = IMAGE_SIZE[1]

    can_crop = tf.logical_and(image_height >= crop_height, image_width >= crop_width)

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

        # clip the bounding boxes to the crop area and compute the retained area ratio
        clipped_bbox, retained_area = clip_boxes_and_compute_retained_area(bbox, crop_bbox, crop_width, crop_height)
        valid = retained_area >= MIN_RETAINED_AREA

        # filter out boxes that are not valid (i.e., those that do not meet the minimum retained area requirement)
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
    """
    Clips bounding boxes to the crop area and computes the retained area ratio.

    Returns:
        tuple: (clipped_bbox, retained_area) where clipped_bbox is the adjusted bounding 
                    boxes and retained_area is the ratio of the area retained after clipping.
    """
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
    It crops an image such that no cone is present, i.e., the crop does not intersect with any bounding box.
    """
    image = load_image(image_path)

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    crop_height = IMAGE_SIZE[0]
    crop_width = IMAGE_SIZE[1]

    can_crop = tf.logical_and(image_height >= crop_height, image_width >= crop_width)

    def make_negative_crop():
        """
        Generates a negative crop that does not intersect with any bounding box.
        It tries NUM_NEGATIVE_CROP_ATTEMPTS random crops and selects one that does not intersect with any bounding box.
        If no valid crop is found, it falls back to loading the full image.

        Returns:
            dict: Dictionary containing the negative cropped image tensor and empty bounding box annotations.
        """
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

        def fallback():
            return load_dataset(image_path, classes, bbox)

        def use_negative_crop():
            """
            Selects a valid negative crop and returns it along with empty bounding boxes and classes.
            """
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


        # if there are valid negative crops, use one; otherwise, fall back to loading the full image.
        return tf.cond(n_valid > 0,
            use_negative_crop,
            fallback
        )

    def fallback():
        return load_dataset(image_path, classes, bbox)

    # if the image is large enough to crop, attempt to create a negative crop; otherwise, load the full image.
    return tf.cond(can_crop,
        make_negative_crop,
        fallback
    )

@tf.function
def load_train_dataset(image_path, classes, bbox):
    """
    Randomly selects one of three strategies for loading a training sample
    """
    n = tf.random.uniform([])
    if n < 0.5:
        return load_dataset(image_path, classes, bbox)
    elif n < 0.90:
        return positive_crop(image_path, classes, bbox)
    else:
        return negative_crop(image_path, classes, bbox)

def augment_sample(sample):
    """
    Applies random data augmentation without increasing dataset size.

    Returns:
        dict: Dictionary containing the augmented image tensor and bounding box annotations.
    """
    images = sample["images"]
    bounding_boxes = sample["bounding_boxes"]

    if tf.random.uniform(()) < 0.5:
        images = tf.image.random_brightness(images, max_delta=25.5)

    if tf.random.uniform(()) < 0.5:
        images = tf.image.random_contrast(images, lower=0.9, upper=1.1)

    if tf.random.uniform(()) < 0.5:
        images = tf.image.random_saturation(images, lower=0.9, upper=1.1)

    # apply clip to ensure pixel values remain in the valid range after augmentation
    images = tf.clip_by_value(images, 0.0, 255.0)

    return {
        "images": images,
        "bounding_boxes": bounding_boxes
    }

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

    Returns:
        tf.data.Dataset: A TensorFlow dataset ready for training with images and bounding box annotations.
    """
    
    # Shuffle the training samples at each epoch.
    train_data = train_data.shuffle(
        buffer_size=num_train,
        seed=SEED,
        reshuffle_each_iteration=True
    )

    #load full_image or crop_image
    train_ds = train_data.map(load_train_dataset,num_parallel_calls=NUM_PARALLEL_CALLS,deterministic=False)
    # we fix the image in 800x800
    # NOTe: crop image are already 800x800, while the full_image are not, so we need to resize them to 800x800
    train_ds = train_ds.map(resize_sample, num_parallel_calls=NUM_PARALLEL_CALLS,deterministic=False)
    train_ds = train_ds.map(augment_sample, num_parallel_calls=NUM_PARALLEL_CALLS, deterministic=False)

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

    # set the dataset options to allow non-deterministic execution for better performance
    train_options = tf.data.Options()
    train_options.experimental_deterministic = False

    train_ds = train_ds.with_options(train_options)

    return train_ds

def resize_sample(sample):
    """
    it resizes the sample from its original size to IMAGE_SIZExIMAGE_SIZE
    keep the aspect ratio and add letterbox padding if needed.

    Returns:
        dict: Dictionary containing the resized image tensor and adjusted bounding box annotations.
    """
    image = sample["images"]
    boxes = sample["bounding_boxes"]["boxes"]
    classes = sample["bounding_boxes"]["classes"]

    # resize the image to IMAGE_SIZE while preserving aspect ratio and adding letterbox padding
    image, scale_x, scale_y, pad_x, pad_y = resize_with_letterbox(image, IMAGE_SIZE)

    boxes = tf.cast(boxes, tf.float32)

    # adjust the bounding boxes according to the scaling and padding applied to the image
    x_min = boxes[:, 0] * scale_x + tf.cast(pad_x, tf.float32)
    y_min = boxes[:, 1] * scale_y + tf.cast(pad_y, tf.float32)
    x_max = boxes[:, 2] * scale_x + tf.cast(pad_x, tf.float32)
    y_max = boxes[:, 3] * scale_y + tf.cast(pad_y, tf.float32)

    # stack the adjusted bounding box coordinates into a single tensor
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

    Returns:
        tf.data.Dataset: A TensorFlow dataset for validation loss evaluation with images and bounding box annotations.
    """
    # Upload the images and fix the size to IMAGE_SIZE
    val_ds = val_data.map(load_dataset, num_parallel_calls=1)
    val_ds = val_ds.map(resize_sample, num_parallel_calls=1)

    # group in batch and bring the format in tuple form.
    val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=False)
    val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=1)

    val_ds = val_ds.prefetch(1)

    return val_ds

def create_inference_views(image_path):
    """
    here we substitute the image_path with the corresponding image.
    the image is resized to IMAGE_SIZExIMAGE_SIZE and then we create the views of the image:
    1) if the image is not larger than IMAGE_SIZE in both dimensions:
       only the full-image view is used;
    2) if four native IMAGE_SIZE patches can cover the whole image:
         use them without resize;
    3) otherwise split the image into four quadrants covering the whole image
         and letterbox each quadrant to IMAGE_SIZE, preserving aspect ratio.
        
    Returns:
        tf.Tensor: A tensor containing the combined views of the image, each resized to IMAGE_SIZE.
    """
    image = load_image(image_path)

    views = create_combined_views(image)

    views.set_shape([
        None,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3,
    ])

    return views

def build_inference_dataset(image_paths):
    """
    Validation inference dataset.

    For images larger than IMAGE_SIZE in both dimensions:
    - exactly 4 patch views are created;
    - native IMAGE_SIZE patches are used when they cover the whole image;
    - otherwise four full-coverage quadrants are letterboxed to IMAGE_SIZE;
    - the full-image letterboxed view is always added.

    Therefore a large image produces exactly 5 views total.

    Returns:
        tf.data.Dataset: A TensorFlow dataset for inference evaluation with images resized to IMAGE_SIZE.
    """
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    # prima:
    # elemento 0 -> [N0, 800, 800, 3]
    # elemento 1 -> [N1, 800, 800, 3]
    #
    # dopo:
    # view 0 -> [800, 800, 3]
    # view 1 -> [800, 800, 3]
    # ...
    ds = ds.map(create_inference_views, num_parallel_calls=1)

    ds = ds.unbatch()

    ds = ds.batch(PATCH_BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(1)

    return ds

def build_inference_metadata(image_shapes):
    """
    Build metadata for inference views using the hybrid 4-patch strategy.

    Strategy:
    - if the image is not larger than IMAGE_SIZE in both dimensions:
      only the full-image view is used;
    - if four native IMAGE_SIZE patches can cover the whole image:
      use them without resize;
    - otherwise split the image into four quadrants covering the whole image
      and letterbox each quadrant to IMAGE_SIZE, preserving aspect ratio.

    The full-image view is always appended.

    Returns:
        list: A list of dictionaries containing metadata for each inference view.
    """
    all_metadata = []

    patch_height, patch_width = IMAGE_SIZE
    target_height, target_width = IMAGE_SIZE

    for image_index, shape in enumerate(image_shapes.numpy()):
        image_height = int(shape[0])
        image_width = int(shape[1])

        local_view_index = 0

        if image_height > patch_height and image_width > patch_width:
            use_native_patches = ( image_width <= 2 * patch_width and image_height <= 2 * patch_height)

            if use_native_patches:
                x_starts = [0, image_width - patch_width]
                y_starts = [0, image_height - patch_height]

                for y_start in y_starts:
                    for x_start in x_starts:
                        all_metadata.append({
                            "view_index": len(all_metadata),
                            "image_index": image_index,
                            "local_view_index": local_view_index,

                            "is_patch": True,
                            "is_resized_patch": False,

                            "offset_x": x_start,
                            "offset_y": y_start,

                            "source_width": patch_width,
                            "source_height": patch_height,

                            "valid_width": patch_width,
                            "valid_height": patch_height,

                            "scale_x": 1.0,
                            "scale_y": 1.0,
                            "pad_x": 0,
                            "pad_y": 0,

                            "image_width": image_width,
                            "image_height": image_height,
                        })

                        local_view_index += 1

            else:
                split_x = image_width // 2
                split_y = image_height // 2

                quadrants = [
                    (0, 0, split_x, split_y),
                    (split_x, 0, image_width - split_x, split_y),
                    (0, split_y, split_x, image_height - split_y),
                    (split_x, split_y, image_width - split_x, image_height - split_y),
                ]

                for x_start, y_start, source_width, source_height in quadrants:
                    scale = min(
                        patch_width / source_width,
                        patch_height / source_height
                    )

                    resized_width = int(np.round(source_width * scale))
                    resized_height = int(np.round(source_height * scale))

                    scale_x = resized_width / source_width
                    scale_y = resized_height / source_height

                    pad_x = (patch_width - resized_width) // 2
                    pad_y = (patch_height - resized_height) // 2

                    all_metadata.append({
                        "view_index": len(all_metadata),
                        "image_index": image_index,
                        "local_view_index": local_view_index,

                        "is_patch": True,
                        "is_resized_patch": True,

                        "offset_x": x_start,
                        "offset_y": y_start,

                        "source_width": source_width,
                        "source_height": source_height,

                        "valid_width": patch_width,
                        "valid_height": patch_height,

                        "scale_x": scale_x,
                        "scale_y": scale_y,
                        "pad_x": pad_x,
                        "pad_y": pad_y,

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

def select_inference_subset_indices(num_images, ratio=0.2):
    num_selected = tf.cast(
        tf.math.ceil(tf.cast(num_images, tf.float32) * ratio),
        tf.int32
    )

    indices = tf.range(num_images)

    shuffled_indices = tf.random.experimental.stateless_shuffle(
        indices,
        seed=[SEED, 0]
    )

    return shuffled_indices[:num_selected]



def get_patch_starts(image_size, patch_size, stride):
    """
    Return at most two starting coordinates along one axis.

    The signature is kept unchanged so the rest of the project keeps working.

    If the image dimension is not larger than the patch dimension, no patch
    position is returned.

    If the image dimension is larger than the patch dimension, two positions
    are used:
        - the beginning of the image
        - the end of the image

    Therefore, when BOTH image dimensions are larger than IMAGE_SIZE,
    x_starts and y_starts each contain two values and exactly 4 patches
    are generated.

    `stride` is intentionally kept only for compatibility.
    """
    del stride

    if image_size <= patch_size:
        return []

    return [0, image_size - patch_size]

def resize_with_letterbox(image, target_size):
    """
    Resize an image to a target size while preserving aspect ratio and adding letterbox padding.
    we need to keep the scale_x and scale_y because we need to adjust the bounding boxes w.r.t. the new image size.
    and we need to keep the pad_x and pad_y because we need to adjust the bounding boxes w.r.t. the new image size.

    1. Compute the scaling factor to fit the image within the target size while preserving aspect ratio.
    2. Resize the image using the computed scaling factor.
    3. Compute the padding needed to reach the target size.
    4. Pad the resized image to the target size.
    5. Return the resized and padded image along with the scaling factors and padding values.

    Returns:
        tuple: (resized_image, scale_x, scale_y, pad_x, pad_y) where resized_image is the image resized to target_size with letterbox padding, 
               scale_x and scale_y are the scaling factors applied to the original image dimensions, and pad_x and pad_y are the padding values 
               added to the resized image to reach the target size.
    """

    target_h, target_w = target_size

    image_size = tf.cast(tf.shape(image)[:2], tf.float32)
    image_height = image_size[0]
    image_width = image_size[1]

    # Compute the scaling factor to fit the image within the target size while preserving aspect ratio.
    scale = tf.minimum(target_w / image_width, target_h / image_height)

    # Resize the image using the computed scaling factor.
    resized_size_float = tf.round(image_size * scale)
    resized_size = tf.cast(resized_size_float, tf.int32)

    # Compute the padding needed to reach the target size.
    resized_h = resized_size[0]
    resized_w = resized_size[1]

    # Resize the image to the new size
    image = tf.image.resize(image, resized_size)

    # Compute the scaling factors and padding values for adjusting bounding boxes.
    scale_y = resized_size_float[0] / image_height
    scale_x = resized_size_float[1] / image_width

    # Compute the padding needed to center the resized image within the target size.
    pad_y = (target_h - resized_h) // 2
    pad_x = (target_w - resized_w) // 2

    # Pad the resized image to the target size using the computed padding values.
    image = tf.image.pad_to_bounding_box(image, pad_y, pad_x, target_h, target_w)

    return image, scale_x, scale_y, pad_x, pad_y


def create_combined_views(image):
    """
    Create a combined set of views for an image, including patches and the full image.
    If the image is larger than IMAGE_SIZE in both dimensions, create patches and include the full image.
    Otherwise, only include the full image.
    
    Returns:
        Tensor: A tensor containing the combined views of the image.
    """
    image = tf.cast(image, tf.float32)

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    patch_height, patch_width = IMAGE_SIZE

    # Determine whether to create patches or just use the full image based on the image dimensions.
    full_image, _, _, _, _ = resize_with_letterbox(image, IMAGE_SIZE)
    full_image = tf.expand_dims(full_image, axis=0)

    def full_image_only():
        return full_image

    def patches_and_full_image():
        patches = create_patches(image, IMAGE_SIZE)
        return tf.concat([patches, full_image], axis=0)

    return tf.cond(
        tf.logical_and(image_height > patch_height, image_width > patch_width),
        patches_and_full_image,
        full_image_only
    )

def create_patches(image, patch_size=IMAGE_SIZE):
    """
    Create patches from an image based on the specified patch size.
    If the image is larger than the patch size in both dimensions, create patches.
    Otherwise, return the full image as a single patch.

    Returns:
        Tensor: A tensor containing the patches of the image.
    """
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    patch_height, patch_width = patch_size

    # Determine whether to use native patches or resized quadrants based on the image dimensions.
    use_native_patches = tf.logical_and(image_width <= 2 * patch_width, image_height <= 2 * patch_height)

    def native_patches():
        x_starts = [0, image_width - patch_width]
        y_starts = [0, image_height - patch_height]

        patches = []

        for y in y_starts:
            for x in x_starts:
                # Crop the image to the specified patch size starting from (x, y).
                patch = tf.image.crop_to_bounding_box(image, y, x, patch_height, patch_width)
                patches.append(patch)

        return tf.stack(patches)

    def quadrant_patches():
        # Split the image into four quadrants and resize each quadrant to the patch size while preserving aspect ratio.
        split_x = image_width // 2
        split_y = image_height // 2

        # Define the four quadrants of the image based on the split coordinates.
        # Each quadrant is represented as (x_start, y_start, width, height).
        quadrants = [
            (0, 0, split_x, split_y),
            (split_x, 0, image_width - split_x, split_y),
            (0, split_y, split_x, image_height - split_y),
            (split_x, split_y, image_width - split_x, image_height - split_y),
        ]

        patches = []

        # For each quadrant, crop the image and resize it to the patch size while preserving aspect ratio.
        for x, y, width, height in quadrants:
            crop = tf.image.crop_to_bounding_box(image, y, x, height, width)
            crop, _, _, _, _ = resize_with_letterbox(crop, patch_size)
            patches.append(crop)

        return tf.stack(patches)

    return tf.cond(
        use_native_patches,
        native_patches,
        quadrant_patches
    )