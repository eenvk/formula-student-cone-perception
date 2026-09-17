#Granati

import tensorflow as tf

from keras_cv.src.models.object_detection.yolo_v8 import yolo_v8_detector
from detection.detection_config import IMAGE_SIZE, yolo_SCORE_THRESHOLD, yolo_PRE_NMS_TOP_K, yolo_NMS_IOU_THRESHOLD, yolo_NMS_MAX_DETECTIONS

IMAGE_HEIGHT = IMAGE_SIZE[0]
IMAGE_WIDTH = IMAGE_SIZE[1]

"""
    params in yolo_v8 implementation, we add  a top_k parameter to limit the number of candidates passed to NMS.
    This is important because yolo can produce a large number of candidate boxes, but for real-time applications,
    we want to limit the number of candidates to a manageable number before applying NMS.
"""

def create_yolo_inference(model):
    """
    Create a reusable inference function bound to the given model.
    The returned function takes a batch of images and returns the predicted boxes, scores, and classes after 
    applying candidate filtering and non-maximum suppression (NMS).
    As input we have a model that outputs raw predictions
    
    Returns:
        A function that takes a batch of images and returns the filtered predictions.
    """
    implementation = get_yolo_implementation()

    # Define a TensorFlow function for graph execution to improve performance, tensorflow optimization
    @tf.function(input_signature=[tf.TensorSpec(
            shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            dtype=tf.float32,
            )
        ]
    )
    def graph_predict(images):
        """
        Perform inference on a batch of images and return the filtered predictions.
        """
        raw_predictions = model(images, training=False) # training=False, because we are in inference mode

        boxes = decode_boxes(raw_predictions["boxes"], images, implementation)
        boxes, scores, classes = select_candidates(boxes, raw_predictions["classes"])

        predictions = apply_nms(boxes, scores, classes)
        predictions["boxes"] = implementation.bounding_box.convert_format(predictions["boxes"], source="xyxy",
                                                                          target=model.bounding_box_format, images=images)

        return predictions

    def infer(inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        predictions = graph_predict(inputs)

        return tf.nest.map_structure(tensor_to_numpy, predictions)

    return infer


def get_yolo_implementation():
    """Create a reusable inference function bound to the given model."""
    implementation = yolo_v8_detector

    required_functions = ("decode_regression_to_boxes", "get_anchors", "dist2bbox", "bounding_box", "ops")

    # Check that the implementation has all the required functions.
    for name in required_functions:
        if not hasattr(implementation, name):
            raise RuntimeError(f"The installed yolo implementation is missing: {name}")

    return implementation


def decode_boxes(raw_boxes, images, implementation):
    """
    Decode the raw box predictions from the yolo model into actual bounding boxes.
    
    Returns:
        A tensor of shape (batch_size, num_boxes, 4) containing the decoded bounding boxes in the format (x1, y1, x2, y2).
    """
    # now we use the yolo implementation to decode the raw boxes into actual bounding boxes
    # then we will modify the thresholding and NMS to filter out low-confidence boxes before applying NMS
    distances = implementation.decode_regression_to_boxes(raw_boxes)

    # get_anchors returns the anchors and strides for the given image shape;
    # anchors are the predefined bounding boxes used by yolo, and strides are the scaling factors for the feature maps.
    anchors, strides = implementation.get_anchors(image_shape=images.shape[1:])
    # The strides are expanded to match the shape of the distances tensor for broadcasting during multiplication.
    strides = implementation.ops.expand_dims(strides, axis=-1)

    # from the implementation we know:
    # Decodes distance predictions into xyxy boxes. Input left / top / right / bottom predictions are transformed 
    # into xyxy box predictions based on anchor points. The resulting xyxy predictions must be scaled by the stride of their
    # corresponding anchor points to yield an absolute xyxy box.

    boxes = implementation.dist2bbox(distances, anchors)

    return boxes * strides #scale


def select_candidates(boxes, class_scores):
    """
    Keep at most TOP_K candidates per image before running NMS.
    Returns:
        boxes: A tensor of shape (batch_size, top_k, 4) containing the selected bounding boxes.
        scores: A tensor of shape (batch_size, top_k) containing the selected confidence scores.
        classes: A tensor of shape (batch_size, top_k) containing the selected class indices.
    """
    #(batch_size, num_boxes, num_classes) = class_scores.shape

    scores = tf.reduce_max(class_scores, axis=-1) # keep the best score for each box
    classes = tf.argmax(class_scores, axis=-1, output_type=tf.int32) # keep the class with the best score for each box

    # Candidates below the threshold cannot become valid detections.
    valid_scores_mask = scores >= yolo_SCORE_THRESHOLD # boolean mask

    invalid_scores = tf.fill(tf.shape(scores), tf.cast(-1.0, scores.dtype)) #auxiliary
    filtered_scores = tf.where(valid_scores_mask, scores, invalid_scores)

    # TOP_K bounds the number of candidates passed to NMS.
    num_candidates = tf.shape(filtered_scores)[1]
    top_k = tf.minimum(num_candidates, yolo_PRE_NMS_TOP_K)

    selected_scores, selected_indices = tf.math.top_k(filtered_scores, k=top_k, sorted=True)

    selected_boxes = tf.gather(boxes, selected_indices, batch_dims=1)
    selected_classes = tf.gather(classes, selected_indices, batch_dims=1)

    return selected_boxes, selected_scores, selected_classes


def prepare_nms_boxes(boxes, classes):
    """
    Convert xyxy to yxyx and apply the existing class-offset strategy
    to ensure that boxes from different classes do not suppress each other during NMS.
    Returns:
        A tensor of shape (batch_size, num_boxes, 4) containing the modified bounding boxes for NMS.
    """
    # no_max_suppression needs boxes in yxyx format, so we convert them from xyxy to yxyx, for each box ...
    boxes_yxyx = tf.stack(
        [
            boxes[..., 1],
            boxes[..., 0],
            boxes[..., 3],
            boxes[..., 2],
        ],
        axis=-1,
    )

    # Keep the same offset calculation used by the original decoder.
    offset_size = tf.reduce_max(tf.abs(boxes_yxyx), axis=[1, 2], keepdims=True) + 1.0

    class_offsets = (tf.cast(classes, boxes_yxyx.dtype)[..., None] * offset_size)

    return boxes_yxyx + class_offsets


def nms_single_image(inputs):
    """
    Apply NMS to one image and pad the selected indices
    Returns:
        selected_indices: A tensor of shape (yolo_NMS_MAX_DETECTIONS,) containing the indices of the selected boxes after NMS, padded with -1 for unused slots.
        num_detections: A scalar tensor indicating the number of valid detections for this image.
    """

    boxes, scores = inputs

    return tf.image.non_max_suppression_padded(
        boxes=boxes,
        scores=scores,
        max_output_size=yolo_NMS_MAX_DETECTIONS,
        iou_threshold=yolo_NMS_IOU_THRESHOLD,
        score_threshold=yolo_SCORE_THRESHOLD,
        pad_to_max_output_size=True,
    )


def apply_nms(boxes, scores, classes):
    """
    Apply NMS to a batch of images and return the selected boxes, scores, and classes,
    padded to the maximum number of detections.

    Returns:
        A dictionary containing the selected boxes, scores, classes, and the number of detections for each image in the batch.
    """

    # Prepare boxes for NMS by converting to yxyx format and applying class offsets to prevent suppression across classes.
    # in this way different classes will not suppress each other during NMS, which is important for multi-class detection tasks.
    nms_boxes = prepare_nms_boxes(boxes, classes)

    output_signature = (
        tf.TensorSpec(
            shape=(yolo_NMS_MAX_DETECTIONS,),
            dtype=tf.int32,
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    selected_indices, num_detections = tf.map_fn(nms_single_image, (nms_boxes, scores), fn_output_signature=output_signature)

    # Gather original coordinates, without the offsets used for NMS.
    selected_boxes = tf.gather(boxes, selected_indices, batch_dims=1)
    selected_scores = tf.gather(scores, selected_indices, batch_dims=1)
    selected_classes = tf.gather(classes, selected_indices, batch_dims=1)

    # Each image may have fewer detections than the padded output size.
    valid_positions = tf.sequence_mask(num_detections, maxlen=yolo_NMS_MAX_DETECTIONS)

    selected_boxes = tf.where(valid_positions[..., None], selected_boxes, tf.zeros_like(selected_boxes))
    selected_scores = tf.where(valid_positions, selected_scores, tf.zeros_like(selected_scores))
    selected_classes = tf.where(valid_positions, selected_classes, tf.zeros_like(selected_classes))

    return {
        "boxes": selected_boxes,
        "confidence": selected_scores,
        "classes": selected_classes,
        "num_detections": num_detections,
    }


def tensor_to_numpy(tensor):
    """Convert one output tensor to a NumPy array."""
    return tensor.numpy()


