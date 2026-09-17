#Granati

import tensorflow as tf

from keras_cv.src.models.object_detection.yolo_v8 import yolo_v8_detector
from detection.detection_config import IMAGE_SIZE, yolo_PRE_NMS_TOP_K

IMAGE_HEIGHT = IMAGE_SIZE[0]
IMAGE_WIDTH = IMAGE_SIZE[1]


def create_yolo_inference(model):
    """ Create Yolo inference function"""
    implementation_yolo = yolo_v8_detector

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32)
    ])
    def graph_predict(images):
        raw_predictions = model(images, training=False)

        boxes = decode_boxes(raw_predictions["boxes"], images, implementation_yolo)
        boxes, class_scores = select_top_k(boxes, raw_predictions["classes"])

        boxes = implementation_yolo.bounding_box.convert_format(
            boxes, source="xyxy", target=model.bounding_box_format, images=images
        )

        predictions = model.prediction_decoder(boxes, class_scores)

        return predictions

    def infer(inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        predictions = graph_predict(inputs)

        return tf.nest.map_structure(tensor_to_numpy, predictions)

    return infer


def decode_boxes(raw_boxes, images, implementation):
    """
    Decode the raw box predictions from the yolo model into actual bounding boxes.
    
    Returns:
        A tensor of shape (batch_size, num_boxes, 4) containing the decoded bounding boxes in the format (x1, y1, x2, y2).
    """
    # now we use the yolo implementation to decode the raw boxes into actual bounding boxes
    distances = implementation.decode_regression_to_boxes(raw_boxes)

    # get_anchors returns the anchors and strides for the given image shape;
    # anchors are the predefined bounding boxes used by yolo, and strides are the scaling factors for the feature maps
    anchors, strides = implementation.get_anchors(image_shape=images.shape[1:])
    # the strides are expanded to match the shape of the distances tensor for broadcasting during multiplication
    strides = implementation.ops.expand_dims(strides, axis=-1)

    # from the implementation we know: decodes distance predictions into xyxy boxes
    # the resulting xyxy predictions must be scaled by the stride
    boxes = implementation.dist2bbox(distances, anchors)

    return boxes * strides #scale


def select_top_k(boxes, class_scores):
    """
    Keep at most PRE_NMS_TOP_K candidates for each image based on the highest class confidence associated 
    with each bounding box

    Returns:
        boxes: Tensor with shape (batch_size, top_k, 4)
        class_scores: Tensor with shape (batch_size, top_k, num_classes)
    """
    scores = tf.reduce_max(class_scores, axis=-1)

    num_candidates = tf.shape(scores)[1]
    top_k = tf.minimum(num_candidates, yolo_PRE_NMS_TOP_K)

    _, selected_indices = tf.math.top_k(scores, k=top_k, sorted=False)

    selected_boxes = tf.gather(boxes, selected_indices, batch_dims=1)
    selected_class_scores = tf.gather(class_scores, selected_indices, batch_dims=1)

    return selected_boxes, selected_class_scores


def tensor_to_numpy(tensor):
    return tensor.numpy()