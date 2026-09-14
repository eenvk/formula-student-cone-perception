#Granati

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization

from detection.detection_config import (
    CLASS_MAPPING,
    GLOBAL_CLIPNORM,
    LEARNING_RATE,
    NUM_CLASSES,
)

from detection.inference import (
    predict_inference_dataset
)

from evaluation.evaluation_utils import (
    DetectionEvaluator,
    ClassificationEvaluator,
)

from dataset.dataset_utils import (
    ground_truth_to_box
)


def configure_gpu():
    """
    Configures GPU memory growth and displays framework versions.

    This function enables dynamic GPU memory allocation to prevent TensorFlow
    from reserving all GPU memory at startup. Also displays versions of
    TensorFlow and KerasCV, and notifies if no GPU is detected.

    Returns:
        None (prints status information to console)
    """
    gpus = tf.config.list_physical_devices("GPU")

    print("TensorFlow:", tf.__version__)
    print("KerasCV:", keras_cv.__version__)
    print("GPU detected:", gpus)

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(
                    gpu,
                    True
                )
            except RuntimeError:
                pass
    else:
        print(
            "\nATTENTION: TensorFlow does not detect a GPU. \n"
        )


def create_model(num_classes=None):
    """
    Creates and compiles a YOLOv8 object detection model.

    Loads a pre-trained YOLOv8-S backbone (trained on COCO dataset),
    creates a detector head, and compiles the model with:
    - Adam optimizer with gradient clipping for stability
    - Binary crossentropy for classification loss
    - CIoU (Complete IoU) for bounding box loss

    Returns:
        keras.Model: Compiled YOLOv8 detector model ready for training
    """
    if num_classes is None:
        num_classes = NUM_CLASSES

    print("Number of classes", num_classes)
    print("\nLoading YOLOv8 backbone...")

    backbone = (
        keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_s_backbone_coco"
        )
    )

    print("Backbone loaded.")

    model = keras_cv.models.YOLOV8Detector(
        num_classes=num_classes,
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=1, #parameter to catch object at different dim. might be increased to 2/3 (more memory needed)
    )

    print("\nYOLOv8 detector created.")
    # model.summary()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM, #a way to limit the increasing of the gradients (more stability)
    )

    model.compile(
        optimizer=optimizer,
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        jit_compile=False
    )

    return model


class InferenceValidation(keras.callbacks.Callback):
    """
    Keras callback for inference the validation set.
    """

    def __init__(self, data, save_path, metadata, y_true, eval_every=10):
        """
        Initializes the callback.

        Args:
            data: Validation dataset to evaluate on (batched)
            save_path: Path where best model weights will be saved
            eval_every: Evaluate metrics every N epochs (default: 10)
        """
        super().__init__()

        self.data = data
        self.save_path = Path(save_path)
        self.eval_every = eval_every
        self.inference_metadata = metadata
        self.y_true = []
        self.best_map = -1.0

        for boxes, classes in zip(
                y_true["boxes"],
                y_true["classes"]
        ):
            image_gt = []

            for box, class_id in zip(boxes, classes):
                image_gt.append(ground_truth_to_box(box, class_id,))

            self.y_true.append(image_gt)

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each training epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary with training metrics (updated with COCO metrics)
        """
        if ((epoch + 1) % self.eval_every != 0):
            return

        print("\nInference on validation set...")
        y_pred = predict_inference_dataset(self.model, self.data, self.inference_metadata)

        if len(self.y_true) != len(y_pred):
            raise ValueError(
                f"Ground truth/prediction mismatch: "
                f"{len(self.y_true)} GT images, "
                f"{len(y_pred)} predicted images."
            )

        detection_evaluator = DetectionEvaluator()
        classification_evaluator = ClassificationEvaluator(iou_threshold=0.5)

        for gt_boxes, pred_boxes in zip(self.y_true, y_pred,):
            detection_evaluator.update(gt_boxes, pred_boxes,)
            classification_evaluator.update(gt_boxes, pred_boxes,)

        detection_report = detection_evaluator.report()
        classification_report = classification_evaluator.report()

        current_f1 = classification_report["macro_f1"]
        current_map = detection_report["mAP@0.5:0.95"]

        print(f"\nInference mAP@0.5:0.95: "f"{current_map:.4f}")
        print(f"\nInference f1: "f"{current_f1:.4f}")

        if logs is not None:
            logs["mAP@0.5:0.95"] = current_map
            logs["macro_f1"] = classification_report["macro_f1"]

        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save_weights(str(self.save_path))

            print("New best model saved to: " f"{self.save_path}")