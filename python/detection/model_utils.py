#Granati

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import keras_cv

from detection.detection_config import GLOBAL_CLIPNORM, LEARNING_RATE, NUM_CLASSES
from detection.inference import predict_inference_dataset
from evaluation.evaluation_utils import  DetectionEvaluator, ClassificationEvaluator
from dataset.dataset_utils import ground_truth_to_box
from detection.decoder import create_yolo_inference


def configure_gpu():
    """
    Configures GPU memory growth and displays framework versions.
    This function enables dynamic GPU memory allocation to prevent TensorFlow
    from reserving all GPU memory at startup
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
    Loads a pre-trained YOLOv8-XS backbone,
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

    backbone = (keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xs_backbone_coco"))

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
        Initializes the callback with validation data, save path, and evaluation frequency.
        Args:
            data: Validation dataset for inference.
            save_path: Path to save the best model weights.
            metadata: Metadata for inference (e.g., image size, preprocessing).
            y_true: Ground truth bounding boxes and class labels for evaluation.
            eval_every: Frequency (in epochs) to run inference and evaluation.
        """
        super().__init__()

        self.data = data
        self.save_path = Path(save_path)
        self.eval_every = eval_every
        self.inference_metadata = metadata
        self.y_true = []
        self.best_map = -1.0

        for boxes, classes in zip(y_true["boxes"], y_true["classes"]):
            image_gt = []

            for box, class_id in zip(boxes, classes):
                image_gt.append(ground_truth_to_box(box, class_id,))

            self.y_true.append(image_gt)

    def set_model(self, model):
        """ Sets the model for inference and creates a YOLO inference function"""
        super().set_model(model)
        self.infer = create_yolo_inference(model)

    def evaluate(self):
        """
        Runs inference on the validation dataset and evaluates predictions against ground truth.
        """
        print("\nInference on validation set...")
        y_pred = predict_inference_dataset(self.infer, self.data, self.inference_metadata)

        # Check that the number of predictions matches the number of ground truth images
        if len(self.y_true) != len(y_pred):
            raise ValueError(
                f"Ground truth/prediction mismatch: "
                f"{len(self.y_true)} GT images, "
                f"{len(y_pred)} predicted images."
            )

        # Evaluate predictions using detection and classification metrics
        detection_evaluator = DetectionEvaluator()
        classification_evaluator = ClassificationEvaluator(iou_threshold=0.5)

        for gt_boxes, pred_boxes in zip(self.y_true, y_pred):
            detection_evaluator.update(gt_boxes, pred_boxes)
            classification_evaluator.update(gt_boxes, pred_boxes)

        detection_report = detection_evaluator.report()
        classification_report = classification_evaluator.report()

        current_map = detection_report["mAP@0.5:0.95"]
        current_f1 = classification_report["macro_f1"]

        print(f"\nInference mAP@0.5:0.95: {current_map:.4f}")
        print(f"Inference macro F1: {current_f1:.4f}")

        return detection_report, classification_report

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback function called at the end of each epoch during training.
        Runs inference and evaluation every `eval_every` epochs, and saves the model if it achieves
        a new best mAP score. (detection mean average precision)
        """
        if (epoch + 1) % self.eval_every != 0:
            return

        detection_report, classification_report = self.evaluate()

        current_map = detection_report["mAP@0.5:0.95"]
        current_f1 = classification_report["macro_f1"]

        if logs is not None:
            logs["mAP@0.5:0.95"] = current_map
            logs["macro_f1"] = current_f1

        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save_weights(str(self.save_path))

            print(f"New best model saved to: {self.save_path}")