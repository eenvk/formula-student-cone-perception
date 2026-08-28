# ============================================================
# YOLO MODEL, METRICS AND VISUALIZATION
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization

from detection_config import (
    CLASS_MAPPING,
    GLOBAL_CLIPNORM,
    LEARNING_RATE,
    NUM_CLASSES,
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
            "\nATTENTION: TensorFlow does not detect a GPU. "
            "Training will run on CPU and will be much slower.\n"
        )


def create_model():
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
    print("\nLoading YOLOv8 backbone...")

    backbone = (
        keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_s_backbone_coco"
        )
    )

    print("Backbone loaded.")

    model = keras_cv.models.YOLOV8Detector(
        num_classes=NUM_CLASSES,
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=1, #parameter to catch object at different dim. might be increased to 2/3 (more memory needed)
    )

    print("\nYOLOv8 detector created.")
    model.summary()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM, #a way to limit the increasing of the gradients (more stability)
    )

    model.compile(
        optimizer=optimizer,
        classification_loss="binary_crossentropy",
        box_loss="ciou",
    )

    return model


class EvaluateCOCOMetricsCallback(
    keras.callbacks.Callback
):
    """
    Keras callback for evaluating COCO metrics during training.
    
    This callback computes COCO metrics (mAP - mean Average Precision) on validation
    data at regular intervals during training. It tracks the best performing model
    and automatically saves weights when validation mAP improves.
    """

    def __init__(self, data, save_path, eval_every=1):
        """
        Initializes the COCO metrics callback.
        
        Args:
            data: Validation dataset to evaluate on (batched)
            save_path: Path where best model weights will be saved
            eval_every: Evaluate metrics every N epochs (default: 1)
        """
        super().__init__()

        self.data = data
        self.save_path = Path(save_path)
        self.eval_every = eval_every

        self.metrics = (
            keras_cv.metrics.BoxCOCOMetrics(
                bounding_box_format="xyxy",
                evaluate_freq=1e9, #disenabled the automatic evalutation (with a large number)
            )
        )

        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each training epoch.
        
        Computes COCO metrics on validation data if the current epoch matches
        the evaluation frequency. Saves model weights if mAP improves.
        
        Args:
            epoch: Current epoch number
            logs: Dictionary with training metrics (updated with COCO metrics)
        """
        if ((epoch + 1) % self.eval_every != 0):
            return

        self.metrics.reset_state()

        print("Validation...")
        for images, y_true in self.data:
            y_pred = self.model.predict(images)

            y_true = bounding_box.to_ragged(y_true)
            y_pred = bounding_box.to_ragged(y_pred)

            self.metrics.update_state(
                y_true,
                y_pred,
            )

        raw_metrics = self.metrics.result(force=True)

        metric_values = {
            name: float(value.numpy())
            if tf.is_tensor(value)
            else float(value)

            for name, value in raw_metrics.items()
        }

        if logs is not None:
            logs.update(metric_values)

        current_map = metric_values["MaP"]

        print(f"\nValidation mAP: " f"{current_map:.5f}")

        if current_map > self.best_map:
            self.best_map = current_map

            self.model.save_weights(
                str(self.save_path)
            )

            print(
                "New best model saved to: "
                f"{self.save_path}"
            )


def visualize_detections(model, dataset, output_path):
    """
    Generates visualization of model predictions on a batch of images.
    
    Takes one batch from the dataset, generates predictions, and creates
    a side-by-side gallery showing ground truth and predicted bounding boxes.
    Saves the visualization to disk.
    
    Args:
        model: Trained YOLOv8 detector model
        dataset: Dataset to visualize (takes first batch)
        output_path: Path where visualization image will be saved (PNG)
    """
    images, y_true = next(
        iter(dataset.take(1))
    )

    y_pred = model.predict(
        images,
        verbose=0,
    )

    y_pred = bounding_box.to_ragged(
        y_pred
    )

    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format="xyxy",
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=False,
        font_scale=0.7,
        class_mapping=CLASS_MAPPING,
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nPredictions saved to: "
        f"{output_path}"
    )
