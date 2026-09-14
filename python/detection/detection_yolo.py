#Granati

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from detection.detection_config import (
    PROJECT_DIR,
    DETECTION_WEIGHTS_PATH,
    EPOCHS,
    EVAL_EVERY,
    SEED,
    VALIDATION_FREQ,
    IMAGE_SIZE
)

from detection.data_pipeline import build_train_val_datasets

from detection.model_utils import (
    InferenceValidation,
    configure_gpu,
    create_model,
)

def visualize_train_samples(train_ds, num_samples=30):
    class_names = {
        0: "yellow_cone",
        1: "blue_cone",
        2: "small_orange_cone",
        3: "big_orange_cone",
    }

    cols = 5
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = np.array(axes).reshape(-1)
    sample_index = 0

    for images, bounding_boxes in train_ds:
        boxes_batch = bounding_boxes["boxes"]
        classes_batch = bounding_boxes["classes"]

        batch_size = int(tf.shape(images)[0])

        for i in range(batch_size):
            if sample_index >= num_samples:
                break

            image = images[i].numpy()
            boxes = boxes_batch[i].numpy()
            classes = classes_batch[i].numpy()

            # Matplotlib visualization.
            if image.max() > 1.0:
                image = np.clip(image, 0, 255).astype(np.uint8)

            ax = axes[sample_index]
            ax.imshow(image)

            # Basic checks
            height, width = image.shape[:2]

            if height != IMAGE_SIZE[0] or width != IMAGE_SIZE[1]:
                print(
                    f"[WARNING sample {sample_index}] "
                    f"Image shape: {image.shape}"
                )

            if len(boxes) != len(classes):
                print(
                    f"[ERROR sample {sample_index}] "
                    f"{len(boxes)} boxes but {len(classes)} classes"
                )

            # Draw ground-truth boxes
            for box, class_id in zip(boxes, classes):
                x1, y1, x2, y2 = box

                if (x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1):
                    print(f"[WARNING sample {sample_index}] " "Invalid bbox: {box}")

                rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
                ax.add_patch(rect)
                class_id = int(class_id)

                ax.text(x1, y1, class_names.get(class_id, str(class_id)), fontsize=8, bbox={"alpha": 0.6})

            ax.set_title(
                f"Sample {sample_index + 1} | "
                f"{len(boxes)} boxes"
            )

            ax.axis("off")

            sample_index += 1

        if sample_index >= num_samples:
            break

    for i in range(sample_index, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    output_path = PROJECT_DIR / "training_samples.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nSaved {sample_index} training samples to:")
    print(output_path)


def main():
    tf.keras.utils.set_random_seed(SEED)
    configure_gpu()

    # DATASET
    train_ds, val_loss_ds, val_inference_ds, val_inference_metadata, inference_y_true, full_val_inference_ds, full_val_inference_metadata, full_inference_y_true = build_train_val_datasets()
    visualize_train_samples(train_ds, num_samples=30)

    print("Dataset is built")

    # MODEL
    model = create_model()
    print("Build model with", model.num_classes, "classes")

    # TRAINING
    callbacks = [
        InferenceValidation(
            val_inference_ds,
            DETECTION_WEIGHTS_PATH,
            val_inference_metadata,
            inference_y_true,
            eval_every=EVAL_EVERY,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_box_loss",
            mode="min",
            patience=6,
            restore_best_weights=False
        ),

        keras.callbacks.BackupAndRestore(backup_dir=str(PROJECT_DIR / "training_backup")),
        keras.callbacks.TerminateOnNaN(), #stop training if NaN values are found
    ]

    history = model.fit(train_ds, validation_data=val_loss_ds,
                        epochs=EPOCHS, callbacks=callbacks, validation_freq=VALIDATION_FREQ)

    print("\n" + "=" * 80)
    print("FINAL VALIDATION ON THE COMPLETE VALIDATION SET")
    print("=" * 80)

    print("\nLoading best detection weights...")
    model.load_weights(DETECTION_WEIGHTS_PATH)

    final_validation = InferenceValidation(
        full_val_inference_ds,
        DETECTION_WEIGHTS_PATH,
        full_val_inference_metadata,
        full_inference_y_true,
    )

    final_validation.set_model(model)
    detection_report, classification_report = final_validation.evaluate()

    print("\nFinal detection report:")
    for metric, value in detection_report.items():
        print(f"{metric}: {value}")

    print("\nFinal classification report:")
    for metric, value in classification_report.items():
        print(f"{metric}: {value}")

    return history


if __name__ == "__main__":
    main()