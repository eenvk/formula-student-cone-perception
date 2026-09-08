# ============================================================
# ENVIRONMENT VARIABLES
# IMPORTANT: they must be set BEFORE importing TensorFlow
# ============================================================

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ============================================================
# IMPORTS
# ============================================================

import tensorflow as tf
from tensorflow import keras

from detection.detection_config import (
    PROJECT_DIR,
    CHECKPOINT_PATH,
    EPOCHS,
    EVAL_EVERY,
    SEED,
    VALIDATION_FREQ,
    RESUME_TRAINING,
    RESUME_FROM_EPOCH
)

from detection.data_pipeline import build_train_val_datasets

from detection.model_utils import (
    InferenceValidation,
    configure_gpu,
    create_model,
    visualize_detections,
)


def main():
    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    tf.keras.utils.set_random_seed(SEED)

    # ========================================================
    # GPU CONFIGURATION
    # ========================================================

    configure_gpu()

    # DATASET
    train_ds, val_loss_ds, val_inference_ds = build_train_val_datasets()

    # MODEL
    model = create_model()
    print("Build model with", model.num_classes, "classes")

    # TRAINING
    if RESUME_TRAINING:
        print(
            f"\nResuming training from epoch "
            f"{RESUME_FROM_EPOCH}..."
        )

        model.load_weights(
            CHECKPOINT_PATH
        )

        print("Checkpoint loaded successfully.")
        
    callbacks = [
        InferenceValidation(
            val_inference_ds,
            CHECKPOINT_PATH,
            eval_every=EVAL_EVERY,
        ),

        keras.callbacks.BackupAndRestore(
            backup_dir=str(
                PROJECT_DIR / "training_backup"
                )
            ),

        keras.callbacks.TerminateOnNaN(), #stop training if NaN values are found
    ]

    history = model.fit(
        train_ds,
        validation_data=val_loss_ds,

        initial_epoch=(
            RESUME_FROM_EPOCH
            if RESUME_TRAINING
            else 0
        ),

        epochs=EPOCHS,
        callbacks=callbacks,
        validation_freq=VALIDATION_FREQ,
    )

    return history


if __name__ == "__main__":
    main()
