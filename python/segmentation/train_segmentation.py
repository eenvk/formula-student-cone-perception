#novkovic

"""Train the U-Net model for cone segmentation."""

import tensorflow as tf

from dataset.dataset_utils import PROJECT_ROOT
from segmentation.segmentation_config import LEARNING_RATE, NUM_EPOCHS
from segmentation.segmentation_dataset import create_train_validation_datasets
from segmentation.segmentation_model import build_unet


MODEL_DIR = PROJECT_ROOT / "models"
BEST_WEIGHTS_PATH = MODEL_DIR / "unet_bce_best.weights.h5"
BACKUP_DIR = MODEL_DIR / "training_backup"

'''
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Compute the Dice coefficient."""

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true, axis=(1, 2, 3)) + tf.reduce_sum(y_pred, axis=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return tf.reduce_mean(dice)


def dice_loss(y_true, y_pred):
    """Compute the Dice loss."""

    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """Combine binary cross-entropy and Dice loss."""

    binary_crossentropy = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    binary_crossentropy = tf.reduce_mean(binary_crossentropy)

    return 0.5 * binary_crossentropy + 0.5 * dice_loss(y_true, y_pred)

'''
def main():
    """Train the cone segmentation model."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, validation_dataset = create_train_validation_datasets()

    model = build_unet()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),loss=tf.keras.losses.BinaryCrossentropy(),)

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=str(BEST_WEIGHTS_PATH),monitor="val_loss",mode="min",save_best_only=True,save_weights_only=True,),
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(BACKUP_DIR),),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=6,restore_best_weights=True,),
    ]

    model.fit(train_dataset,validation_data=validation_dataset,epochs=NUM_EPOCHS,callbacks=callbacks,)


if __name__ == "__main__":
    main()