#Novkovic

"""Training process of the U-Net model."""

import tensorflow as tf

from dataset.dataset_utils import PROJECT_ROOT
from segmentation.segmentation_config import LEARNING_RATE, NUM_EPOCHS
from segmentation.segmentation_dataset import create_train_validation_datasets
from segmentation.segmentation_model import build_unet


MODEL_DIR = PROJECT_ROOT / "models"
BEST_WEIGHTS_PATH = MODEL_DIR / "unet_best.weights.h5"
BACKUP_DIR = MODEL_DIR / "training_backup"

def main():
    """Creates the model directory, loads the training and validation datasets,
    builds the U-Net, compiles it with the Adam optimizer and Binary Cross-Entropy loss,
    and starts the training process."""

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