#Novkovic

"""Segmentation inference and mask export for the C++ evaluation pipeline."""

import cv2

from dataset.dataset_utils import PROJECT_ROOT, load_image
from segmentation.predict_segmentation import predict_segmentation
from segmentation.segmentation_model import build_unet


MODEL_DIR = PROJECT_ROOT / "models"
SEGMENTATION_WEIGHTS_PATH = MODEL_DIR / "unet_best.weights.h5"


def load_segmentation_model():
    """Build the U-Net model and load the trained weights."""

    if not SEGMENTATION_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Segmentation weights not found: {SEGMENTATION_WEIGHTS_PATH}")

    model = build_unet()
    model.load_weights(str(SEGMENTATION_WEIGHTS_PATH))

    return model


def load_segmentation_image(image_path):
    """Load one image for segmentation inference."""

    return load_image(image_path)


def run_segmentation_inference(model, image_bgr, boxes):
    """Run U-Net segmentation using the detected bounding boxes."""

    return predict_segmentation(image_bgr, boxes, model)


def save_segmentation_prediction(image_path, predicted_mask, mask_dir):
    """Save one predicted semantic mask as a lossless PNG image."""

    mask_dir.mkdir(parents=True, exist_ok=True)

    mask_path = mask_dir / f"{image_path.stem}.png"

    if not cv2.imwrite(str(mask_path), predicted_mask):
        raise RuntimeError(f"Could not save mask: {mask_path}")