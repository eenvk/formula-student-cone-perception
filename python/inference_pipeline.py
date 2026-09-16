#Renzi

"""Optimized yolo + unet inference pipeline on one frame"""

import time
from timing.timing_utils import elapsed_ms
from detection.inference import detect_frame
from segmentation.predict_segmentation import predict_segmentation


def run_pipeline_on_frame(image_bgr, yolo_infer, unet_infer):
    """Run detection and segmentation on one frame."""
    pipeline_start = time.perf_counter()

    predicted_boxes, detection_timings, num_views = detect_frame(image_bgr, yolo_infer)
    predicted_mask, segmentation_timings = predict_segmentation(image_bgr, predicted_boxes, unet_infer)

    timings = {
        **detection_timings,
        **segmentation_timings,
        "pipeline_total": elapsed_ms(pipeline_start),
    }

    return predicted_boxes, predicted_mask, timings, num_views


def warmup_pipeline(image_bgr, yolo_infer, unet_infer):
    """Run one complete frame before timing starts."""
    print("Running one pipeline warm-up frame...")

    _, _, _, num_views = run_pipeline_on_frame(image_bgr, yolo_infer, unet_infer)

    print(f"Warm-up completed with {num_views} YOLO views")
    print("Warm-up time is excluded from FPS")