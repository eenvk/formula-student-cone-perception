#Renzi

"""Evaluate the execution time of the complete inference pipeline."""

from dataset.dataset_utils import load_image
from inference_pipeline import run_pipeline_on_frame, warmup_pipeline


def evaluate_time(image_paths, yolo_infer, unet_infer):
    """Measure detection, segmentation and total pipeline execution time."""
    if not image_paths:
        raise ValueError("At least one image is required for timing evaluation.")

    warmup_image = load_image(image_paths[0])
    warmup_pipeline(warmup_image, yolo_infer, unet_infer)

    detection_seconds = 0.0
    segmentation_seconds = 0.0
    total_seconds = 0.0

    for image_index, image_path in enumerate(image_paths):
        image_bgr = load_image(image_path)

        predicted_boxes, _, frame_timings, num_views = run_pipeline_on_frame(image_bgr, yolo_infer, unet_infer)

        detection_ms = frame_timings["detection_preprocess"] + frame_timings["yolo_inference"] + frame_timings["detection_postprocess"]

        segmentation_ms = frame_timings["segmentation_preprocess"] + frame_timings["unet_inference"] + frame_timings["segmentation_postprocess"]

        detection_seconds += detection_ms / 1000.0
        segmentation_seconds += segmentation_ms / 1000.0
        total_seconds += frame_timings["pipeline_total"] / 1000.0

        print(f"[{image_index + 1}/{len(image_paths)}] {image_path.name} | views: {num_views} | boxes: {len(predicted_boxes)} | pipeline: {frame_timings['pipeline_total']:.2f} ms")

    fps = len(image_paths) / total_seconds if total_seconds > 0.0 else 0.0

    return {
        "num_images": len(image_paths),
        "detection_seconds": detection_seconds,
        "segmentation_seconds": segmentation_seconds,
        "total_seconds": total_seconds,
        "fps": fps,
    }


def print_time_report(report):
    """Print the pipeline timing results."""
    print()
    print("Inference timing")
    print("----------------")
    print(f"Images: {report['num_images']}")
    print(f"Detection time: {report['detection_seconds']:.6f} s")
    print(f"Segmentation time: {report['segmentation_seconds']:.6f} s")
    print(f"Total pipeline time: {report['total_seconds']:.6f} s")
    print(f"FPS: {report['fps']:.6f}")