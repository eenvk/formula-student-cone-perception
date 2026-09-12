# Renzi

"""Inference timing and FPS utilities."""

import csv
import time


def measure_execution(function, *args):
    """Execute a function and return its result together with the elapsed time."""

    start_time = time.perf_counter()

    result = function(*args)

    elapsed_seconds = time.perf_counter() - start_time

    return result, elapsed_seconds


def compute_fps(num_images, total_seconds):
    """Compute the average number of processed frames per second."""

    if total_seconds <= 0.0:
        return 0.0

    return num_images / total_seconds


def save_timing(timing_path, num_images, detection_seconds, segmentation_seconds):
    """Save detection time, segmentation time, total time, and FPS."""

    total_seconds = detection_seconds + segmentation_seconds
    fps = compute_fps(num_images, total_seconds)

    timing_path.parent.mkdir(parents=True, exist_ok=True)

    with timing_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["num_images", "detection_seconds", "segmentation_seconds", "total_seconds", "fps"])
        writer.writerow([num_images, detection_seconds, segmentation_seconds, total_seconds, fps])