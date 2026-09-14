"""
evaluate_segmentation_realtime.py

Evaluate and profile the COMPLETE detection -> segmentation pipeline
one ORIGINAL IMAGE / FRAME at a time.

Important:
- Frames are processed sequentially.
- Inside one frame, YOLO processes that frame's patch/full-image views together.
- The KerasCV prediction decoder/NMS is NOT used in the real-time pipeline.
- Candidate boxes are decoded directly.
- A confidence threshold is applied first, then only the strongest candidates are kept.
- A lightweight class-aware NMS is finally applied ONLY to those prefiltered candidates.
- YOLO views from different original images are NEVER mixed.
- Ground-truth generation, evaluation, and visualization are measured
  separately and are NOT included in the real-time pipeline latency.
- Disk image loading is also reported separately. The main pipeline latency
  starts once the frame is already available in memory, which better
  approximates an onboard camera pipeline.

The detection views are generated with the same create_combined_views()
function used by the existing inference dataset.
"""

import time
import random

import cv2
import numpy as np
import tensorflow as tf

from dataset.dataset_utils import (
    BACKGROUND_ID,
    PROJECT_ROOT,
    annotation_to_semantic_mask,
    create_overlay,
    get_segmentation_test_pairs,
    load_image,
)

from detection.data_pipeline import (
    build_inference_metadata,
    create_combined_views,
)

from detection.detection_config import (
    DETECTION_WEIGHTS_PATH,
    SEED,
)

from detection.inference import postprocess_inference_dataset
from detection.model_utils import configure_gpu, create_model
from evaluation.evaluation_utils import SegmentationEvaluator

from segmentation.segmentation_config import (
    MASK_THRESHOLD,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    INPUT_CHANNELS,
    RANDOM_SEED,
)

from segmentation.segmentation_dataset import (
    create_crop_box,
    letterbox_sample,
)

from segmentation.segmentation_model import build_unet


MODEL_DIR = PROJECT_ROOT / "models"
SEGMENTATION_WEIGHTS_PATH = MODEL_DIR / "unet_best.weights.h5"
VISUALIZATION_DIR = MODEL_DIR / "segmentation_test_realtime"

WARMUP_RUNS = 2
SAVE_VISUALIZATIONS = False
MAX_IMAGES = None

# Cover every actual batch shape; split larger workloads without dropping cones.
PIPELINE_UNET_MAX_BATCH = 32
PIPELINE_UNET_WARMUP_RUNS = 3

# Isolated diagnostic, run AFTER the pipeline benchmark (never included in FPS).
RUN_UNET_COMPARISON = False
UNET_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
UNET_WARMUP_RUNS = 3
UNET_REPEATS = 20
UNET_ATOL = 1e-5
UNET_RTOL = 1e-4

# YOLO inference without the expensive KerasCV prediction decoder.
# Apply threshold first, then top-k, then a lightweight NMS only on the reduced set.
YOLO_SCORE_THRESHOLD = 0.25
YOLO_PRE_NMS_TOP_K = 100
YOLO_NMS_IOU_THRESHOLD = 0.50
YOLO_NMS_MAX_DETECTIONS = 100
YOLO_WARMUP_RUNS = 3


def elapsed_ms(start_time):
    return (time.perf_counter() - start_time) * 1000.0


def create_yolo_threshold_nms_inference(model):
    """Create a reusable YOLO graph with threshold + top-k + lightweight NMS."""
    import importlib

    implementation = importlib.import_module(type(model).__module__)
    required = ("decode_regression_to_boxes", "get_anchors", "dist2bbox", "bounding_box", "ops")

    if any(not hasattr(implementation, name) for name in required):
        raise RuntimeError(
            "Installed YOLO implementation lacks the decoder helpers expected by this script."
        )

    @tf.function(input_signature=[tf.TensorSpec((None, 800, 800, 3), tf.float32)])
    def graph_predict(images):
        raw_outputs = model(images, training=False)

        distances = implementation.decode_regression_to_boxes(raw_outputs["boxes"])
        anchors, strides = implementation.get_anchors(image_shape=images.shape[1:])
        strides = implementation.ops.expand_dims(strides, axis=-1)
        boxes = implementation.dist2bbox(distances, anchors) * strides

        boxes_xyxy = implementation.bounding_box.convert_format(
            boxes,
            source="xyxy",
            target="xyxy",
            images=images,
        )

        class_scores = raw_outputs["classes"]
        confidence = tf.reduce_max(class_scores, axis=-1)
        classes = tf.argmax(class_scores, axis=-1, output_type=tf.int32)

        # 1) Threshold BEFORE the NMS.
        threshold_mask = confidence >= YOLO_SCORE_THRESHOLD
        filtered_scores = tf.where(
            threshold_mask,
            confidence,
            tf.fill(tf.shape(confidence), tf.constant(-1.0, dtype=confidence.dtype)),
        )

        # 2) Keep only the strongest candidates.
        num_candidates = tf.shape(filtered_scores)[1]
        k = tf.minimum(num_candidates, YOLO_PRE_NMS_TOP_K)

        top_scores, top_indices = tf.math.top_k(
            filtered_scores,
            k=k,
            sorted=True,
        )

        top_boxes = tf.gather(
            boxes_xyxy,
            top_indices,
            batch_dims=1,
        )

        top_classes = tf.gather(
            classes,
            top_indices,
            batch_dims=1,
        )

        # tf.image.non_max_suppression_padded expects y1,x1,y2,x2.
        top_boxes_yxyx = tf.stack(
            [
                top_boxes[..., 1],
                top_boxes[..., 0],
                top_boxes[..., 3],
                top_boxes[..., 2],
            ],
            axis=-1,
        )

        # Class-aware NMS: boxes from different classes are offset so that
        # they cannot suppress one another.
        max_coordinate = (
            tf.reduce_max(
                tf.abs(top_boxes_yxyx),
                axis=[1, 2],
                keepdims=True,
            )
            + 1.0
        )

        class_offsets = (
            tf.cast(
                top_classes,
                top_boxes_yxyx.dtype,
            )[..., None]
            * max_coordinate
        )

        nms_boxes = top_boxes_yxyx + class_offsets

        def nms_single(args):
            boxes_one, scores_one = args

            selected_indices, valid_count = tf.image.non_max_suppression_padded(
                boxes=boxes_one,
                scores=scores_one,
                max_output_size=YOLO_NMS_MAX_DETECTIONS,
                iou_threshold=YOLO_NMS_IOU_THRESHOLD,
                score_threshold=YOLO_SCORE_THRESHOLD,
                pad_to_max_output_size=True,
            )

            return selected_indices, valid_count

        selected_indices, num_detections = tf.map_fn(
            nms_single,
            (nms_boxes, top_scores),
            fn_output_signature=(
                tf.TensorSpec(
                    (YOLO_NMS_MAX_DETECTIONS,),
                    tf.int32,
                ),
                tf.TensorSpec(
                    (),
                    tf.int32,
                ),
            ),
        )

        selected_boxes = tf.gather(
            top_boxes,
            selected_indices,
            batch_dims=1,
        )

        selected_scores = tf.gather(
            top_scores,
            selected_indices,
            batch_dims=1,
        )

        selected_classes = tf.gather(
            top_classes,
            selected_indices,
            batch_dims=1,
        )

        valid_positions = tf.sequence_mask(
            num_detections,
            maxlen=YOLO_NMS_MAX_DETECTIONS,
        )

        selected_boxes = tf.where(
            valid_positions[..., None],
            selected_boxes,
            tf.zeros_like(selected_boxes),
        )

        selected_scores = tf.where(
            valid_positions,
            selected_scores,
            tf.zeros_like(selected_scores),
        )

        selected_classes = tf.where(
            valid_positions,
            selected_classes,
            tf.zeros_like(selected_classes),
        )

        selected_boxes = implementation.bounding_box.convert_format(
            selected_boxes,
            source="xyxy",
            target=model.bounding_box_format,
            images=images,
        )

        return {
            "boxes": selected_boxes,
            "confidence": selected_scores,
            "classes": selected_classes,
            "num_detections": num_detections,
        }

    def infer(inputs):
        tensor = tf.convert_to_tensor(
            inputs,
            dtype=tf.float32,
        )

        outputs = graph_predict(tensor)

        return tf.nest.map_structure(
            lambda x: x.numpy(),
            outputs,
        )

    return infer, graph_predict


def warmup_yolo(infer):
    """Warm the reusable YOLO graph outside measured frame times."""
    if YOLO_WARMUP_RUNS < 1:
        raise ValueError("YOLO_WARMUP_RUNS must be positive.")

    rng = np.random.default_rng(SEED)

    inputs = rng.random(
        (1, 800, 800, 3),
        dtype=np.float32,
    )

    start = time.perf_counter()

    print(
        f"\nYOLO threshold + top-k + NMS graph warm-up: {YOLO_WARMUP_RUNS} call(s)",
        flush=True,
    )

    for _ in range(YOLO_WARMUP_RUNS):
        infer(inputs)

    print(
        f"YOLO graph warm-up time: {time.perf_counter() - start:.2f} s "
        "(excluded from pipeline FPS)",
        flush=True,
    )


def build_single_frame_detection_input(image_bgr):
    """Build YOLO input views for one original frame as one host NumPy batch."""
    image_height, image_width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    views, _ = create_combined_views(image_rgb)

    image_shapes = tf.constant([[image_height, image_width]], dtype=tf.int32)
    metadata = build_inference_metadata(image_shapes)
    if len(views) != len(metadata):
        raise ValueError(
            f"Views/metadata mismatch: {len(views)} views, {len(metadata)} metadata elements."
        )

    inputs = np.ascontiguousarray(views, dtype=np.float32)
    return inputs, metadata, len(views)


def detect_single_frame(image_bgr, yolo_infer):
    """Run YOLO on one frame with threshold + top-k + lightweight NMS."""
    timings = {}

    start = time.perf_counter()
    inputs, metadata, num_views = build_single_frame_detection_input(image_bgr)
    timings["detection_preprocess"] = elapsed_ms(start)

    start = time.perf_counter()
    raw_predictions = yolo_infer(inputs)
    timings["yolo_inference"] = elapsed_ms(start)

    start = time.perf_counter()
    predictions = postprocess_inference_dataset(raw_predictions, metadata)
    timings["detection_postprocess"] = elapsed_ms(start)

    if len(predictions) != 1:
        raise ValueError(f"Expected exactly one prediction group, got {len(predictions)}.")

    return predictions[0], timings, num_views

def prepare_segmentation_inputs(image_bgr, boxes):
    """Shared crop preparation for the pipeline and isolated U-Net diagnostic."""
    image_height, image_width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB,)

    model_inputs = []
    roi_metadata = []

    rng = random.Random(RANDOM_SEED)

    for box in boxes:
        x_min, y_min, x_max, y_max = create_crop_box(box,image_height,image_width,rng,training=False,)

        image_crop = image_rgb[y_min:y_max,x_min:x_max]
        image_crop, _, letterbox_metadata = letterbox_sample(image_crop)
        image_crop = image_crop.astype(np.float32) / 255.0

        model_inputs.append(image_crop)
        roi_metadata.append((box,x_min,y_min,x_max,y_max,letterbox_metadata))

    model_inputs = np.stack(model_inputs)

    return model_inputs, roi_metadata


def create_unet_inference(model):
    """Create one reusable graph and a host-output wrapper for the pipeline."""
    if PIPELINE_UNET_MAX_BATCH < 1:
        raise ValueError("PIPELINE_UNET_MAX_BATCH must be positive.")

    @tf.function(input_signature=[tf.TensorSpec(
        (None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), tf.float32
    )])
    def graph_predict(inputs):
        return model(inputs, training=False)

    def infer(inputs):
        if len(inputs) == 0:
            return np.empty((0, INPUT_HEIGHT, INPUT_WIDTH, 1), dtype=np.float32)
        outputs = [
            graph_predict(inputs[start:start + PIPELINE_UNET_MAX_BATCH]).numpy()
            for start in range(0, len(inputs), PIPELINE_UNET_MAX_BATCH)
        ]
        return outputs[0] if len(outputs) == 1 else np.concatenate(outputs, axis=0)

    return infer, graph_predict


def warmup_unet(infer):
    """Warm all batch sizes used by the wrapper, outside measured frame times."""
    if PIPELINE_UNET_WARMUP_RUNS < 1:
        raise ValueError("PIPELINE_UNET_WARMUP_RUNS must be positive.")
    rng = np.random.default_rng(SEED)
    inputs = rng.random((PIPELINE_UNET_MAX_BATCH, INPUT_HEIGHT, INPUT_WIDTH,
                         INPUT_CHANNELS), dtype=np.float32)
    start = time.perf_counter()
    print(f"\nU-Net graph warm-up: batches 1..{PIPELINE_UNET_MAX_BATCH}, "
          f"{PIPELINE_UNET_WARMUP_RUNS} calls each", flush=True)
    for size in range(1, PIPELINE_UNET_MAX_BATCH + 1):
        for _ in range(PIPELINE_UNET_WARMUP_RUNS):
            infer(inputs[:size])
        print(f"  Warmed U-Net batch {size}/{PIPELINE_UNET_MAX_BATCH}", flush=True)
    print(f"U-Net graph warm-up time: {time.perf_counter() - start:.2f} s "
          "(excluded from pipeline FPS)", flush=True)


def predict_segmentation_profiled(image_bgr,boxes,unet_infer):
    """
    Same logic as predict_segmentation(), split into preprocessing,
    U-Net inference and postprocessing timings.
    """
    timings = {"segmentation_preprocess": 0.0, "unet_inference": 0.0, "segmentation_postprocess": 0.0}
    image_height, image_width = image_bgr.shape[:2]

    semantic_mask = np.full((image_height, image_width), BACKGROUND_ID, dtype=np.uint8)

    if not boxes:
        return semantic_mask, timings

    start = time.perf_counter()

    model_inputs, roi_metadata = prepare_segmentation_inputs(image_bgr, boxes)

    timings["segmentation_preprocess"] = elapsed_ms(start)

    start = time.perf_counter()

    predictions = unet_infer(model_inputs)
    timings["unet_inference"] = elapsed_ms(start)

    start = time.perf_counter()

    score_mask = np.full((image_height, image_width),-1.0,dtype=np.float32)

    for prediction, metadata in zip(predictions,roi_metadata):
        box, x_min, y_min, x_max, y_max, letterbox_metadata = metadata
        top_padding, left_padding, resized_height, resized_width = letterbox_metadata

        probability_map = prediction[:, :, 0]

        probability_map = probability_map[
            top_padding:top_padding + resized_height,
            left_padding:left_padding + resized_width,
        ]

        roi_width = x_max - x_min
        roi_height = y_max - y_min

        probability_map = cv2.resize(probability_map,(roi_width, roi_height),interpolation=cv2.INTER_LINEAR)
        binary_mask = probability_map >= MASK_THRESHOLD

        detection_score = (
            box.score
            if box.score is not None
            else 1.0
        )

        semantic_region = semantic_mask[y_min:y_max,x_min:x_max,]
        score_region = score_mask[y_min:y_max,x_min:x_max]

        update_pixels = (binary_mask& (detection_score > score_region))

        semantic_region[update_pixels] = box.class_id
        score_region[update_pixels] = detection_score

    timings["segmentation_postprocess"] = elapsed_ms(start)

    return semantic_mask, timings


def run_pipeline_on_frame(image_bgr, yolo_infer, unet_infer):
    """
    Run the production-relevant pipeline on ONE frame.
    """
    pipeline_start = time.perf_counter()

    predicted_boxes, detection_timings, num_views = detect_single_frame(image_bgr, yolo_infer)
    predicted_mask, segmentation_timings = predict_segmentation_profiled(image_bgr,predicted_boxes,unet_infer,)

    pipeline_total = elapsed_ms(pipeline_start)

    timings = {**detection_timings,**segmentation_timings,"pipeline_total": pipeline_total}

    return predicted_boxes, predicted_mask, timings, num_views


def print_statistics(name, values):
    values = np.asarray(values, dtype=np.float64)

    print(
        f"{name:<30}"
        f"{np.mean(values):>12.2f}"
        f"{np.median(values):>12.2f}"
        f"{np.percentile(values, 95):>12.2f}"
        f"{np.std(values):>12.2f}"
    )


def benchmark_unet_modes(segmentation_model, source_image_path, source_boxes):
    """Compare host NumPy inputs -> host NumPy outputs on identical crop batches.

    First use is NOT cold start: the model has already run in the pipeline, and
    modes share TensorFlow/cuDNN caches. Only subsequent repeated calls are used
    for steady-state statistics. Repeated crops are a compute workload, not extra
    independent accuracy samples.
    """
    if not source_boxes:
        print("\nU-Net comparison skipped: no detected cones.")
        return
    if UNET_WARMUP_RUNS < 1 or UNET_REPEATS < 1:
        raise ValueError("U-Net warm-up and repeats must be positive.")
    if not UNET_BATCH_SIZES or any(n < 1 for n in UNET_BATCH_SIZES):
        raise ValueError("U-Net batch sizes must be positive.")

    crops, _ = prepare_segmentation_inputs(load_image(source_image_path), source_boxes)
    input_shape = tuple(crops.shape[1:])

    @tf.function(input_signature=[tf.TensorSpec((None, *input_shape), tf.float32)])
    def graph_predict(inputs):
        return segmentation_model(inputs, training=False)

    # All paths start with the same host array and finish with a host array.
    # Explicit batch_size prevents predict() from splitting larger test batches.
    modes = {
        "predict": lambda x: segmentation_model.predict(x, batch_size=len(x), verbose=0),
        "direct": lambda x: segmentation_model(x, training=False).numpy(),
        "tf.function": lambda x: graph_predict(x).numpy(),
    }
    names = list(modes)
    order_rng = random.Random(SEED)
    print("\n" + "=" * 82)
    print("ISOLATED U-NET COMPARISON (excluded from pipeline FPS)")
    print(f"Source: {source_image_path.name}; real crops={len(crops)}; shape={input_shape}")
    print(f"Warm-up/mode/batch={UNET_WARMUP_RUNS}; timed repeats={UNET_REPEATS}")
    print("Timing: host NumPy input -> host NumPy output; preprocessing excluded.")
    print("First-use times share caches and are NOT independent cold-start measurements.")
    print("Agreement is against direct mode on these crops, not dataset accuracy.", flush=True)

    for batch_size in UNET_BATCH_SIZES:
        indices = np.arange(batch_size) % len(crops)
        batch = np.ascontiguousarray(crops[indices], dtype=np.float32)
        print(f"\nBatch={batch_size}; unique crops={min(batch_size, len(crops))}; "
              f"repeated crops={max(0, batch_size - len(crops))}", flush=True)
        first_use = {}
        first_outputs = {}
        order = names.copy()
        order_rng.shuffle(order)
        print("First-use order: " + ", ".join(order), flush=True)
        for name in order:
            start = time.perf_counter()
            first_outputs[name] = modes[name](batch)
            first_use[name] = elapsed_ms(start)
            print(f"  {name}: first use {first_use[name]:.2f} ms", flush=True)

        for _ in range(UNET_WARMUP_RUNS):
            order_rng.shuffle(order)
            for name in order:
                modes[name](batch)

        samples = {name: [] for name in names}
        last_outputs = {}
        for repeat in range(UNET_REPEATS):
            order_rng.shuffle(order)
            for name in order:
                start = time.perf_counter()
                last_outputs[name] = modes[name](batch)
                samples[name].append(elapsed_ms(start))
            if (repeat + 1) % 5 == 0 or repeat + 1 == UNET_REPEATS:
                print(f"  Repeats: {repeat + 1}/{UNET_REPEATS}", flush=True)

        print(f"{'Mode':<30}{'Mean [ms]':>12}{'Median':>12}{'P95':>12}{'Std':>12}")
        for name in names:
            print_statistics(name, samples[name])
        for name in names:
            errors = []
            flips = []
            close = True
            for outputs in (first_outputs, last_outputs):
                reference = outputs["direct"]
                candidate = outputs[name]
                if candidate.shape != reference.shape:
                    raise ValueError(f"Output shape mismatch for {name}: {candidate.shape}")
                errors.append(float(np.max(np.abs(candidate - reference))))
                flips.append(float(np.mean((candidate >= MASK_THRESHOLD) !=
                                           (reference >= MASK_THRESHOLD))))
                close = close and bool(np.allclose(candidate, reference,
                                                   atol=UNET_ATOL, rtol=UNET_RTOL))
            print(f"  {name}: allclose={close} (atol={UNET_ATOL}, rtol={UNET_RTOL}), "
                  f"max_abs_diff={max(errors):.8g}, mask_disagreement={100 * max(flips):.6f}%")
        print(f"tf.function trace count: {graph_predict.experimental_get_tracing_count()}", flush=True)



def main():
    tf.keras.utils.set_random_seed(SEED)
    configure_gpu()

    if not SEGMENTATION_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Segmentation weights not found: {SEGMENTATION_WEIGHTS_PATH}"
        )

    if not DETECTION_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Detection weights not found: {DETECTION_WEIGHTS_PATH}"
        )

    if SAVE_VISUALIZATIONS:
        VISUALIZATION_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("\nLoading YOLO detector...")

    detector_model = create_model()
    detector_model.load_weights(str(DETECTION_WEIGHTS_PATH))
    yolo_infer, yolo_graph = create_yolo_threshold_nms_inference(detector_model)

    print("Loading U-Net segmentation model...")

    segmentation_model = build_unet()
    segmentation_model.load_weights(
        SEGMENTATION_WEIGHTS_PATH
    )

    unet_infer, unet_graph = create_unet_inference(segmentation_model)

    evaluator = SegmentationEvaluator()

    pairs = get_segmentation_test_pairs()

    if MAX_IMAGES is not None:
        pairs = pairs[:MAX_IMAGES]

    if not pairs:
        raise RuntimeError("No test images found.")

    print(f"\nTest images: {len(pairs)}")
    warmup_yolo(yolo_infer)
    print(f"Pipeline YOLO threshold/NMS trace count after warm-up: {yolo_graph.experimental_get_tracing_count()}")
    warmup_unet(unet_infer)
    print(f"Pipeline U-Net trace count after warm-up: {unet_graph.experimental_get_tracing_count()}")

    print(
        f"\nRunning {WARMUP_RUNS} "
        f"complete pipeline warm-up run(s)..."
    )

    warmup_image = load_image(
        pairs[0][0]
    )

    for warmup_index in range(WARMUP_RUNS):
        _, _, _, num_views = run_pipeline_on_frame(
            warmup_image,
            yolo_infer,
            unet_infer,
        )

        print(
            f"Warm-up {warmup_index + 1}/{WARMUP_RUNS} "
            f"completed ({num_views} YOLO views)."
        )

    print("Warm-up completed.\n")

    stage_names = [
        "detection_preprocess",
        "yolo_inference",
        "detection_postprocess",
        "segmentation_preprocess",
        "unet_inference",
        "segmentation_postprocess",
        "pipeline_total",
    ]

    timings = {
        name: []
        for name in stage_names
    }

    auxiliary_timings = {
        "image_load_io": [],
        "ground_truth_and_evaluation": [],
        "visualization_io": [],
    }

    view_counts = []
    detection_counts = []
    unet_source_path = None
    unet_source_boxes = []

    benchmark_start = time.perf_counter()

    for image_index, (
        image_path,
        annotation_path,
    ) in enumerate(pairs):

        # Disk I/O: deliberately excluded from pipeline_total.
        start = time.perf_counter()

        image_bgr = load_image(image_path)

        auxiliary_timings["image_load_io"].append(
            elapsed_ms(start)
        )

        (
            predicted_boxes,
            predicted_mask,
            frame_timings,
            num_views,
        ) = run_pipeline_on_frame(
            image_bgr,
            yolo_infer,
            unet_infer,
        )

        for name in stage_names:
            timings[name].append(
                frame_timings[name]
            )

        if RUN_UNET_COMPARISON and len(predicted_boxes) > len(unet_source_boxes):
            unet_source_path = image_path
            unet_source_boxes = list(predicted_boxes)

        view_counts.append(num_views)
        detection_counts.append(
            len(predicted_boxes)
        )

        # Ground truth + metric evaluation.
        # Excluded from real-time pipeline latency.
        start = time.perf_counter()

        image_height, image_width = image_bgr.shape[:2]

        gt_mask = annotation_to_semantic_mask(
            annotation_path,
            image_height,
            image_width,
        )

        evaluator.update(
            gt_mask,
            predicted_mask,
        )

        auxiliary_timings[
            "ground_truth_and_evaluation"
        ].append(
            elapsed_ms(start)
        )

        visualization_time = 0.0

        # Debug output only: excluded from real-time pipeline latency.
        if SAVE_VISUALIZATIONS:
            start = time.perf_counter()

            gt_overlay = create_overlay(
                image_bgr,
                gt_mask,
            )

            predicted_overlay = create_overlay(
                image_bgr,
                predicted_mask,
            )

            comparison = np.hstack(
                (
                    image_bgr,
                    gt_overlay,
                    predicted_overlay,
                )
            )

            output_path = (
                VISUALIZATION_DIR
                / f"{image_index:03d}_{image_path.stem}.jpg"
            )

            cv2.imwrite(
                str(output_path),
                comparison,
            )

            visualization_time = elapsed_ms(start)

        auxiliary_timings[
            "visualization_io"
        ].append(
            visualization_time
        )

        print(
            f"[{image_index + 1:02d}/{len(pairs):02d}] "
            f"{image_path.name} | "
            f"views={num_views:2d} | "
            f"detections={len(predicted_boxes):2d} | "
            f"pipeline={frame_timings['pipeline_total']:.2f} ms"
        )

    benchmark_wall_time = (
        time.perf_counter()
        - benchmark_start
    )

    print()
    print("=" * 82)
    print("REAL-TIME FRAME-BY-FRAME PIPELINE TIMING")
    print("=" * 82)

    print(f"{'Stage':<30}"f"{'Mean [ms]':>12}"f"{'Median':>12}"f"{'P95':>12}"f"{'Std':>12}")

    print("-" * 82)

    for name in stage_names:
        print_statistics(
            name,
            timings[name],
        )

    print("-" * 82)

    mean_pipeline_ms = float(
        np.mean(
            timings["pipeline_total"]
        )
    )

    median_pipeline_ms = float(
        np.median(
            timings["pipeline_total"]
        )
    )

    p95_pipeline_ms = float(
        np.percentile(
            timings["pipeline_total"],
            95,
        )
    )

    mean_fps = 1000.0 / mean_pipeline_ms
    median_fps = 1000.0 / median_pipeline_ms

    print(
        f"Mean pipeline latency:   "
        f"{mean_pipeline_ms:.2f} ms/frame"
    )

    print(
        f"Median pipeline latency: "
        f"{median_pipeline_ms:.2f} ms/frame"
    )

    print(
        f"P95 pipeline latency:    "
        f"{p95_pipeline_ms:.2f} ms/frame"
    )

    print(
        f"FPS from mean latency:   "
        f"{mean_fps:.2f}"
    )

    print(
        f"FPS from median latency: "
        f"{median_fps:.2f}"
    )

    print()
    print("PIPELINE TIME DISTRIBUTION")
    print("-" * 50)

    for name in stage_names[:-1]:
        mean_stage = float(
            np.mean(
                timings[name]
            )
        )

        percentage = (
            mean_stage
            / mean_pipeline_ms
            * 100.0
        )

        print(
            f"{name:<30}: "
            f"{mean_stage:8.2f} ms "
            f"({percentage:6.2f}%)"
        )

    print()
    print("AUXILIARY / EVALUATION TIMES")
    print("(excluded from real-time pipeline FPS)")
    print("-" * 50)

    for name, values in auxiliary_timings.items():
        print(
            f"{name:<30}: "
            f"{np.mean(values):8.2f} ms mean"
        )

    print()
    print("WORKLOAD")
    print("-" * 50)

    print(
        f"YOLO views/frame: "
        f"mean={np.mean(view_counts):.2f}, "
        f"min={np.min(view_counts)}, "
        f"max={np.max(view_counts)}"
    )

    print(
        f"Detected cones/frame: "
        f"mean={np.mean(detection_counts):.2f}, "
        f"min={np.min(detection_counts)}, "
        f"max={np.max(detection_counts)}"
    )

    report = evaluator.report()

    print()
    print("=" * 82)
    print("SEGMENTATION ACCURACY")
    print("=" * 82)

    for class_name, iou in (
        report["iou_per_class"].items()
    ):
        print(
            f"{class_name:<25}: "
            f"IoU = {iou:.4f}"
        )

    print("-" * 82)

    print(
        f"Cone mIoU: "
        f"{report['mIoU']:.4f}"
    )

    if SAVE_VISUALIZATIONS:
        print(
            f"Visualizations saved in: "
            f"{VISUALIZATION_DIR}"
        )

    print()
    print(
        f"Total benchmark wall time "
        f"(including evaluation/debug): "
        f"{benchmark_wall_time:.2f} s"
    )


    print(f"Pipeline YOLO threshold/NMS trace count after benchmark: {yolo_graph.experimental_get_tracing_count()}")
    print(f"Pipeline U-Net trace count after benchmark: {unet_graph.experimental_get_tracing_count()}")

    if RUN_UNET_COMPARISON:
        diagnostic_start = time.perf_counter()
        benchmark_unet_modes(segmentation_model, unet_source_path, unet_source_boxes)
        print(f"U-Net diagnostic wall time: {time.perf_counter() - diagnostic_start:.2f} s "
              "(additional to benchmark wall time above)")


if __name__ == "__main__":
    main()
