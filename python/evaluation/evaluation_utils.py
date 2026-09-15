#renzi
"""Evaluation utilities for segmentation, classification, and detection."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from dataset.dataset_utils import (
    Box,
    CLASS_ID_TO_NAME,
    CONE_CLASS_IDS,
    IGNORE_ID,
    NUM_CLASSES,
)


class SegmentationEvaluator:
    """Accumulate a dataset-level pixel confusion matrix and compute IoU."""

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_id: int = IGNORE_ID) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        """Clear all accumulated samples."""

        self.confusion_matrix.fill(0)

    def update(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> None:
        """Add one ground-truth/prediction mask pair."""

        if gt_mask.ndim != 2 or pred_mask.ndim != 2:
            raise ValueError("gt_mask and pred_mask must be 2D arrays.")
        if gt_mask.shape != pred_mask.shape:
            raise ValueError(f"Shape mismatch: gt={gt_mask.shape}, pred={pred_mask.shape}")

        gt_flat = gt_mask.reshape(-1).astype(np.int64, copy=False)
        pred_flat = pred_mask.reshape(-1).astype(np.int64, copy=False)
        valid = gt_flat != self.ignore_id
        gt_valid = gt_flat[valid]
        pred_valid = pred_flat[valid]

        if np.any((gt_valid < 0) | (gt_valid >= self.num_classes)):
            raise ValueError("Ground-truth mask contains invalid class IDs.")
        if np.any((pred_valid < 0) | (pred_valid >= self.num_classes)):
            raise ValueError("Prediction mask contains invalid class IDs.")

        indices = gt_valid * self.num_classes + pred_valid
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self) -> dict[int, float]:
        """Return dataset-level IoU for every class."""

        result: dict[int, float] = {}

        for class_id in range(self.num_classes):
            true_positive = int(self.confusion_matrix[class_id, class_id])
            false_positive = int(self.confusion_matrix[:, class_id].sum()) - true_positive
            false_negative = int(self.confusion_matrix[class_id, :].sum()) - true_positive
            denominator = true_positive + false_positive + false_negative
            result[class_id] = true_positive / denominator if denominator > 0 else float("nan")

        return result

    def mean_iou(self, class_ids: Sequence[int] = CONE_CLASS_IDS) -> float:
        """Return the unweighted mean IoU over the requested classes."""

        iou_per_class = self.per_class_iou()
        values = [iou_per_class[class_id] for class_id in class_ids if class_id in iou_per_class and not np.isnan(iou_per_class[class_id])]
        return float(np.mean(values)) if values else float("nan")

    def report(self) -> dict[str, Any]:
        """Return named per-class IoU values and cone-class mIoU."""

        iou_per_class = self.per_class_iou()
        return {
            "iou_per_class": {CLASS_ID_TO_NAME.get(class_id, str(class_id)): value for class_id, value in iou_per_class.items()},
            "mIoU": self.mean_iou(),
        }



def box_iou(box_a: Box, box_b: Box) -> float:
    """Compute intersection over union for two half-open xyxy boxes."""

    x_min = max(box_a.x_min, box_b.x_min)
    y_min = max(box_a.y_min, box_b.y_min)
    x_max = min(box_a.x_max, box_b.x_max)
    y_max = min(box_a.y_max, box_b.y_max)
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    union = box_a.area + box_b.area - intersection
    return intersection / union if union > 0 else 0.0


def match_by_iou(gt_boxes: Sequence[Box], pred_boxes: Sequence[Box], iou_threshold: float = 0.5) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily match boxes one-to-one by descending IoU, ignoring class."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in the range [0, 1].")

    candidates = []

    for gt_index, gt_box in enumerate(gt_boxes):
        for pred_index, pred_box in enumerate(pred_boxes):
            iou = box_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int]] = []

    for _, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_index, pred_index))

    unmatched_gt = [index for index in range(len(gt_boxes)) if index not in matched_gt]
    unmatched_pred = [index for index in range(len(pred_boxes)) if index not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


class ClassificationEvaluator:
    """Compute macro F1 after geometry-only one-to-one box matching."""

    def __init__(self, class_ids: Sequence[int] = CONE_CLASS_IDS, iou_threshold: float = 0.5) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in the range [0, 1].")
        self.class_ids = tuple(class_ids)
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated counts."""

        self.true_positive = {class_id: 0 for class_id in self.class_ids}
        self.false_positive = {class_id: 0 for class_id in self.class_ids}
        self.false_negative = {class_id: 0 for class_id in self.class_ids}

    def update(self, gt_boxes: Sequence[Box], pred_boxes: Sequence[Box]) -> None:
        """Add one image worth of ground truth and predictions."""

        matches, unmatched_gt, unmatched_pred = match_by_iou(gt_boxes, pred_boxes, self.iou_threshold)

        for gt_index, pred_index in matches:
            gt_class = gt_boxes[gt_index].class_id
            pred_class = pred_boxes[pred_index].class_id
            if gt_class == pred_class:
                if gt_class in self.true_positive:
                    self.true_positive[gt_class] += 1
            else:
                if pred_class in self.false_positive:
                    self.false_positive[pred_class] += 1
                if gt_class in self.false_negative:
                    self.false_negative[gt_class] += 1

        for gt_index in unmatched_gt:
            gt_class = gt_boxes[gt_index].class_id
            if gt_class in self.false_negative:
                self.false_negative[gt_class] += 1

        for pred_index in unmatched_pred:
            pred_class = pred_boxes[pred_index].class_id
            if pred_class in self.false_positive:
                self.false_positive[pred_class] += 1

    def per_class_f1(self) -> dict[int, dict[str, float]]:
        """Return precision, recall and F1 for every cone class."""

        result: dict[int, dict[str, float]] = {}

        for class_id in self.class_ids:
            true_positive = self.true_positive[class_id]
            false_positive = self.false_positive[class_id]
            false_negative = self.false_negative[class_id]

            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive > 0
                else 0.0
            )

            recall = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative > 0
                else 0.0
            )

            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0.0
                else 0.0
            )

            result[class_id] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        return result


    def macro_f1(self) -> float:
        """Return the unweighted mean F1 across cone classes."""

        values = [
            metrics["f1"]
            for metrics in self.per_class_f1().values()
        ]

        return float(np.mean(values)) if values else float("nan")


    def report(self) -> dict[str, Any]:
        """Return named per-class precision, recall, F1 and macro F1."""

        metrics_per_class = self.per_class_f1()

        return {
            "precision_per_class": {
                CLASS_ID_TO_NAME[class_id]: metrics_per_class[class_id]["precision"]
                for class_id in self.class_ids
            },
            "recall_per_class": {
                CLASS_ID_TO_NAME[class_id]: metrics_per_class[class_id]["recall"]
                for class_id in self.class_ids
            },
            "f1_per_class": {
                CLASS_ID_TO_NAME[class_id]: metrics_per_class[class_id]["f1"]
                for class_id in self.class_ids
            },
            "macro_f1": self.macro_f1(),
        }




def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute COCO-style 101-point interpolated average precision."""

    if recall.ndim != 1 or precision.ndim != 1 or recall.shape != precision.shape:
        raise ValueError("recall and precision must be equally sized 1D arrays.")
    if recall.size == 0:
        return 0.0

    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    interpolated = np.zeros_like(recall_thresholds)

    for index, threshold in enumerate(recall_thresholds):
        valid = recall >= threshold
        interpolated[index] = np.max(precision_envelope[valid]) if np.any(valid) else 0.0

    return float(np.mean(interpolated))


class DetectionEvaluator:
    """Compute per-class AP and mAP over IoU thresholds 0.50 through 0.95."""

    IOU_THRESHOLDS = tuple(float(value) for value in np.round(np.arange(0.50, 1.00, 0.05), 2))

    def __init__(self, class_ids: Sequence[int] = CONE_CLASS_IDS) -> None:
        self.class_ids = tuple(class_ids)
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated images."""

        self._gt_by_image: list[list[Box]] = []
        self._pred_by_image: list[list[Box]] = []

    def update(self, gt_boxes: Sequence[Box], pred_boxes: Sequence[Box]) -> None:
        """Add one image worth of ground truth and scored predictions."""

        if any(box.score is not None for box in gt_boxes):
            raise ValueError("Ground-truth boxes must not have a confidence score.")
        if any(box.score is None for box in pred_boxes):
            raise ValueError("Every predicted box must have a confidence score.")

        self._gt_by_image.append(list(gt_boxes))
        self._pred_by_image.append(list(pred_boxes))

    def _average_precision_for_class(self, class_id: int, iou_threshold: float) -> float:
        gt_per_image = [[box for box in boxes if box.class_id == class_id] for boxes in self._gt_by_image]
        num_gt = sum(len(boxes) for boxes in gt_per_image)
        predictions: list[tuple[float, int, int, Box]] = []

        for image_index, boxes in enumerate(self._pred_by_image):
            for prediction_index, box in enumerate(boxes):
                if box.class_id == class_id:
                    predictions.append((float(box.score), image_index, prediction_index, box))

        if num_gt == 0:
            return float("nan") if not predictions else 0.0
        if not predictions:
            return 0.0

        predictions.sort(key=lambda item: (-item[0], item[1], item[2]))
        gt_used = [[False] * len(boxes) for boxes in gt_per_image]
        true_positive = np.zeros(len(predictions), dtype=np.float64)
        false_positive = np.zeros(len(predictions), dtype=np.float64)

        for rank, (_, image_index, _, pred_box) in enumerate(predictions):
            best_iou = -1.0
            best_gt_index = -1

            for gt_index, gt_box in enumerate(gt_per_image[image_index]):
                if gt_used[image_index][gt_index]:
                    continue
                iou = box_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            if best_gt_index >= 0 and best_iou >= iou_threshold:
                gt_used[image_index][best_gt_index] = True
                true_positive[rank] = 1.0
            else:
                false_positive[rank] = 1.0

        cumulative_tp = np.cumsum(true_positive)
        cumulative_fp = np.cumsum(false_positive)
        recall = cumulative_tp / float(num_gt)
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, np.finfo(np.float64).eps)
        return _average_precision(recall, precision)

    def per_class_ap(self) -> dict[int, float]:
        """Return AP@0.5:0.95 for every cone class."""

        result: dict[int, float] = {}

        for class_id in self.class_ids:
            values = [self._average_precision_for_class(class_id, threshold) for threshold in self.IOU_THRESHOLDS]
            finite_values = [value for value in values if not np.isnan(value)]
            result[class_id] = float(np.mean(finite_values)) if finite_values else float("nan")

        return result

    def mean_average_precision(self) -> float:
        """Return the unweighted mean AP across classes present in the ground truth."""

        values = [value for value in self.per_class_ap().values() if not np.isnan(value)]
        return float(np.mean(values)) if values else float("nan")

    def report(self) -> dict[str, Any]:
        """Return named per-class AP values and mAP@0.5:0.95."""

        ap_per_class = self.per_class_ap()
        return {
            "AP@0.5:0.95_per_class": {CLASS_ID_TO_NAME[class_id]: ap_per_class[class_id] for class_id in self.class_ids},
            "mAP@0.5:0.95": self.mean_average_precision(),
        }




# COCO-style area ranges (in px^2), used for AP_small / AP_medium / AP_large.
OBJECT_SIZE_RANGES: tuple[tuple[str, float, float], ...] = (
    ("small", 0.0, 32.0 ** 2),
    ("medium", 32.0 ** 2, 96.0 ** 2),
    ("large", 96.0 ** 2, float("inf")),
)

# Pixel-height bins for the fine-grained "how far can we see a cone" analysis.
HEIGHT_BINS_PX: tuple[tuple[float, float], ...] = (
    (0.0, 16.0),
    (16.0, 32.0),
    (32.0, 64.0),
    (64.0, float("inf")),
)


def _mean_ignoring_nan(values: Sequence[float]) -> float:
    """Return the mean of the finite values, or NaN if none are finite."""

    finite_values = [value for value in values if not np.isnan(value)]
    return float(np.mean(finite_values)) if finite_values else float("nan")


class ComprehensiveDetectionEvaluator(DetectionEvaluator):
    """Extend DetectionEvaluator with global metrics, a full per-class table,
    and breakdowns by object size and pixel height. Accumulation is
    inherited unchanged from DetectionEvaluator.update()."""

    def average_precision_at(self, iou_threshold: float) -> dict[int, float]:
        """Return per-class AP at a single IoU threshold (e.g. 0.50 or 0.75)."""

        return {class_id: self._average_precision_for_class(class_id, iou_threshold) for class_id in self.class_ids}

    def mean_ap_at(self, iou_threshold: float) -> float:
        """Return mAP across classes at a single IoU threshold."""

        return _mean_ignoring_nan(list(self.average_precision_at(iou_threshold).values()))

    def global_precision_recall_f1(self, iou_threshold: float = 0.5) -> dict[str, float]:
        """Return class-agnostic Precision/Recall/F1: 'is something detected at
        this location at all', ignoring whether the predicted class is right."""

        true_positive = false_positive = false_negative = 0

        for gt_boxes, pred_boxes in zip(self._gt_by_image, self._pred_by_image):
            matches, unmatched_gt, unmatched_pred = match_by_iou(gt_boxes, pred_boxes, iou_threshold)
            true_positive += len(matches)
            false_positive += len(unmatched_pred)
            false_negative += len(unmatched_gt)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    def global_summary(self, iou_threshold: float = 0.5) -> dict[str, float]:
        """Global mAP@50, mAP@75, mAP@50:95, plus Precision/Recall/F1."""

        precision_recall_f1 = self.global_precision_recall_f1(iou_threshold)
        return {
            "mAP@0.50": self.mean_ap_at(0.50),
            "mAP@0.75": self.mean_ap_at(0.75),
            "mAP@0.5:0.95": self.mean_average_precision(),
            "Precision": precision_recall_f1["precision"],
            "Recall": precision_recall_f1["recall"],
            "F1": precision_recall_f1["f1"],
        }

    def per_class_precision_recall_f1(self, iou_threshold: float = 0.5) -> dict[int, dict[str, float]]:
        """Per-class Precision/Recall/F1: gt and predictions are first filtered
        to the same class, then matched geometrically."""

        result: dict[int, dict[str, float]] = {}

        for class_id in self.class_ids:
            true_positive = false_positive = false_negative = 0

            for gt_boxes, pred_boxes in zip(self._gt_by_image, self._pred_by_image):
                gt_of_class = [box for box in gt_boxes if box.class_id == class_id]
                pred_of_class = [box for box in pred_boxes if box.class_id == class_id]
                matches, unmatched_gt, unmatched_pred = match_by_iou(gt_of_class, pred_of_class, iou_threshold)
                true_positive += len(matches)
                false_positive += len(unmatched_pred)
                false_negative += len(unmatched_gt)

            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
            result[class_id] = {"precision": precision, "recall": recall, "f1": f1}

        return result

    def per_class_summary(self, iou_threshold: float = 0.5) -> dict[str, dict[str, float]]:
        """Per-class table with AP@50, AP@75, AP@50:95, Precision, Recall and F1."""

        ap50 = self.average_precision_at(0.50)
        ap75 = self.average_precision_at(0.75)
        ap50_95 = self.per_class_ap()
        precision_recall_f1 = self.per_class_precision_recall_f1(iou_threshold)

        table: dict[str, dict[str, float]] = {}
        for class_id in self.class_ids:
            name = CLASS_ID_TO_NAME.get(class_id, str(class_id))
            table[name] = {
                "AP@0.50": ap50[class_id],
                "AP@0.75": ap75[class_id],
                "AP@0.5:0.95": ap50_95[class_id],
                "Precision": precision_recall_f1[class_id]["precision"],
                "Recall": precision_recall_f1[class_id]["recall"],
                "F1": precision_recall_f1[class_id]["f1"],
            }
        return table

    def _average_precision_for_class_and_size(
            self, class_id: int, iou_threshold: float, area_range: tuple[float, float]
    ) -> float:
        """AP restricted to ground-truth boxes whose area falls in area_range.
        Ground-truth boxes of the same class outside the range are neither
        counted as false negatives nor do predictions matched to them count
        as false positives (COCO-style 'ignore' region)."""

        lo, hi = area_range
        gt_in_range_per_image: list[list[Box]] = []
        gt_ignored_per_image: list[list[Box]] = []

        for boxes in self._gt_by_image:
            gt_in_range_per_image.append([box for box in boxes if box.class_id == class_id and lo <= box.area < hi])
            gt_ignored_per_image.append([box for box in boxes if box.class_id == class_id and not (lo <= box.area < hi)])

        num_gt = sum(len(boxes) for boxes in gt_in_range_per_image)

        predictions: list[tuple[float, int, int, Box]] = []
        for image_index, boxes in enumerate(self._pred_by_image):
            for prediction_index, box in enumerate(boxes):
                if box.class_id == class_id:
                    predictions.append((float(box.score), image_index, prediction_index, box))

        if num_gt == 0:
            return float("nan") if not predictions else 0.0
        if not predictions:
            return 0.0

        predictions.sort(key=lambda item: (-item[0], item[1], item[2]))
        gt_used = [[False] * len(boxes) for boxes in gt_in_range_per_image]
        ignored_used = [[False] * len(boxes) for boxes in gt_ignored_per_image]
        true_positive = np.zeros(len(predictions), dtype=np.float64)
        false_positive = np.zeros(len(predictions), dtype=np.float64)

        for rank, (_, image_index, _, pred_box) in enumerate(predictions):
            best_iou = -1.0
            best_index = -1
            best_is_ignored = False

            for gt_index, gt_box in enumerate(gt_in_range_per_image[image_index]):
                if gt_used[image_index][gt_index]:
                    continue
                iou = box_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou, best_index, best_is_ignored = iou, gt_index, False

            for gt_index, gt_box in enumerate(gt_ignored_per_image[image_index]):
                if ignored_used[image_index][gt_index]:
                    continue
                iou = box_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou, best_index, best_is_ignored = iou, gt_index, True

            if best_index >= 0 and best_iou >= iou_threshold:
                if best_is_ignored:
                    ignored_used[image_index][best_index] = True
                else:
                    gt_used[image_index][best_index] = True
                    true_positive[rank] = 1.0
            else:
                false_positive[rank] = 1.0

        cumulative_tp = np.cumsum(true_positive)
        cumulative_fp = np.cumsum(false_positive)
        recall = cumulative_tp / float(num_gt)
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, np.finfo(np.float64).eps)
        return _average_precision(recall, precision)

    def ap_by_size(self, size_ranges: Sequence[tuple[str, float, float]] = OBJECT_SIZE_RANGES) -> dict[str, float]:
        """AP_small / AP_medium / AP_large, i.e. AP@0.5:0.95 averaged
        over classes, restricted to ground-truth boxes of each area range."""

        result: dict[str, float] = {}
        for size_name, lo, hi in size_ranges:
            values = [
                self._average_precision_for_class_and_size(class_id, iou_threshold, (lo, hi))
                for class_id in self.class_ids
                for iou_threshold in self.IOU_THRESHOLDS
            ]
            result[size_name] = _mean_ignoring_nan(values)
        return result

    def recall_by_size(
            self, size_ranges: Sequence[tuple[str, float, float]] = OBJECT_SIZE_RANGES, iou_threshold: float = 0.5
    ) -> dict[str, float]:
        """Recall_small / Recall_medium / Recall_large at a fixed IoU
        threshold, pooled class-agnostically across all cone classes."""

        result: dict[str, float] = {}
        for size_name, lo, hi in size_ranges:
            true_positive = 0
            total_gt = 0
            for gt_boxes, pred_boxes in zip(self._gt_by_image, self._pred_by_image):
                gt_in_range = [box for box in gt_boxes if box.class_id in self.class_ids and lo <= box.area < hi]
                total_gt += len(gt_in_range)
                matches, _, _ = match_by_iou(gt_in_range, pred_boxes, iou_threshold)
                true_positive += len(matches)
            result[size_name] = true_positive / total_gt if total_gt > 0 else float("nan")
        return result

    def detection_rate_by_height_bin(
            self, height_bins: Sequence[tuple[float, float]] = HEIGHT_BINS_PX, iou_threshold: float = 0.5
    ) -> dict[str, float]:
        """Recall/detection-rate as a function of the real bounding-box
        height in pixels, pooled class-agnostically across all cone classes.
        Reveals thresholds like 'detection collapses below N px', which maps
        directly to cone distance."""

        result: dict[str, float] = {}
        for lo, hi in height_bins:
            label = f"{lo:g}-{hi:g}px" if hi != float("inf") else f">{lo:g}px"
            true_positive = 0
            total_gt = 0
            for gt_boxes, pred_boxes in zip(self._gt_by_image, self._pred_by_image):
                gt_in_bin = [
                    box for box in gt_boxes
                    if box.class_id in self.class_ids and lo <= (box.y_max - box.y_min) < hi
                ]
                total_gt += len(gt_in_bin)
                matches, _, _ = match_by_iou(gt_in_bin, pred_boxes, iou_threshold)
                true_positive += len(matches)
            result[label] = true_positive / total_gt if total_gt > 0 else float("nan")
        return result

    def full_report(self, iou_threshold: float = 0.5) -> dict[str, Any]:
        """Combine the global, per-class, size and height breakdowns above into a single report dict."""

        return {
            "global": self.global_summary(iou_threshold),
            "per_class": self.per_class_summary(iou_threshold),
            "by_object_area": {
                "AP@0.5:0.95": self.ap_by_size(),
                "Recall": self.recall_by_size(iou_threshold=iou_threshold),
            },
            "by_pixel_height": self.detection_rate_by_height_bin(iou_threshold=iou_threshold),
        }



class QualitativeErrorSampler:
    """Accumulate per-image ground truth/prediction pairs and select a
    representative sample (best detections, false positives, false negatives,
    correctly-detected small cones, likely occlusions) for visual inspection.

    `image` can be anything accepted by matplotlib's imshow (e.g. an HxWx3
    numpy array). Box overlays require x_min/y_min/x_max/y_max, class_id,
    area and (for predictions) score on each Box.
    """

    def __init__(self, iou_threshold: float = 0.5, small_area_threshold: float = 32.0 ** 2) -> None:
        self.iou_threshold = iou_threshold
        self.small_area_threshold = small_area_threshold
        self._records: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Clear all accumulated images."""

        self._records = []

    def add_image(self, image_id: Any, image: Any, gt_boxes: Sequence[Box], pred_boxes: Sequence[Box]) -> None:
        """Add one image worth of ground truth, predictions, and the raw image."""

        matches, unmatched_gt, unmatched_pred = match_by_iou(gt_boxes, pred_boxes, self.iou_threshold)
        correct_matches = [(g, p) for g, p in matches if gt_boxes[g].class_id == pred_boxes[p].class_id]
        misclassified_matches = [(g, p) for g, p in matches if gt_boxes[g].class_id != pred_boxes[p].class_id]

        num_small_correct = sum(1 for g, _ in correct_matches if gt_boxes[g].area < self.small_area_threshold)
        num_small_gt = sum(1 for box in gt_boxes if box.area < self.small_area_threshold)

        gt_boxes_list = list(gt_boxes)
        occlusion_pairs = 0
        for i in range(len(gt_boxes_list)):
            for j in range(i + 1, len(gt_boxes_list)):
                if box_iou(gt_boxes_list[i], gt_boxes_list[j]) > 0.1:
                    occlusion_pairs += 1

        self._records.append({
            "image_id": image_id,
            "image": image,
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes,
            "unmatched_gt": unmatched_gt,
            "unmatched_pred": unmatched_pred,
            "num_tp": len(correct_matches),
            "num_fp": len(unmatched_pred) + len(misclassified_matches),
            "num_fn": len(unmatched_gt) + len(misclassified_matches),
            "num_small_correct": num_small_correct,
            "num_small_gt": num_small_gt,
            "occlusion_pairs": occlusion_pairs,
        })

    def select(self, per_category: int = 3, extra_criteria: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
        """Return up to `per_category` records for each qualitative bucket.
        `extra_criteria` optionally maps a category name to a key already
        present in each record (e.g. a custom tag) to sort/select by, for
        dataset-specific tags such as truncated/knocked_over."""

        if not self._records:
            return {}

        def top_by(key: str, minimum: float = 0.0) -> list[dict[str, Any]]:
            candidates = [record for record in self._records if record[key] > minimum]
            return sorted(candidates, key=lambda record: record[key], reverse=True)[:per_category]

        selections: dict[str, list[dict[str, Any]]] = {
            "best_detections": sorted(self._records, key=lambda record: (record["num_fp"] + record["num_fn"], -record["num_tp"]))[:per_category],
            "small_objects_correct": top_by("num_small_correct"),
            "distant_small_cones": top_by("num_small_gt"),
            "false_positives": top_by("num_fp"),
            "false_negatives": top_by("num_fn"),
            "likely_occlusions": top_by("occlusion_pairs"),
        }

        if extra_criteria:
            for category_name, record_key in extra_criteria.items():
                selections[category_name] = top_by(record_key)

        return selections

    def render(self, record: dict[str, Any], output_path: str) -> None:
        """Draw ground-truth (green, solid) and predicted (red, dashed) boxes
        with class name + confidence, and save the overlay to output_path."""

        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(10, 8))
        axis.imshow(record["image"])

        for box in record["gt_boxes"]:
            axis.add_patch(patches.Rectangle(
                (box.x_min, box.y_min), box.x_max - box.x_min, box.y_max - box.y_min,
                linewidth=2, edgecolor="lime", facecolor="none",
                                        ))
            axis.text(
                box.x_min, max(box.y_min - 4, 0),
                f"GT:{CLASS_ID_TO_NAME.get(box.class_id, box.class_id)}",
                color="lime", fontsize=8, backgroundcolor="black",
            )

        for box in record["pred_boxes"]:
            axis.add_patch(patches.Rectangle(
                (box.x_min, box.y_min), box.x_max - box.x_min, box.y_max - box.y_min,
                linewidth=2, edgecolor="red", linestyle="--", facecolor="none",
                                        ))
            score_text = f" {box.score:.2f}" if box.score is not None else ""
            axis.text(
                box.x_min, box.y_max + 10,
                f"Pred:{CLASS_ID_TO_NAME.get(box.class_id, box.class_id)}{score_text}",
                color="red", fontsize=8, backgroundcolor="black",
                           )

        axis.set_title(str(record["image_id"]))
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def export(self, output_dir: str, per_category: int = 3, extra_criteria: dict[str, Any] | None = None) -> dict[str, list[str]]:
        """Select representative images per category and render them all to
        output_dir. Returns the saved file path per category."""

        import os

        os.makedirs(output_dir, exist_ok=True)
        selections = self.select(per_category, extra_criteria)

        exported: dict[str, list[str]] = {}
        for category, records in selections.items():
            exported[category] = []
            for index, record in enumerate(records):
                path = os.path.join(output_dir, f"{category}_{index}_{record['image_id']}.png")
                self.render(record, path)
                exported[category].append(path)
        return exported
