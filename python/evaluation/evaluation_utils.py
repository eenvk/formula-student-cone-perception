#renzi
"""Evaluation utilities for segmentation, classification, and detection"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from dataset.dataset_utils import (Box,CLASS_ID_TO_NAME,CONE_CLASS_IDS)

def box_iou(box_a: Box, box_b: Box) -> float:
    """Compute intersection over union for two half-open xyxy boxes"""

    x_min = max(box_a.x_min, box_b.x_min)
    y_min = max(box_a.y_min, box_b.y_min)
    x_max = min(box_a.x_max, box_b.x_max)
    y_max = min(box_a.y_max, box_b.y_max)
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    union = box_a.area + box_b.area - intersection
    return intersection / union if union > 0 else 0.0


def match_by_iou(gt_boxes: Sequence[Box], pred_boxes: Sequence[Box], iou_threshold: float = 0.5) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily match boxes one-to-one by descending IoU, ignoring class"""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in the range [0, 1]")

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
    """Compute macro F1 after geometry-only one-to-one box matching"""

    def __init__(self, class_ids: Sequence[int] = CONE_CLASS_IDS, iou_threshold: float = 0.5) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in the range [0, 1]")
        self.class_ids = tuple(class_ids)
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated counts"""

        self.true_positive = {class_id: 0 for class_id in self.class_ids}
        self.false_positive = {class_id: 0 for class_id in self.class_ids}
        self.false_negative = {class_id: 0 for class_id in self.class_ids}

    def update(self, gt_boxes: Sequence[Box], pred_boxes: Sequence[Box]) -> None:
        """Add one image worth of ground truth and predictions"""

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
        """Return precision, recall and F1 for every cone class"""

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
        """Return the unweighted mean F1 across cone classes"""

        values = [
            metrics["f1"]
            for metrics in self.per_class_f1().values()
        ]

        return float(np.mean(values)) if values else float("nan")


    def report(self) -> dict[str, Any]:
        """Return named per-class precision, recall, F1 and macro F1"""

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
    """Compute coco-style 101-point interpolated average precision"""

    if recall.ndim != 1 or precision.ndim != 1 or recall.shape != precision.shape:
        raise ValueError("recall and precision must be equally sized 1D arrays")
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
    """Compute per-class AP and mAP over IoU thresholds 0.50 through 0.95"""

    IOU_THRESHOLDS = tuple(float(value) for value in np.round(np.arange(0.50, 1.00, 0.05), 2))

    def __init__(self, class_ids: Sequence[int] = CONE_CLASS_IDS) -> None:
        self.class_ids = tuple(class_ids)
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated images"""

        self._gt_by_image: list[list[Box]] = []
        self._pred_by_image: list[list[Box]] = []

    def update(self, gt_boxes: Sequence[Box], pred_boxes: Sequence[Box]) -> None:
        """Add one image worth of ground truth and scored predictions"""

        if any(box.score is not None for box in gt_boxes):
            raise ValueError("Ground-truth boxes must not have a confidence score")
        if any(box.score is None for box in pred_boxes):
            raise ValueError("Every predicted box must have a confidence score")

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
        """Return ap@0.5:0.95 for every cone class"""

        result: dict[int, float] = {}

        for class_id in self.class_ids:
            values = [self._average_precision_for_class(class_id, threshold) for threshold in self.IOU_THRESHOLDS]
            finite_values = [value for value in values if not np.isnan(value)]
            result[class_id] = float(np.mean(finite_values)) if finite_values else float("nan")

        return result

    def mean_average_precision(self) -> float:
        """Return the unweighted mean ap across classes present in the ground truth"""

        values = [value for value in self.per_class_ap().values() if not np.isnan(value)]
        return float(np.mean(values)) if values else float("nan")

    def report(self) -> dict[str, Any]:
        """Return named per-class ap values and map@0.5:0.95"""

        ap_per_class = self.per_class_ap()
        return {
            "AP@0.5:0.95_per_class": {CLASS_ID_TO_NAME[class_id]: ap_per_class[class_id] for class_id in self.class_ids},
            "mAP@0.5:0.95": self.mean_average_precision(),
        }
