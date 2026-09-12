//Granati

#include "detection_evaluator.h"

#include "dataset_utils.h"
#include "evaluation_common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

    double average_precision(const std::vector<double>& recall, const std::vector<double>& precision) {
        if (recall.size() != precision.size()) {
            throw std::invalid_argument("recall and precision must have the same size.");
        }

        if (recall.empty()) {
            return 0.0;
        }

        std::vector<double> precision_envelope = precision;

        for (int index = static_cast<int>(precision_envelope.size()) - 2; index >= 0; --index) {
            precision_envelope[static_cast<std::size_t>(index)] = std::max(precision_envelope[static_cast<std::size_t>(index)], precision_envelope[static_cast<std::size_t>(index + 1)]);
        }

        double interpolated_sum = 0.0;

        for (int threshold_index = 0; threshold_index <= 100; ++threshold_index) {
            const double threshold = static_cast<double>(threshold_index) / 100.0;
            double interpolated_precision = 0.0;

            for (std::size_t index = 0; index < recall.size(); ++index) {
                if (recall[index] >= threshold) {
                    interpolated_precision = std::max(interpolated_precision, precision_envelope[index]);
                }
            }

            interpolated_sum += interpolated_precision;
        }

        return interpolated_sum / 101.0;
    }

}

DetectionEvaluator::DetectionEvaluator() : DetectionEvaluator(default_cone_class_ids()) {
}

DetectionEvaluator::DetectionEvaluator(const std::vector<int>& class_ids) : class_ids_(class_ids) {
    reset();
}

void DetectionEvaluator::reset() {
    gt_by_image_.clear();
    pred_by_image_.clear();
}

void DetectionEvaluator::update(const std::vector<Box>& gt_boxes, const std::vector<Box>& pred_boxes) {
    for (const Box& box : gt_boxes) {
        if (box.score.has_value()) {
            throw std::invalid_argument("Ground-truth boxes must not have a confidence score.");
        }
    }

    for (const Box& box : pred_boxes) {
        if (!box.score.has_value()) {
            throw std::invalid_argument("Every predicted box must have a confidence score.");
        }
    }

    gt_by_image_.push_back(gt_boxes);
    pred_by_image_.push_back(pred_boxes);
}

double DetectionEvaluator::average_precision_for_class(int class_id, double iou_threshold) const {
    std::vector<std::vector<Box>> gt_per_image;
    gt_per_image.reserve(gt_by_image_.size());

    std::size_t num_gt = 0;

    for (const std::vector<Box>& boxes : gt_by_image_) {
        std::vector<Box> filtered_boxes;

        for (const Box& box : boxes) {
            if (box.class_id == class_id) {
                filtered_boxes.push_back(box);
            }
        }

        num_gt += filtered_boxes.size();
        gt_per_image.push_back(filtered_boxes);
    }

    struct PredictionRecord {
        double score;
        int image_index;
        int prediction_index;
        Box box;
    };

    std::vector<PredictionRecord> predictions;

    for (std::size_t image_index = 0; image_index < pred_by_image_.size(); ++image_index) {
        const std::vector<Box>& boxes = pred_by_image_[image_index];

        for (std::size_t prediction_index = 0; prediction_index < boxes.size(); ++prediction_index) {
            const Box& box = boxes[prediction_index];

            if (box.class_id == class_id) {
                predictions.push_back({box.score.value(), static_cast<int>(image_index), static_cast<int>(prediction_index), box});
            }
        }
    }

    if (num_gt == 0) {
        return predictions.empty() ? std::numeric_limits<double>::quiet_NaN() : 0.0;
    }

    if (predictions.empty()) {
        return 0.0;
    }

    const auto compare_predictions = [](const PredictionRecord& first, const PredictionRecord& second) {
        if (first.score != second.score) {
            return first.score > second.score;
        }

        if (first.image_index != second.image_index) {
            return first.image_index < second.image_index;
        }

        return first.prediction_index < second.prediction_index;
    };

    std::sort(predictions.begin(), predictions.end(), compare_predictions);

    std::vector<std::vector<bool>> gt_used;
    gt_used.reserve(gt_per_image.size());

    for (const std::vector<Box>& boxes : gt_per_image) {
        gt_used.emplace_back(boxes.size(), false);
    }

    std::vector<double> true_positive(predictions.size(), 0.0);
    std::vector<double> false_positive(predictions.size(), 0.0);

    for (std::size_t rank = 0; rank < predictions.size(); ++rank) {
        const PredictionRecord& prediction = predictions[rank];
        const std::vector<Box>& image_gt = gt_per_image[static_cast<std::size_t>(prediction.image_index)];

        double best_iou = -1.0;
        int best_gt_index = -1;

        for (std::size_t gt_index = 0; gt_index < image_gt.size(); ++gt_index) {
            if (gt_used[static_cast<std::size_t>(prediction.image_index)][gt_index]) {
                continue;
            }

            const double iou = box_iou(image_gt[gt_index], prediction.box);

            if (iou > best_iou) {
                best_iou = iou;
                best_gt_index = static_cast<int>(gt_index);
            }
        }

        if (best_gt_index >= 0 && best_iou >= iou_threshold) {
            gt_used[static_cast<std::size_t>(prediction.image_index)][static_cast<std::size_t>(best_gt_index)] = true;
            true_positive[rank] = 1.0;
        } else {
            false_positive[rank] = 1.0;
        }
    }

    std::vector<double> recall(predictions.size(), 0.0);
    std::vector<double> precision(predictions.size(), 0.0);

    double cumulative_tp = 0.0;
    double cumulative_fp = 0.0;

    for (std::size_t index = 0; index < predictions.size(); ++index) {
        cumulative_tp += true_positive[index];
        cumulative_fp += false_positive[index];

        recall[index] = cumulative_tp / static_cast<double>(num_gt);
        precision[index] = cumulative_tp / std::max(cumulative_tp + cumulative_fp, std::numeric_limits<double>::epsilon());
    }

    return average_precision(recall, precision);
}

std::map<int, double> DetectionEvaluator::per_class_ap() const {
    static const std::array<double, 10> iou_thresholds = {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95};

    std::map<int, double> result;

    for (int class_id : class_ids_) {
        std::vector<double> finite_values;

        for (double threshold : iou_thresholds) {
            const double value = average_precision_for_class(class_id, threshold);

            if (!std::isnan(value)) {
                finite_values.push_back(value);
            }
        }

        result[class_id] = average_values(finite_values);
    }

    return result;
}

double DetectionEvaluator::mean_average_precision() const {
    const std::map<int, double> ap_per_class = per_class_ap();
    std::vector<double> finite_values;

    for (const auto& [class_id, value] : ap_per_class) {
        static_cast<void>(class_id);

        if (!std::isnan(value)) {
            finite_values.push_back(value);
        }
    }

    return average_values(finite_values);
}

DetectionReport DetectionEvaluator::report() const {
    const std::map<int, double> ap_per_class = per_class_ap();
    DetectionReport result;

    for (const auto& [class_id, ap] : ap_per_class) {
        result.ap_per_class[class_id_to_name(class_id)] = ap;
    }

    result.mean_average_precision = mean_average_precision();

    return result;
}