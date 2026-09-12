//Granati

#include "classification_evaluator.h"

#include "dataset_utils.h"
#include "evaluation_common.h"

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

ClassificationEvaluator::ClassificationEvaluator(double iou_threshold) : ClassificationEvaluator(default_cone_class_ids(), iou_threshold) {
}

ClassificationEvaluator::ClassificationEvaluator(const std::vector<int>& class_ids, double iou_threshold) : class_ids_(class_ids), iou_threshold_(iou_threshold) {
    if (iou_threshold_ < 0.0 || iou_threshold_ > 1.0) {
        throw std::invalid_argument("iou_threshold must be in the range [0, 1].");
    }

    reset();
}

void ClassificationEvaluator::reset() {
    true_positive_.clear();
    false_positive_.clear();
    false_negative_.clear();

    for (int class_id : class_ids_) {
        true_positive_[class_id] = 0;
        false_positive_[class_id] = 0;
        false_negative_[class_id] = 0;
    }
}

void ClassificationEvaluator::update(const std::vector<Box>& gt_boxes, const std::vector<Box>& pred_boxes) {
    const MatchingResult matching = match_by_iou(gt_boxes, pred_boxes, iou_threshold_);

    for (const auto& [gt_index, pred_index] : matching.matches) {
        const int gt_class = gt_boxes[static_cast<std::size_t>(gt_index)].class_id;
        const int pred_class = pred_boxes[static_cast<std::size_t>(pred_index)].class_id;

        if (gt_class == pred_class) {
            if (contains_class_id(class_ids_, gt_class)) {
                ++true_positive_[gt_class];
            }
        } else {
            if (contains_class_id(class_ids_, pred_class)) {
                ++false_positive_[pred_class];
            }

            if (contains_class_id(class_ids_, gt_class)) {
                ++false_negative_[gt_class];
            }
        }
    }

    for (int gt_index : matching.unmatched_gt) {
        const int gt_class = gt_boxes[static_cast<std::size_t>(gt_index)].class_id;

        if (contains_class_id(class_ids_, gt_class)) {
            ++false_negative_[gt_class];
        }
    }

    for (int pred_index : matching.unmatched_pred) {
        const int pred_class = pred_boxes[static_cast<std::size_t>(pred_index)].class_id;

        if (contains_class_id(class_ids_, pred_class)) {
            ++false_positive_[pred_class];
        }
    }
}

std::map<int, ClassMetrics> ClassificationEvaluator::per_class_f1() const {
    std::map<int, ClassMetrics> result;

    for (int class_id : class_ids_) {
        const long long true_positive = true_positive_.at(class_id);
        const long long false_positive = false_positive_.at(class_id);
        const long long false_negative = false_negative_.at(class_id);

        const double precision = true_positive + false_positive > 0 ? static_cast<double>(true_positive) / static_cast<double>(true_positive + false_positive) : 0.0;
        const double recall = true_positive + false_negative > 0 ? static_cast<double>(true_positive) / static_cast<double>(true_positive + false_negative) : 0.0;
        const double f1 = precision + recall > 0.0 ? 2.0 * precision * recall / (precision + recall) : 0.0;

        result[class_id] = {precision, recall, f1};
    }

    return result;
}

double ClassificationEvaluator::macro_f1() const {
    const std::map<int, ClassMetrics> metrics_per_class = per_class_f1();
    std::vector<double> values;

    for (const auto& [class_id, metrics] : metrics_per_class) {
        static_cast<void>(class_id);
        values.push_back(metrics.f1);
    }

    return average_values(values);
}

ClassificationReport ClassificationEvaluator::report() const {
    const std::map<int, ClassMetrics> metrics_per_class = per_class_f1();
    ClassificationReport result;

    for (const auto& [class_id, metrics] : metrics_per_class) {
        const std::string class_name = class_id_to_name(class_id);
        result.precision_per_class[class_name] = metrics.precision;
        result.recall_per_class[class_name] = metrics.recall;
        result.f1_per_class[class_name] = metrics.f1;
    }

    result.macro_f1 = macro_f1();

    return result;
}