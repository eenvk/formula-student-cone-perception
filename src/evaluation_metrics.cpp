//Renzi

#include "evaluation_metrics.h"

#include "dataset_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

    struct MatchingResult {
        std::vector<std::pair<int, int>> matches;
        std::vector<int> unmatched_gt;
        std::vector<int> unmatched_pred;
    };

    int box_area(const Box& box) {
        return std::max(0, box.x_max - box.x_min) * std::max(0, box.y_max - box.y_min);
    }

    double box_iou(const Box& box_a, const Box& box_b) {
        const int x_min = std::max(box_a.x_min, box_b.x_min);
        const int y_min = std::max(box_a.y_min, box_b.y_min);
        const int x_max = std::min(box_a.x_max, box_b.x_max);
        const int y_max = std::min(box_a.y_max, box_b.y_max);

        const int intersection = std::max(0, x_max - x_min) * std::max(0, y_max - y_min);
        const int union_area = box_area(box_a) + box_area(box_b) - intersection;

        if (union_area <= 0) {
            return 0.0;
        }

        return static_cast<double>(intersection) / static_cast<double>(union_area);
    }

    MatchingResult match_by_iou(const std::vector<Box>& gt_boxes, const std::vector<Box>& pred_boxes, double iou_threshold) {
        if (iou_threshold < 0.0 || iou_threshold > 1.0) {
            throw std::invalid_argument("iou_threshold must be in the range [0, 1].");
        }

        struct Candidate {
            double iou;
            int gt_index;
            int pred_index;
        };

        std::vector<Candidate> candidates;

        for (std::size_t gt_index = 0; gt_index < gt_boxes.size(); ++gt_index) {
            for (std::size_t pred_index = 0; pred_index < pred_boxes.size(); ++pred_index) {
                const double iou = box_iou(gt_boxes[gt_index], pred_boxes[pred_index]);

                if (iou >= iou_threshold) {
                    candidates.push_back({iou, static_cast<int>(gt_index), static_cast<int>(pred_index)});
                }
            }
        }

        const auto compare_candidates = [](const Candidate& first, const Candidate& second) {
            if (first.iou != second.iou) {
                return first.iou > second.iou;
            }

            if (first.gt_index != second.gt_index) {
                return first.gt_index < second.gt_index;
            }

            return first.pred_index < second.pred_index;
        };

        std::sort(candidates.begin(), candidates.end(), compare_candidates);

        std::vector<bool> matched_gt(gt_boxes.size(), false);
        std::vector<bool> matched_pred(pred_boxes.size(), false);

        MatchingResult result;

        for (const Candidate& candidate : candidates) {
            if (matched_gt[static_cast<std::size_t>(candidate.gt_index)] || matched_pred[static_cast<std::size_t>(candidate.pred_index)]) {
                continue;
            }

            matched_gt[static_cast<std::size_t>(candidate.gt_index)] = true;
            matched_pred[static_cast<std::size_t>(candidate.pred_index)] = true;

            result.matches.emplace_back(candidate.gt_index, candidate.pred_index);
        }

        for (std::size_t index = 0; index < matched_gt.size(); ++index) {
            if (!matched_gt[index]) {
                result.unmatched_gt.push_back(static_cast<int>(index));
            }
        }

        for (std::size_t index = 0; index < matched_pred.size(); ++index) {
            if (!matched_pred[index]) {
                result.unmatched_pred.push_back(static_cast<int>(index));
            }
        }

        return result;
    }

    double average_values(const std::vector<double>& values) {
        if (values.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double sum = std::accumulate(values.begin(), values.end(), 0.0);

        return sum / static_cast<double>(values.size());
    }

    std::vector<int> default_cone_class_ids() {
        return std::vector<int>(CONE_CLASS_IDS.begin(), CONE_CLASS_IDS.end());
    }

    bool contains_class_id(const std::vector<int>& class_ids, int class_id) {
        return std::find(class_ids.begin(), class_ids.end(), class_id) != class_ids.end();
    }

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

SegmentationEvaluator::SegmentationEvaluator(int num_classes, int ignore_id) : num_classes_(num_classes), ignore_id_(ignore_id) {
    if (num_classes_ <= 0) {
        throw std::invalid_argument("num_classes must be positive.");
    }

    reset();
}

void SegmentationEvaluator::reset() {
    confusion_matrix_.assign(static_cast<std::size_t>(num_classes_), std::vector<long long>(static_cast<std::size_t>(num_classes_), 0));
}

void SegmentationEvaluator::update(const cv::Mat& gt_mask, const cv::Mat& pred_mask) {
    if (gt_mask.empty() || pred_mask.empty() || gt_mask.dims != 2 || pred_mask.dims != 2 || gt_mask.channels() != 1 || pred_mask.channels() != 1) {
        throw std::invalid_argument("gt_mask and pred_mask must be non-empty 2D single-channel images.");
    }

    if (gt_mask.rows != pred_mask.rows || gt_mask.cols != pred_mask.cols) {
        throw std::invalid_argument("Ground-truth and prediction mask shapes must match.");
    }

    cv::Mat gt_int;
    cv::Mat pred_int;

    gt_mask.convertTo(gt_int, CV_32S);
    pred_mask.convertTo(pred_int, CV_32S);

    for (int row = 0; row < gt_int.rows; ++row) {
        for (int column = 0; column < gt_int.cols; ++column) {
            const int gt_class = gt_int.at<int>(row, column);
            const int pred_class = pred_int.at<int>(row, column);

            if (gt_class == ignore_id_) {
                continue;
            }

            if (gt_class < 0 || gt_class >= num_classes_) {
                throw std::invalid_argument("Ground-truth mask contains invalid class IDs.");
            }

            if (pred_class < 0 || pred_class >= num_classes_) {
                throw std::invalid_argument("Prediction mask contains invalid class IDs.");
            }

            ++confusion_matrix_[static_cast<std::size_t>(gt_class)][static_cast<std::size_t>(pred_class)];
        }
    }
}

std::map<int, double> SegmentationEvaluator::per_class_iou() const {
    std::map<int, double> result;

    for (int class_id = 0; class_id < num_classes_; ++class_id) {
        const long long true_positive = confusion_matrix_[static_cast<std::size_t>(class_id)][static_cast<std::size_t>(class_id)];

        long long predicted_as_class = 0;
        long long ground_truth_class = 0;

        for (int other_class = 0; other_class < num_classes_; ++other_class) {
            predicted_as_class += confusion_matrix_[static_cast<std::size_t>(other_class)][static_cast<std::size_t>(class_id)];
            ground_truth_class += confusion_matrix_[static_cast<std::size_t>(class_id)][static_cast<std::size_t>(other_class)];
        }

        const long long false_positive = predicted_as_class - true_positive;
        const long long false_negative = ground_truth_class - true_positive;
        const long long denominator = true_positive + false_positive + false_negative;

        if (denominator > 0) {
            result[class_id] = static_cast<double>(true_positive) / static_cast<double>(denominator);
        } else {
            result[class_id] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    return result;
}

double SegmentationEvaluator::mean_iou() const {
    return mean_iou(default_cone_class_ids());
}

double SegmentationEvaluator::mean_iou(const std::vector<int>& class_ids) const {
    const std::map<int, double> iou_per_class = per_class_iou();
    std::vector<double> values;

    for (int class_id : class_ids) {
        const auto iterator = iou_per_class.find(class_id);

        if (iterator != iou_per_class.end() && !std::isnan(iterator->second)) {
            values.push_back(iterator->second);
        }
    }

    return average_values(values);
}

SegmentationReport SegmentationEvaluator::report() const {
    const std::map<int, double> iou_per_class = per_class_iou();

    SegmentationReport result;

    for (const auto& [class_id, iou] : iou_per_class) {
        result.iou_per_class[class_id_to_name(class_id)] = iou;
    }

    result.mean_iou = mean_iou();

    return result;
}

const std::vector<std::vector<long long>>& SegmentationEvaluator::confusion_matrix() const {
    return confusion_matrix_;
}

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