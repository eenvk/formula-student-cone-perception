//Novkovic

#include "segmentation_evaluator.h"

#include "dataset_utils.h"
#include "evaluation_common.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

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