//Renzi

#include "evaluation_common.h"

#include "dataset_utils.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

    int box_area(const Box& box) {
        return std::max(0, box.x_max - box.x_min) * std::max(0, box.y_max - box.y_min);
    }

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