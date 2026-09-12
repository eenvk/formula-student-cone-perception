//Renzi

#ifndef EVALUATION_COMMON_H
#define EVALUATION_COMMON_H

#include "box.h"

#include <utility>
#include <vector>

struct MatchingResult {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_gt;
    std::vector<int> unmatched_pred;
};

double box_iou(const Box& box_a, const Box& box_b);
MatchingResult match_by_iou(const std::vector<Box>& gt_boxes, const std::vector<Box>& pred_boxes, double iou_threshold = 0.5);
double average_values(const std::vector<double>& values);
std::vector<int> default_cone_class_ids();
bool contains_class_id(const std::vector<int>& class_ids, int class_id);

#endif