//Granati

#ifndef DETECTION_EVALUATOR_H
#define DETECTION_EVALUATOR_H

#include "box.h"

#include <map>
#include <string>
#include <vector>

struct DetectionReport {
    std::map<std::string, double> ap_per_class;
    double mean_average_precision;
};

class DetectionEvaluator {
public:
    DetectionEvaluator();
    explicit DetectionEvaluator(const std::vector<int>& class_ids);

    void reset();
    void update(const std::vector<Box>& gt_boxes, const std::vector<Box>& pred_boxes);
    std::map<int, double> per_class_ap() const;
    double mean_average_precision() const;
    DetectionReport report() const;

private:
    double average_precision_for_class(int class_id, double iou_threshold) const;

    std::vector<int> class_ids_;
    std::vector<std::vector<Box>> gt_by_image_;
    std::vector<std::vector<Box>> pred_by_image_;
};

#endif