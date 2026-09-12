//Novkovic

#ifndef SEGMENTATION_EVALUATOR_H
#define SEGMENTATION_EVALUATOR_H

#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct SegmentationReport {
    std::map<std::string, double> iou_per_class;
    double mean_iou;
};

class SegmentationEvaluator {
public:
    explicit SegmentationEvaluator(int num_classes = 5, int ignore_id = 255);

    void reset();
    void update(const cv::Mat& gt_mask, const cv::Mat& pred_mask);
    std::map<int, double> per_class_iou() const;
    double mean_iou() const;
    double mean_iou(const std::vector<int>& class_ids) const;
    SegmentationReport report() const;
    const std::vector<std::vector<long long>>& confusion_matrix() const;

private:
    int num_classes_;
    int ignore_id_;
    std::vector<std::vector<long long>> confusion_matrix_;
};

#endif