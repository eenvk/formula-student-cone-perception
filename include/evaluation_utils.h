// Author: Renzi

#ifndef EVALUATION_UTILS_H
#define EVALUATION_UTILS_H

#include "box.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

struct SegmentationReport {
    std::map<std::string, double> iou_per_class;
    double mean_iou;
};

struct MatchingResult {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_gt;
    std::vector<int> unmatched_pred;
};

struct ClassMetrics {
    double precision;
    double recall;
    double f1;
};

struct ClassificationReport {
    std::map<std::string, double> precision_per_class;
    std::map<std::string, double> recall_per_class;
    std::map<std::string, double> f1_per_class;
    double macro_f1;
};

struct DetectionReport {
    std::map<std::string, double> ap_per_class;
    double mean_average_precision;
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

double box_iou(const Box& box_a, const Box& box_b);
MatchingResult match_by_iou(const std::vector<Box>& gt_boxes, const std::vector<Box>& pred_boxes, double iou_threshold = 0.5);

class ClassificationEvaluator {
public:
    explicit ClassificationEvaluator(double iou_threshold = 0.5);
    ClassificationEvaluator(const std::vector<int>& class_ids, double iou_threshold = 0.5);

    void reset();
    void update(const std::vector<Box>& gt_boxes, const std::vector<Box>& pred_boxes);
    std::map<int, ClassMetrics> per_class_f1() const;
    double macro_f1() const;
    ClassificationReport report() const;

private:
    std::vector<int> class_ids_;
    double iou_threshold_;
    std::map<int, long long> true_positive_;
    std::map<int, long long> false_positive_;
    std::map<int, long long> false_negative_;
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

struct EvaluationReport {
    SegmentationReport segmentation;
    ClassificationReport classification;
    DetectionReport detection;
};

class PipelineEvaluator {
public:
    void update(const cv::Mat& gt_mask, const cv::Mat& pred_mask, const std::vector<Box>& gt_boxes, const std::vector<Box>& pred_boxes);
    EvaluationReport report() const;

private:
    SegmentationEvaluator segmentation_evaluator_;
    ClassificationEvaluator classification_evaluator_;
    DetectionEvaluator detection_evaluator_;
};

#endif
