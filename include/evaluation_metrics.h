//Renzi

#ifndef EVALUATION_METRICS_H
#define EVALUATION_METRICS_H

#include "box.h"

#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

// Summary of segmentation metrics.
struct SegmentationReport {
    std::map<std::string, double> iou_per_class;
    double mean_iou;
};

// Precision, recall and F1 values for a single class.
struct ClassMetrics {
    double precision;
    double recall;
    double f1;
};

// Summary of classification metrics.
struct ClassificationReport {
    std::map<std::string, double> precision_per_class;
    std::map<std::string, double> recall_per_class;
    std::map<std::string, double> f1_per_class;
    double macro_f1;
};

// Summary of object-detection metrics.
struct DetectionReport {
    std::map<std::string, double> ap_per_class;
    double mean_average_precision;
};

// Accumulates a confusion matrix and computes IoU-based metrics.
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

// Matches predicted and ground-truth boxes to compute classification metrics.
class ClassificationEvaluator {
public:
    explicit ClassificationEvaluator(double iou_threshold = 0.5);

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

// Stores detections across images and computes mAP@0.5:0.95.
class DetectionEvaluator {
public:
    DetectionEvaluator();

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