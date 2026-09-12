//Granati

#ifndef CLASSIFICATION_EVALUATOR_H
#define CLASSIFICATION_EVALUATOR_H

#include "box.h"

#include <map>
#include <string>
#include <vector>

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

#endif