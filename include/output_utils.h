// Author:

#ifndef OUTPUT_UTILS_H
#define OUTPUT_UTILS_H

#include "box.h"
#include "evaluation_utils.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct ImageMetrics {
    std::string image_name;
    double mean_iou;
    double macro_f1;
    double mean_average_precision;
    std::size_t ground_truth_boxes;
    std::size_t predicted_boxes;
};

void save_segmentation_comparison(const cv::Mat& image_bgr, const cv::Mat& gt_mask, const cv::Mat& pred_mask, const std::filesystem::path& output_path);
void save_detection_comparison(const cv::Mat& image_bgr, const std::vector<Box>& gt_boxes, const std::vector<Box>& pred_boxes, const std::filesystem::path& output_path);
void save_metrics_report(const EvaluationReport& report, const std::optional<double>& fps, const std::filesystem::path& output_path);
void save_per_image_metrics(const std::vector<ImageMetrics>& metrics, const std::filesystem::path& output_path);

#endif
