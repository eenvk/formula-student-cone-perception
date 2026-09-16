//Renzi

#ifndef REPORT_OUTPUT_H
#define REPORT_OUTPUT_H

#include "evaluation_metrics.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

// Metrics collected for a single test image.
struct ImageMetrics{
    std::string image_name;
    double mean_iou;
    double macro_f1;
    double mean_average_precision;
    std::size_t ground_truth_boxes;
    std::size_t predicted_boxes;
};

// Writes aggregate metrics and per-image results to disk.
void save_metrics_report(const SegmentationReport& segmentation_report, const ClassificationReport& classification_report, const DetectionReport& detection_report, const std::optional<double>& fps, const std::filesystem::path& output_path);
void save_per_image_metrics(const std::vector<ImageMetrics>& metrics, const std::filesystem::path& output_path);

#endif