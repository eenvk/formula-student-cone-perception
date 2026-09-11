// Author:

#include "output_utils.h"

#include "dataset_utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

std::string format_value(double value) {
    if (std::isnan(value)) {
        return "nan";
    }

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << value;
    return stream.str();
}

std::string box_label(const Box& box, bool show_score) {
    std::string label = class_id_to_name(box.class_id);

    if (show_score && box.score.has_value()) {
        std::ostringstream stream;
        stream << label << " " << std::fixed << std::setprecision(2) << box.score.value();
        return stream.str();
    }

    return label;
}

void add_title(cv::Mat& image, const std::string& title) {
    cv::putText(image, title, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(image, title, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

cv::Mat draw_boxes(const cv::Mat& image_bgr, const std::vector<Box>& boxes, bool show_score) {
    cv::Mat result = image_bgr.clone();

    for (const Box& box : boxes) {
        const int x_min = std::clamp(box.x_min, 0, image_bgr.cols - 1);
        const int y_min = std::clamp(box.y_min, 0, image_bgr.rows - 1);
        const int x_max = std::clamp(box.x_max - 1, 0, image_bgr.cols - 1);
        const int y_max = std::clamp(box.y_max - 1, 0, image_bgr.rows - 1);

        if (x_max < x_min || y_max < y_min) {
            continue;
        }

        const cv::Scalar color = class_color_bgr(box.class_id);
        cv::rectangle(result, cv::Point(x_min, y_min), cv::Point(x_max, y_max), color, 2);
        cv::putText(result, box_label(box, show_score), cv::Point(x_min, std::max(20, y_min - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }

    return result;
}

void save_image(const cv::Mat& image, const std::filesystem::path& output_path) {
    std::filesystem::create_directories(output_path.parent_path());

    if (!cv::imwrite(output_path.string(), image)) {
        throw std::runtime_error("Could not save image: " + output_path.string());
    }
}

}

void save_segmentation_comparison(const cv::Mat& image_bgr, const cv::Mat& gt_mask, const cv::Mat& pred_mask, const std::filesystem::path& output_path) {
    cv::Mat original = image_bgr.clone();
    cv::Mat gt_overlay = create_overlay(image_bgr, gt_mask);
    cv::Mat pred_overlay = create_overlay(image_bgr, pred_mask);

    add_title(original, "Original");
    add_title(gt_overlay, "Ground truth");
    add_title(pred_overlay, "Prediction");

    cv::Mat comparison;
    cv::hconcat(std::vector<cv::Mat>{original, gt_overlay, pred_overlay}, comparison);
    save_image(comparison, output_path);
}

void save_detection_comparison(const cv::Mat& image_bgr, const std::vector<Box>& gt_boxes, const std::vector<Box>& pred_boxes, const std::filesystem::path& output_path) {
    cv::Mat gt_image = draw_boxes(image_bgr, gt_boxes, false);
    cv::Mat pred_image = draw_boxes(image_bgr, pred_boxes, true);

    add_title(gt_image, "Ground truth");
    add_title(pred_image, "Prediction");

    cv::Mat comparison;
    cv::hconcat(std::vector<cv::Mat>{gt_image, pred_image}, comparison);
    save_image(comparison, output_path);
}

void save_metrics_report(const EvaluationReport& report, const std::optional<double>& fps, const std::filesystem::path& output_path) {
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error("Could not create metrics report.");
    }

    file << "Segmentation results\n";
    file << "--------------------\n";

    for (const auto& [class_name, iou] : report.segmentation.iou_per_class) {
        file << class_name << " IoU: " << format_value(iou) << "\n";
    }

    file << "mIoU: " << format_value(report.segmentation.mean_iou) << "\n\n";
    file << "Classification results\n";
    file << "----------------------\n";

    for (const auto& [class_name, f1] : report.classification.f1_per_class) {
        file << class_name << " F1: " << format_value(f1) << "\n";
    }

    file << "Macro F1: " << format_value(report.classification.macro_f1) << "\n\n";
    file << "Detection results\n";
    file << "-----------------\n";

    for (const auto& [class_name, ap] : report.detection.ap_per_class) {
        file << class_name << " AP@0.5:0.95: " << format_value(ap) << "\n";
    }

    file << "mAP@0.5:0.95: " << format_value(report.detection.mean_average_precision) << "\n";

    if (fps.has_value()) {
        file << "\nFPS: " << format_value(fps.value()) << "\n";
    }
}

void save_per_image_metrics(const std::vector<ImageMetrics>& metrics, const std::filesystem::path& output_path) {
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error("Could not create per-image metrics file.");
    }

    file << "image_name,miou,macro_f1,map_50_95,ground_truth_boxes,predicted_boxes\n";

    for (const ImageMetrics& image_metrics : metrics) {
        file << image_metrics.image_name << "," << format_value(image_metrics.mean_iou) << "," << format_value(image_metrics.macro_f1) << "," << format_value(image_metrics.mean_average_precision) << "," << image_metrics.ground_truth_boxes << "," << image_metrics.predicted_boxes << "\n";
    }
}
