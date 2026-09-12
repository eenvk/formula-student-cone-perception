//Renzi

#include "report_output.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

    std::string format_value(double value) {
        if (std::isnan(value)) {
            return "nan";
        }

        std::ostringstream stream;
        stream << std::fixed << std::setprecision(6) << value;
        return stream.str();
    }

}

void save_metrics_report(const SegmentationReport& segmentation_report, const ClassificationReport& classification_report, const DetectionReport& detection_report, const std::optional<double>& fps, const std::filesystem::path& output_path) {
    std::filesystem::create_directories(output_path.parent_path());

    std::ofstream file(output_path);

    if (!file.is_open()) {
        throw std::runtime_error("Could not create metrics report.");
    }

    file << "Segmentation results\n";
    file << "--------------------\n";

    for (const auto& [class_name, iou] : segmentation_report.iou_per_class) {
        file << class_name << " IoU: " << format_value(iou) << "\n";
    }

    file << "mIoU: " << format_value(segmentation_report.mean_iou) << "\n\n";

    file << "Classification results\n";
    file << "----------------------\n";

    for (const auto& [class_name, precision] : classification_report.precision_per_class) {
        const double recall = classification_report.recall_per_class.at(class_name);
        const double f1 = classification_report.f1_per_class.at(class_name);

        file << class_name << ": Precision=" << format_value(precision) << " Recall=" << format_value(recall) << " F1=" << format_value(f1) << "\n";
    }

    file << "Macro F1: " << format_value(classification_report.macro_f1) << "\n\n";

    file << "Detection results\n";
    file << "-----------------\n";

    for (const auto& [class_name, ap] : detection_report.ap_per_class) {
        file << class_name << " AP@0.5:0.95: " << format_value(ap) << "\n";
    }

    file << "mAP@0.5:0.95: " << format_value(detection_report.mean_average_precision) << "\n";

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