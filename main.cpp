//

#include "evaluation_metrics.h"
#include "dataset_utils.h"
#include "detection_output.h"
#include "detection_prediction_io.h"
#include "report_output.h"
#include "segmentation_output.h"
#include "segmentation_prediction_io.h"
#include "timing_io.h"

#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifndef PROJECT_ROOT
#define PROJECT_ROOT "."
#endif

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    try {
        fs::path dataset_root = fs::path(PROJECT_ROOT) / "dataset" / "test_set" / "segmentation_test";
        fs::path prediction_root = fs::path(PROJECT_ROOT) / "evaluation" / "cpp_predictions";
        fs::path output_root = fs::path(PROJECT_ROOT) / "evaluation" / "cpp_results";

        if (argc == 4) {
            dataset_root = argv[1];
            prediction_root = argv[2];
            output_root = argv[3];
        } else if (argc != 1) {
            std::cerr << "Usage: " << argv[0] << " [test_dataset_root prediction_root output_root]\n";
            return 1;
        }

        const std::vector<DatasetPair> pairs = find_test_pairs(dataset_root / "img", dataset_root / "ann");
        const DetectionPredictions predictions = load_detection_predictions(prediction_root / "detections.csv");
        const fs::path mask_dir = prediction_root / "masks";

        SegmentationEvaluator segmentation_evaluator;
        ClassificationEvaluator classification_evaluator;
        DetectionEvaluator detection_evaluator;

        std::vector<ImageMetrics> per_image_metrics;

        std::cout << "Test images: " << pairs.size() << "\n";

        for (std::size_t index = 0; index < pairs.size(); ++index) {
            const DatasetPair& pair = pairs[index];

            std::cout << "Processing " << index + 1 << "/" << pairs.size() << ": " << pair.image_path.filename() << "\n";

            const cv::Mat image = load_image(pair.image_path);
            const GroundTruth ground_truth = load_ground_truth(pair.annotation_path, image.rows, image.cols);

            const auto prediction_iterator = predictions.find(pair.image_path.filename().string());
            const std::vector<Box> predicted_boxes = prediction_iterator == predictions.end() ? std::vector<Box>{} : prediction_iterator->second;

            const cv::Mat predicted_mask = load_prediction_mask(mask_dir, pair.image_path, image.size());

            segmentation_evaluator.update(ground_truth.semantic_mask, predicted_mask);
            classification_evaluator.update(ground_truth.boxes, predicted_boxes);
            detection_evaluator.update(ground_truth.boxes, predicted_boxes);

            SegmentationEvaluator image_segmentation_evaluator;
            ClassificationEvaluator image_classification_evaluator;
            DetectionEvaluator image_detection_evaluator;

            image_segmentation_evaluator.update(ground_truth.semantic_mask, predicted_mask);
            image_classification_evaluator.update(ground_truth.boxes, predicted_boxes);
            image_detection_evaluator.update(ground_truth.boxes, predicted_boxes);

            const SegmentationReport image_segmentation_report = image_segmentation_evaluator.report();
            const ClassificationReport image_classification_report = image_classification_evaluator.report();
            const DetectionReport image_detection_report = image_detection_evaluator.report();

            per_image_metrics.push_back({pair.image_path.filename().string(), image_segmentation_report.mean_iou, image_classification_report.macro_f1, image_detection_report.mean_average_precision, ground_truth.boxes.size(), predicted_boxes.size()});

            std::ostringstream filename;
            filename << std::setw(3) << std::setfill('0') << index << "_" << pair.image_path.stem().string() << ".jpg";

            save_segmentation_comparison(image, ground_truth.semantic_mask, predicted_mask, output_root / "segmentation" / filename.str());
            save_detection_comparison(image, ground_truth.boxes, predicted_boxes, output_root / "detection" / filename.str());
        }

        const SegmentationReport segmentation_report = segmentation_evaluator.report();
        const ClassificationReport classification_report = classification_evaluator.report();
        const DetectionReport detection_report = detection_evaluator.report();

        const std::optional<double> fps = load_inference_fps(prediction_root / "timing.csv");

        save_metrics_report(segmentation_report, classification_report, detection_report, fps, output_root / "metrics.txt");
        save_per_image_metrics(per_image_metrics, output_root / "per_image_metrics.csv");

        std::cout << "Macro F1: " << classification_report.macro_f1 << "\n";
        std::cout << "mAP@0.5:0.95: " << detection_report.mean_average_precision << "\n";
        std::cout << "\nmIoU: " << segmentation_report.mean_iou << "\n";

        if (fps.has_value()) {
            std::cout << "FPS: " << fps.value() << "\n";
        }

        std::cout << "Results saved in: " << output_root << "\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}