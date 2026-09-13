//

#include "dataset_utils.h"
#include "detection_output.h"
#include "detection_prediction_io.h"
#include "evaluation_metrics.h"
#include "inference_runner.h"
#include "report_output.h"
#include "segmentation_output.h"
#include "segmentation_prediction_io.h"
#include "timing_io.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const std::filesystem::path project_root = PROJECT_ROOT;

    const std::filesystem::path image_dir = project_root / "dataset" / "test_set" / "segmentation_test" / "img";
    const std::filesystem::path annotation_dir = project_root / "dataset" / "test_set" / "segmentation_test" / "ann";

    const std::filesystem::path prediction_dir = project_root / "evaluation" / "cpp_predictions";
    const std::filesystem::path detection_output_dir = project_root / "evaluation" / "results" / "detection";
    const std::filesystem::path segmentation_output_dir = project_root / "evaluation" / "results" / "segmentation";

    const std::filesystem::path report_path = project_root / "evaluation" / "metrics_report.txt";
    const std::filesystem::path per_image_metrics_path = project_root / "evaluation" / "per_image_metrics.csv";

    const bool run_inference = true;

    if (run_inference) {
        run_python_inference(image_dir, prediction_dir);
    }

    const DetectionPredictions detection_predictions = load_detection_predictions(prediction_dir / "detections.csv");
    const std::vector<DatasetPair> test_pairs = find_test_pairs(image_dir, annotation_dir);

    SegmentationEvaluator segmentation_evaluator;
    ClassificationEvaluator classification_evaluator;
    DetectionEvaluator detection_evaluator;

    std::vector<ImageMetrics> per_image_metrics;

    for (const DatasetPair& pair : test_pairs) {
        const cv::Mat image = load_image(pair.image_path);
        const GroundTruth ground_truth = load_ground_truth(pair.annotation_path, image.rows, image.cols);

        const std::string image_name = pair.image_path.filename().string();
        const std::string image_stem = pair.image_path.stem().string();

        std::vector<Box> predicted_boxes;

        const DetectionPredictions::const_iterator detection_iterator = detection_predictions.find(image_name);

        if (detection_iterator != detection_predictions.end()) {
            predicted_boxes = detection_iterator->second;
        }

        const cv::Mat predicted_mask = load_prediction_mask(prediction_dir / "masks", pair.image_path, image.size());

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

        per_image_metrics.push_back({image_name, image_segmentation_report.mean_iou, image_classification_report.macro_f1, image_detection_report.mean_average_precision, ground_truth.boxes.size(), predicted_boxes.size()});

        const std::filesystem::path detection_output_path = detection_output_dir / (image_stem + ".jpg");
        const std::filesystem::path segmentation_output_path = segmentation_output_dir / (image_stem + ".jpg");

        save_detection_comparison(image, ground_truth.boxes, predicted_boxes, detection_output_path);
        save_segmentation_comparison(image, ground_truth.semantic_mask, predicted_mask, segmentation_output_path);

        std::cout << image_name << " | GT boxes: " << ground_truth.boxes.size() << " | predicted boxes: " << predicted_boxes.size() << std::endl;
    }

    const SegmentationReport segmentation_report = segmentation_evaluator.report();
    const ClassificationReport classification_report = classification_evaluator.report();
    const DetectionReport detection_report = detection_evaluator.report();

    const double fps = load_fps(prediction_dir / "timing.csv");

    save_metrics_report(segmentation_report, classification_report, detection_report, fps, report_path);
    save_per_image_metrics(per_image_metrics, per_image_metrics_path);

    std::cout << std::endl;
    std::cout << "Evaluation completed." << std::endl;
    std::cout << "mIoU: " << segmentation_report.mean_iou << std::endl;
    std::cout << "Macro F1: " << classification_report.macro_f1 << std::endl;
    std::cout << "mAP@0.5:0.95: " << detection_report.mean_average_precision << std::endl;
    std::cout << "FPS: " << fps << std::endl;

    std::cout << std::endl;
    std::cout << "Metrics report: " << report_path << std::endl;
    std::cout << "Per-image metrics: " << per_image_metrics_path << std::endl;
    std::cout << "Detection results: " << detection_output_dir << std::endl;
    std::cout << "Segmentation results: " << segmentation_output_dir << std::endl;

    return 0;
}