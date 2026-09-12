//Novkovic

#include "segmentation_output.h"

#include "dataset_utils.h"
#include "visualization_common.h"

#include <vector>

#include <opencv2/imgproc.hpp>

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