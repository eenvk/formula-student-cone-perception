//Granati

#include "detection_output.h"

#include "dataset_utils.h"
#include "visualization_common.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>


// Adds a label to a bounding box, optionally including the confidence score.
static std::string box_label(const Box& box, bool show_score) {
    std::string label = class_id_to_name(box.class_id);

    // If the score is available and show_score is true, append the score to the label.
    if (show_score && box.score.has_value()) {
        std::ostringstream stream;

        std::string score = std::to_string(box.score.value());
        score = score.substr(0, score.find('.') + 3);

        return label + " " + score;
    }

    return label;
}

// Draws bounding boxes on an image, optionally showing the confidence score.
static cv::Mat draw_boxes(const cv::Mat& image_bgr, const std::vector<Box>& boxes, bool show_score) {
    cv::Mat result = image_bgr.clone();

    for (const Box& box : boxes) {
        // Clamp the bounding box coordinates to ensure they are within the image boundaries.
        const int x_min = std::clamp(box.x_min, 0, image_bgr.cols - 1);
        const int y_min = std::clamp(box.y_min, 0, image_bgr.rows - 1);
        const int x_max = std::clamp(box.x_max - 1, 0, image_bgr.cols - 1);
        const int y_max = std::clamp(box.y_max - 1, 0, image_bgr.rows - 1);

        if (x_max < x_min || y_max < y_min) {
            continue;
        }

        const cv::Scalar color = class_color_bgr(box.class_id);

        // Draw the rectangle and label on the image.
        cv::rectangle(result, cv::Point(x_min, y_min), cv::Point(x_max, y_max), color, 2);
        cv::putText(result, box_label(box, show_score), cv::Point(x_min, std::max(20, y_min - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }

    return result;
}

// Saves a side-by-side comparison of ground truth and predicted bounding boxes on an image.
void save_detection_comparison(const cv::Mat& image_bgr, const std::vector<Box>& gt_boxes, const std::vector<Box>& pred_boxes, const std::filesystem::path& output_path) {
    cv::Mat gt_image = draw_boxes(image_bgr, gt_boxes, false); // no need of scores
    cv::Mat pred_image = draw_boxes(image_bgr, pred_boxes, true);

    add_title(gt_image, "Ground truth");
    add_title(pred_image, "Prediction");

    cv::Mat comparison;
    cv::hconcat(std::vector<cv::Mat>{gt_image, pred_image}, comparison);

    save_image(comparison, output_path);
}