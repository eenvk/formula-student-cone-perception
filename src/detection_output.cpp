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

namespace {

    std::string box_label(const Box& box, bool show_score) {
        std::string label = class_id_to_name(box.class_id);

        if (show_score && box.score.has_value()) {
            std::ostringstream stream;
            stream << label << " " << std::fixed << std::setprecision(2) << box.score.value();
            return stream.str();
        }

        return label;
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