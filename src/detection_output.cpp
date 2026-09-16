//Granati

#include "detection_output.h"

#include "dataset_utils.h"
#include "visualization_common.h"

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

static std::string box_label(const Box& box, bool is_pred){
    std::string label = class_id_to_name(box.class_id);

    //in prediction i show the score
    if(is_pred && box.score.has_value()){
        std::string score = std::to_string(box.score.value());

        score = score.substr(0, score.find('.') + 3);

        return label + " " + score;
    }

    return label;
}

static cv::Mat draw_boxes(const cv::Mat& image_bgr, const std::vector<Box>& boxes, bool show_score){
    cv::Mat result = image_bgr.clone();

    for(const Box& box : boxes){

        //check that the box is inside the image
        int x_min = std::clamp(box.x_min, 0, image_bgr.cols - 1);
        int y_min = std::clamp(box.y_min, 0, image_bgr.rows - 1);
        int x_max = std::clamp(box.x_max - 1, 0, image_bgr.cols - 1);
        int y_max = std::clamp(box.y_max - 1, 0, image_bgr.rows - 1);

        if(x_max < x_min || y_max < y_min){
            continue;
        }

        cv::Scalar color = class_color_bgr(box.class_id);

        //draw the bounding box
        cv::rectangle(result, cv::Point(x_min, y_min), cv::Point(x_max, y_max), color, 2);

        std::string label = box_label(box, show_score);
        cv::Point text_position(x_min, std::max(20, y_min - 5));

        cv::putText(result, label, text_position, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }

    return result;
}

void save_detection_comparison(const cv::Mat& image_bgr, const std::vector<Box>& gt_boxes, const std::vector<Box>& pred_boxes, const std::filesystem::path& output_path){
    cv::Mat gt_image = draw_boxes(image_bgr, gt_boxes, false);
    cv::Mat pred_image = draw_boxes(image_bgr, pred_boxes, true);

    add_title(gt_image, "Ground truth");
    add_title(pred_image, "Prediction");

    cv::Mat comparison;
    std::vector<cv::Mat> images = {gt_image, pred_image};

    cv::hconcat(images, comparison);

    save_image(comparison, output_path);
}