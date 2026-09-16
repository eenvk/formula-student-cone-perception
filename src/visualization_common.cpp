//Renzi

#include "visualization_common.h"
#include "dataset_utils.h"

#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

// Validates the basic format expected for semantic masks.
    void validate_mask(const cv::Mat &mask) {
        if (mask.empty() || mask.dims != 2 || mask.channels() != 1) {
            throw std::invalid_argument("Mask must be a non-empty single-channel image.");
        }
    }

// Converts class IDs into BGR colors for visualization.
    cv::Mat colorize_mask(const cv::Mat &mask) {
        validate_mask(mask);
        cv::Mat color_mask(mask.rows, mask.cols, CV_8UC3);

        for (int row = 0; row < mask.rows; ++row) {
            for (int column = 0; column < mask.cols; ++column) {
                const int class_id = static_cast<int>(mask.at<unsigned char>(row, column));
                const cv::Scalar color = class_color_bgr(class_id);
                color_mask.at<cv::Vec3b>(row, column) = cv::Vec3b(static_cast<unsigned char>(color[0]),
                                                                  static_cast<unsigned char>(color[1]),
                                                                  static_cast<unsigned char>(color[2]));
            }
        }

        return color_mask;
    }

}

// Returns the BGR color associated with a class.
cv::Scalar class_color_bgr(int class_id){
    switch (class_id){
        case BACKGROUND_ID:
            return cv::Scalar(0, 0, 0);
        case YELLOW_CONE_ID:
            return cv::Scalar(0, 255, 255);
        case BLUE_CONE_ID:
            return cv::Scalar(255, 0, 0);
        case SMALL_ORANGE_CONE_ID:
            return cv::Scalar(0, 165, 255);
        case BIG_ORANGE_CONE_ID:
            return cv::Scalar(255, 0, 255);
        case IGNORE_ID:
            return cv::Scalar(255, 255, 255);
        default:
            throw std::invalid_argument("Invalid class ID: " + std::to_string(class_id));
    }
}

// Blends the colored semantic mask with the original image.
cv::Mat create_overlay(const cv::Mat& image_bgr, const cv::Mat& mask, double alpha){
    if (image_bgr.empty() || image_bgr.channels() != 3){
        throw std::invalid_argument("Image must be a non-empty three-channel image.");
    }

    validate_mask(mask);

    if (image_bgr.size() != mask.size()){
        throw std::invalid_argument("Image and mask sizes must match.");
    }

    if (alpha < 0.0 || alpha > 1.0){
        throw std::invalid_argument("Alpha must be in the range [0, 1].");
    }

    const cv::Mat color_mask = colorize_mask(mask);

    cv::Mat blended;
    cv::addWeighted(image_bgr, 1.0 - alpha, color_mask, alpha, 0.0, blended);

    cv::Mat foreground;
    cv::compare(mask, BACKGROUND_ID, foreground, cv::CMP_NE);

    cv::Mat overlay = image_bgr.clone();
    blended.copyTo(overlay, foreground);
    return overlay;
}

// Draws a readable title using a white stroke with a black inner line.
void add_title(cv::Mat& image, const std::string& title){
    cv::putText(image, title, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(image, title, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

// Creates the destination directory and writes the image to disk.
void save_image(const cv::Mat& image, const std::filesystem::path& output_path){
    std::filesystem::create_directories(output_path.parent_path());

    if (!cv::imwrite(output_path.string(), image)){
        throw std::runtime_error("Could not save image: " + output_path.string());
    }
}