// Novkovic

#include "segmentation_prediction_io.h"

#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>

cv::Mat load_prediction_mask(const std::filesystem::path& mask_dir, const std::filesystem::path& image_path, const cv::Size& expected_size) {
    const std::filesystem::path mask_path = mask_dir / (image_path.stem().string() + ".png");

    if (!std::filesystem::is_regular_file(mask_path)) {
        throw std::runtime_error("Prediction mask not found for: " + image_path.filename().string());
    }

    const cv::Mat mask = cv::imread(mask_path.string(), cv::IMREAD_GRAYSCALE);

    if (mask.empty()) {
        throw std::runtime_error("Could not decode prediction mask: " + mask_path.string());
    }

    if (mask.size() != expected_size) {
        throw std::runtime_error("Prediction mask size does not match image: " + image_path.filename().string());
    }

    return mask;
}