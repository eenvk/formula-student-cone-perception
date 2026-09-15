//Novkovic

#ifndef SEGMENTATION_PREDICTION_IO_H
#define SEGMENTATION_PREDICTION_IO_H

#include <filesystem>

#include <opencv2/core.hpp>

//Loads a predicted segmentation mask and checks that it has the expected size
cv::Mat load_prediction_mask(const std::filesystem::path& mask_dir, const std::filesystem::path& image_path, const cv::Size& expected_size);

#endif