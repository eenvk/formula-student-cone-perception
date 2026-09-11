// Author:

#ifndef PREDICTION_IO_H
#define PREDICTION_IO_H

#include "box.h"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

using DetectionPredictions = std::unordered_map<std::string, std::vector<Box>>;

DetectionPredictions load_detection_predictions(const std::filesystem::path& csv_path);
cv::Mat load_prediction_mask(const std::filesystem::path& mask_dir, const std::filesystem::path& image_path, const cv::Size& expected_size);
std::optional<double> load_inference_fps(const std::filesystem::path& timing_path);

#endif
