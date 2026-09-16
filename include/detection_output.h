//Granati

#ifndef DETECTION_OUTPUT_H
#define DETECTION_OUTPUT_H

#include "box.h"

#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>

//save the comparison image with the ground truth and prediction side by side
void save_detection_comparison(const cv::Mat& image_bgr, const std::vector<Box>& gt_boxes, const std::vector<Box>& pred_boxes, const std::filesystem::path& output_path);

#endif