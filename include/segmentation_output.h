//Novkovic

#ifndef SEGMENTATION_OUTPUT_H
#define SEGMENTATION_OUTPUT_H

#include <filesystem>

#include <opencv2/core.hpp>

//creates and saves a comparison between the original image,ground truth,predicted segmentation
void save_segmentation_comparison(const cv::Mat& image_bgr, const cv::Mat& gt_mask, const cv::Mat& pred_mask, const std::filesystem::path& output_path);

#endif