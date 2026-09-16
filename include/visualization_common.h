//Renzi

#ifndef VISUALIZATION_COMMON_H
#define VISUALIZATION_COMMON_H

#include <filesystem>
#include <string>

#include <opencv2/core.hpp>


// Common helpers used to visualize and save result images.
void add_title(cv::Mat& image, const std::string& title);
void save_image(const cv::Mat& image, const std::filesystem::path& output_path);
cv::Scalar class_color_bgr(int class_id);
cv::Mat create_overlay(const cv::Mat& image_bgr, const cv::Mat& mask, double alpha = 0.45);
#endif