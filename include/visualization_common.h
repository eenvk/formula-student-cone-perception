//Renzi

#ifndef VISUALIZATION_COMMON_H
#define VISUALIZATION_COMMON_H

#include <filesystem>
#include <string>

#include <opencv2/core.hpp>

// Common helpers used to annotate and save result images.
void add_title(cv::Mat& image, const std::string& title);
void save_image(const cv::Mat& image, const std::filesystem::path& output_path);

#endif