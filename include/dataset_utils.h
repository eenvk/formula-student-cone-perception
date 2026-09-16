//Renzi

#ifndef DATASET_UTILS_H
#define DATASET_UTILS_H

#include "box.h"

#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

// Class identifiers shared by masks, detections and evaluation code.
constexpr int BACKGROUND_ID = 0;
constexpr int YELLOW_CONE_ID = 1;
constexpr int BLUE_CONE_ID = 2;
constexpr int SMALL_ORANGE_CONE_ID = 3;
constexpr int BIG_ORANGE_CONE_ID = 4;
constexpr int IGNORE_ID = 255;
constexpr int NUM_CLASSES = 5;

inline constexpr std::array<int, 4> CONE_CLASS_IDS = {YELLOW_CONE_ID, BLUE_CONE_ID, SMALL_ORANGE_CONE_ID, BIG_ORANGE_CONE_ID};

// Associates each test image with its annotation file.
struct DatasetPair{
    std::filesystem::path image_path;
    std::filesystem::path annotation_path;
};

// Ground-truth data extracted from an annotation.
struct GroundTruth{
    cv::Mat semantic_mask;
    std::vector<Box> boxes;
};

// Class conversion and visualization helpers.
bool is_cone_class_id(int class_id);
std::string class_id_to_name(int class_id);
cv::Scalar class_color_bgr(int class_id);

// Dataset loading and preprocessing utilities.
cv::Mat load_image(const std::filesystem::path& image_path);
std::vector<DatasetPair> find_test_pairs(const std::filesystem::path& image_dir, const std::filesystem::path& annotation_dir);
GroundTruth load_ground_truth(const std::filesystem::path& annotation_path, int image_height, int image_width);
cv::Mat create_overlay(const cv::Mat& image_bgr, const cv::Mat& mask, double alpha = 0.45);

#endif
