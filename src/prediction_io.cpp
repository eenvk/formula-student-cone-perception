// Author:

#include "prediction_io.h"

#include "dataset_utils.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>


DetectionPredictions load_detection_predictions(const std::filesystem::path& csv_path) {
    std::ifstream file(csv_path);

    if (!file.is_open()) {
        throw std::runtime_error("Detection CSV not found: " + csv_path.string());
    }

    std::string header;

    if (!std::getline(file, header)) {
        throw std::runtime_error("Detection CSV is empty.");
    }

    if (!header.empty() && header.back() == '\r') {
        header.pop_back();
    }

    if (header != "image_name,x_min,y_min,x_max,y_max,class_id,score") {
        throw std::runtime_error("Invalid detection CSV header.");
    }

    DetectionPredictions predictions;
    std::string line;
    int line_number = 1;

    while (std::getline(file, line)) {
        ++line_number;

        if (line.empty()) {
            continue;
        }

        std::stringstream stream(line);

        std::string image_name;
        std::string x_min;
        std::string y_min;
        std::string x_max;
        std::string y_max;
        std::string class_id;
        std::string score;

        if (!std::getline(stream, image_name, ',') || !std::getline(stream, x_min, ',') || !std::getline(stream, y_min, ',') || !std::getline(stream, x_max, ',') || !std::getline(stream, y_max, ',') || !std::getline(stream, class_id, ',') || !std::getline(stream, score)) {
            throw std::runtime_error("Invalid detection CSV row at line " + std::to_string(line_number));
        }

        Box box;
        box.x_min = std::stoi(x_min);
        box.y_min = std::stoi(y_min);
        box.x_max = std::stoi(x_max);
        box.y_max = std::stoi(y_max);
        box.class_id = std::stoi(class_id);
        box.score = std::stod(score);

        if (box.x_min < 0 || box.y_min < 0 || box.x_max <= box.x_min || box.y_max <= box.y_min) {
            throw std::runtime_error("Invalid bounding box at line " + std::to_string(line_number));
        }

        if (!is_cone_class_id(box.class_id)) {
            throw std::runtime_error("Invalid class ID at line " + std::to_string(line_number));
        }

        if (box.score.value() < 0.0 || box.score.value() > 1.0) {
            throw std::runtime_error("Invalid score at line " + std::to_string(line_number));
        }

        predictions[image_name].push_back(box);
    }

    return predictions;
}


cv::Mat load_prediction_mask(const std::filesystem::path& mask_dir, const std::filesystem::path& image_path, const cv::Size& expected_size) {
    std::filesystem::path mask_path = mask_dir / (image_path.filename().string() + ".png");

    if (!std::filesystem::is_regular_file(mask_path)) {
        mask_path = mask_dir / (image_path.stem().string() + ".png");
    }

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


std::optional<double> load_inference_fps(const std::filesystem::path& timing_path) {
    if (!std::filesystem::is_regular_file(timing_path)) {
        return std::nullopt;
    }

    std::ifstream file(timing_path);

    if (!file.is_open()) {
        return std::nullopt;
    }

    std::string header;
    std::string line;

    if (!std::getline(file, header) || !std::getline(file, line)) {
        return std::nullopt;
    }

    if (!header.empty() && header.back() == '\r') {
        header.pop_back();
    }

    if (header != "num_images,detection_seconds,segmentation_seconds,total_seconds,fps") {
        return std::nullopt;
    }

    std::stringstream stream(line);

    std::string num_images;
    std::string detection_seconds;
    std::string segmentation_seconds;
    std::string total_seconds;
    std::string fps;

    if (!std::getline(stream, num_images, ',') || !std::getline(stream, detection_seconds, ',') || !std::getline(stream, segmentation_seconds, ',') || !std::getline(stream, total_seconds, ',') || !std::getline(stream, fps)) {
        return std::nullopt;
    }

    return std::stod(fps);
}