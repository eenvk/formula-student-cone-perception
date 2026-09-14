//Renzi

#include "timing_io.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

double load_fps(const std::filesystem::path& csv_path) {
    std::ifstream file(csv_path);

    if (!file.is_open()) {
        throw std::runtime_error("Timing CSV not found: " + csv_path.string());
    }

    std::string header;

    if (!std::getline(file, header)) {
        throw std::runtime_error("Timing CSV is empty.");
    }

    if (!header.empty() && header.back() == '\r') {
        header.pop_back();
    }

    if (header != "num_images,detection_seconds,segmentation_seconds,total_seconds,fps") {
        throw std::runtime_error("Invalid timing CSV header.");
    }

    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("Timing CSV does not contain timing values.");
    }

    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }

    std::stringstream stream(line);

    std::string num_images;
    std::string detection_seconds;
    std::string segmentation_seconds;
    std::string total_seconds;
    std::string fps;

    if (!std::getline(stream, num_images, ',') || !std::getline(stream, detection_seconds, ',') || !std::getline(stream, segmentation_seconds, ',') || !std::getline(stream, total_seconds, ',') || !std::getline(stream, fps)) {
        throw std::runtime_error("Invalid timing CSV row.");
    }

    const int image_count = std::stoi(num_images);
    const double fps_value = std::stod(fps);

    if (image_count <= 0) {
        throw std::runtime_error("Invalid number of images in timing CSV.");
    }

    if (fps_value < 0.0) {
        throw std::runtime_error("Invalid FPS value in timing CSV.");
    }

    return fps_value;
}