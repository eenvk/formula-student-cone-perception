//Renzi

#include "timing_io.h"

#include <fstream>
#include <sstream>
#include <string>

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