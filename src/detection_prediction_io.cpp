//

#include "detection_prediction_io.h"

#include "dataset_utils.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

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