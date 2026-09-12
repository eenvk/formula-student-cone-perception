//

#ifndef DETECTION_PREDICTION_IO_H
#define DETECTION_PREDICTION_IO_H

#include "box.h"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

using DetectionPredictions = std::unordered_map<std::string, std::vector<Box>>;

DetectionPredictions load_detection_predictions(const std::filesystem::path& csv_path);

#endif