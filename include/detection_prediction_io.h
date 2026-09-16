//Granati

#ifndef DETECTION_PREDICTION_IO_H
#define DETECTION_PREDICTION_IO_H

#include "box.h"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

using DetectionPredictions = std::unordered_map<std::string, std::vector<Box>>;

//load predictions from a csv file, the format is:
//image_name,x_min,y_min,x_max,y_max,class_id,score
DetectionPredictions load_detection_predictions(const std::filesystem::path& csv_path);

#endif