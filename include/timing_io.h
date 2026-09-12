//Renzi

#ifndef TIMING_IO_H
#define TIMING_IO_H

#include <filesystem>
#include <optional>

std::optional<double> load_inference_fps(const std::filesystem::path& timing_path);

#endif