//Renzi

#ifndef TIMING_IO_H
#define TIMING_IO_H

#include <filesystem>

// Reads the FPS value produced by the inference pipeline.
double load_fps(const std::filesystem::path& csv_path);

#endif