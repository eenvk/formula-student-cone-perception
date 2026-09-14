//Renzi
#ifndef INFERENCE_RUNNER_H
#define INFERENCE_RUNNER_H

#include <filesystem>

// Launches the Python inference pipeline from the C++ executable.
void run_python_inference(const std::filesystem::path& image_dir, const std::filesystem::path& output_dir);

#endif