#ifndef INFERENCE_RUNNER_H
#define INFERENCE_RUNNER_H

#include <filesystem>

void run_python_inference(const std::filesystem::path& image_dir, const std::filesystem::path& output_dir);

#endif