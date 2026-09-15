//Renzi

#include "inference_runner.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

// Builds the platform-specific command and executes the Python inference bridge.
void run_python_inference(const std::filesystem::path& image_dir, const std::filesystem::path& output_dir) {
    const std::filesystem::path project_root = PROJECT_ROOT;
    const std::filesystem::path inference_script = project_root / "python" / "inference_bridge.py";

    if (!std::filesystem::is_regular_file(inference_script)) {
        throw std::runtime_error("Inference script not found: " + inference_script.string());
    }

#ifdef _WIN32
    const std::filesystem::path python_executable = project_root / ".venv" / "Scripts" / "python.exe";

    if (!std::filesystem::is_regular_file(python_executable)) {
        throw std::runtime_error("Python executable not found: " + python_executable.string());
    }

    const std::string command = "cmd /C \"\"" + python_executable.string() + "\" \"" + inference_script.string() + "\" --image_dir \"" + image_dir.string() + "\" --output_dir \"" + output_dir.string() + "\"\"";
#else
    const std::string command = "python \"" + inference_script.string() + "\" --image_dir \"" + image_dir.string() + "\" --output_dir \"" + output_dir.string() + "\"";
#endif

    std::cout << "Running Python inference..." << std::endl;

    const int result = std::system(command.c_str());

    if (result != 0) {
        throw std::runtime_error("Python inference failed.");
    }
}