//Renzi

#include "visualization_common.h"

#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Draws a readable title using a white stroke with a black inner line.
void add_title(cv::Mat& image, const std::string& title){
    cv::putText(image, title, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(image, title, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

// Creates the destination directory and writes the image to disk.
void save_image(const cv::Mat& image, const std::filesystem::path& output_path){
    std::filesystem::create_directories(output_path.parent_path());

    if (!cv::imwrite(output_path.string(), image)){
        throw std::runtime_error("Could not save image: " + output_path.string());
    }
}