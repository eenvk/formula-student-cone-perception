cmake_minimum_required(VERSION 3.10)

project(formula_student_cone_perception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(main
        main.cpp
        src/dataset_utils.cpp
        src/evaluation_common.cpp
        src/segmentation_evaluator.cpp
        src/classification_evaluator.cpp
        src/detection_prediction_io.cpp
        src/timing_io.cpp
        src/visualization_common.cpp
        src/segmentation_output.cpp
        src/detection_output.cpp
        src/report_output.cpp
        src/detection_evaluator.cpp
        src/segmentation_prediction_io.cpp
)

target_link_libraries(main ${OpenCV_LIBS} ZLIB::ZLIB)
target_compile_definitions(main PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")