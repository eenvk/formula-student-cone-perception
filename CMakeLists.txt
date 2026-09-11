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
        src/evaluation_utils.cpp
        src/output_utils.cpp
        src/prediction_io.cpp
)

target_link_libraries(main ${OpenCV_LIBS} ZLIB::ZLIB)
target_compile_definitions(main PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")