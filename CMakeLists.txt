cmake_minimum_required(VERSION 3.10)

project(formula_student_cone_perception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(main
        main.cpp
)

target_link_libraries(main ${OpenCV_LIBS})