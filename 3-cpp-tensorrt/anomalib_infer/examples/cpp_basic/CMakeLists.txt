cmake_minimum_required(VERSION 3.21)

project(anomalib_infer_cpp_basic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8 /W3)
endif()

# 不写死本机路径。查找顺序：-DOPENCV_DIR > 环境变量 OPENCV_DIR > CMake 常规搜索
if(NOT DEFINED OPENCV_DIR AND DEFINED ENV{OPENCV_DIR})
    set(OPENCV_DIR "$ENV{OPENCV_DIR}" CACHE PATH "OpenCV build root")
endif()
if(DEFINED OPENCV_DIR)
    find_package(OpenCV REQUIRED PATHS "${OPENCV_DIR}" NO_DEFAULT_PATH)
else()
    find_package(OpenCV REQUIRED)
endif()

add_executable(anomalib_infer_cpp_basic main.cpp)

target_include_directories(anomalib_infer_cpp_basic PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(anomalib_infer_cpp_basic PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../4-qt-studio/bin
)

target_link_libraries(anomalib_infer_cpp_basic PRIVATE
    anomalib_infer
    ${OpenCV_LIBS}
)
