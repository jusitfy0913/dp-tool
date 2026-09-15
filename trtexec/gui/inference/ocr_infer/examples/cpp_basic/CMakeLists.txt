cmake_minimum_required(VERSION 3.21)
project(ocr_basic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ocr_basic main.cpp)

target_include_directories(ocr_basic PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../include"
)

target_link_directories(ocr_basic PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../../bin"
)

target_link_libraries(ocr_basic PRIVATE ocr_infer)
