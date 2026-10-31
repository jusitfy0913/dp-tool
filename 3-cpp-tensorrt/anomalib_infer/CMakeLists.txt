cmake_minimum_required(VERSION 3.21)

project(anomalib_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/W3> $<$<COMPILE_LANGUAGE:CXX>:/MP>)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -DNOMINMAX)
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
message(STATUS "[anomalib_infer] OpenCV ${OpenCV_VERSION} found at ${OPENCV_DIR}")

if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    if(DEFINED ENV{CUDA_PATH})
        set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_PATH}" CACHE PATH "")
    else()
        set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8" CACHE PATH "")
    endif()
endif()

set(TRT_INCLUDE "${CUDA_TOOLKIT_ROOT_DIR}/include")
if(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/nvinfer.lib")
    set(TRT_LIBDIR "${CUDA_TOOLKIT_ROOT_DIR}/lib")
elseif(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvinfer.lib")
    set(TRT_LIBDIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
else()
    set(TRT_LIBDIR "${CUDA_TOOLKIT_ROOT_DIR}/lib")
endif()

if(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/cudart.lib")
    set(CUDA_LIBDIR "${CUDA_TOOLKIT_ROOT_DIR}/lib")
else()
    set(CUDA_LIBDIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
endif()

set(ANOMALIB_INFER_SRC
    src/anomalib_infer_c.cpp
)

add_library(anomalib_infer SHARED ${ANOMALIB_INFER_SRC})

target_compile_definitions(anomalib_infer PRIVATE ANOMALIB_INFER_EXPORTS)

target_include_directories(anomalib_infer
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${TRT_INCLUDE}
            ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(anomalib_infer PRIVATE
    ${TRT_LIBDIR}
    ${CUDA_LIBDIR}
)

target_link_libraries(anomalib_infer PRIVATE
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    cudart
)

if(NOT DEFINED TRTFORGE_BIN_DIR)
    set(TRTFORGE_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../4-qt-studio/bin")
endif()
set_target_properties(anomalib_infer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TRTFORGE_BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TRTFORGE_BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TRTFORGE_BIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRTFORGE_BIN_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${TRTFORGE_BIN_DIR}"
)
