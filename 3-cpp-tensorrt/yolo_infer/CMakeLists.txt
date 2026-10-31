cmake_minimum_required(VERSION 3.21)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" CACHE STRING "" FORCE)

project(yolo_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/W3> $<$<COMPILE_LANGUAGE:CXX>:/MP>)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -DNOMINMAX)
endif()

# --- OpenCV ---
# 不写死本机路径。查找顺序：-DOPENCV_DIR > 环境变量 OPENCV_DIR > CMake 常规搜索
if(NOT DEFINED OPENCV_DIR AND DEFINED ENV{OPENCV_DIR})
    set(OPENCV_DIR "$ENV{OPENCV_DIR}" CACHE PATH "OpenCV build root")
endif()
if(DEFINED OPENCV_DIR)
    find_package(OpenCV REQUIRED PATHS "${OPENCV_DIR}" NO_DEFAULT_PATH)
else()
    find_package(OpenCV REQUIRED)
endif()
message(STATUS "[yolo_infer] OpenCV ${OpenCV_VERSION} found at ${OpenCV_DIR}")

# --- CUDA / TensorRT 8.5+（继承父工程变量）---
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

# --- 源文件 ---
set(YOLO_INFER_SRC
    src/infer.cu
    src/yolo.cu
    src/yolo_infer_c.cpp
)

add_library(yolo_infer SHARED ${YOLO_INFER_SRC})

target_compile_definitions(yolo_infer PRIVATE YOLO_INFER_EXPORTS)

target_include_directories(yolo_infer
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${TRT_INCLUDE}
            ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(yolo_infer PRIVATE
    ${TRT_LIBDIR}
    ${CUDA_LIBDIR}
)

target_link_libraries(yolo_infer PRIVATE
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    cudart
)

# 输出到 Qt 程序目录 4-qt-studio/bin，与 TRTForge.exe 放在一起
if(NOT DEFINED TRTFORGE_BIN_DIR)
    set(TRTFORGE_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../4-qt-studio/bin")
endif()
set_target_properties(yolo_infer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TRTFORGE_BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TRTFORGE_BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TRTFORGE_BIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRTFORGE_BIN_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${TRTFORGE_BIN_DIR}"
)
