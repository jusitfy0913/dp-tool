cmake_minimum_required(VERSION 3.21)

project(onnx_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8 /W3 /MP)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -DNOMINMAX)
endif()

# ---------------------------------------------------------------------------
# OpenCV：与另外两个推理 DLL 完全相同的查找顺序
# ---------------------------------------------------------------------------
if(NOT DEFINED OPENCV_DIR AND DEFINED ENV{OPENCV_DIR})
    set(OPENCV_DIR "$ENV{OPENCV_DIR}" CACHE PATH "OpenCV build root")
endif()
if(DEFINED OPENCV_DIR)
    find_package(OpenCV REQUIRED PATHS "${OPENCV_DIR}" NO_DEFAULT_PATH)
else()
    find_package(OpenCV REQUIRED)
endif()
message(STATUS "[onnx_infer] OpenCV ${OpenCV_VERSION} found at ${OPENCV_DIR}")

# ---------------------------------------------------------------------------
# ONNX Runtime：只需要头文件与运行期 DLL
#
# 刻意不链接 onnxruntime.lib：
#   后端进程里可能已经加载了 Python 的 onnxruntime.dll，按模块名链接会复用那一份。
#   ort_session.cpp 用 LoadLibraryExW + OrtGetApiBase 显式加载，来源可控。
# ---------------------------------------------------------------------------
if(NOT DEFINED ONNXRUNTIME_ROOT)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime root")
    else()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime root")
    endif()
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_c_api.h")
    message(WARNING
        "[onnx_infer] 找不到 ONNX Runtime 头文件：${ONNXRUNTIME_ROOT}/include/onnxruntime_c_api.h\n"
        "  先运行：powershell -NoProfile -ExecutionPolicy Bypass -File scripts/fetch_onnxruntime.ps1\n"
        "  然后重新配置。本次跳过 onnx_infer 目标，不影响 yolo_infer / anomalib_infer。")
    return()
endif()

# ---------------------------------------------------------------------------
# 源文件
# ---------------------------------------------------------------------------
set(ONNX_INFER_SRC
    src/ort_session.cpp
    src/yolo_onnx.cpp
    src/anomalib_onnx.cpp
    src/classify_onnx.cpp
    src/onnx_infer_c.cpp
)

add_library(onnx_infer SHARED ${ONNX_INFER_SRC})

target_compile_definitions(onnx_infer PRIVATE ONNX_INFER_EXPORTS)

target_include_directories(onnx_infer
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/../anomalib_infer/src
            ${ONNXRUNTIME_ROOT}/include
            ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(onnx_infer PRIVATE ${OpenCV_LIBS})

if(NOT DEFINED TRTFORGE_BIN_DIR)
    set(TRTFORGE_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../4-qt-studio/bin")
endif()

set_target_properties(onnx_infer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TRTFORGE_BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TRTFORGE_BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TRTFORGE_BIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRTFORGE_BIN_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${TRTFORGE_BIN_DIR}"
)

# 把 ORT 运行库放到 DLL 旁边。ort_session.cpp 优先按本 DLL 所在目录查找，
# 部署时只要整个 bin 目录一起拷贝就能工作。
if(EXISTS "${ONNXRUNTIME_ROOT}/lib")
    add_custom_command(TARGET onnx_infer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${ONNXRUNTIME_ROOT}/lib" "${TRTFORGE_BIN_DIR}"
        COMMENT "[onnx_infer] 复制 ONNX Runtime 运行库到 ${TRTFORGE_BIN_DIR}"
    )
endif()
