cmake_minimum_required(VERSION 3.21)

project(ocr_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    cmake_policy(SET CMP0091 NEW)
    # Must match paddle_inference.dll (official GPU/CPU packages use /MD).
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/W3> $<$<COMPILE_LANGUAGE:CXX>:/MP>)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -DNOMINMAX -DGOOGLE_GLOG_DLL_DECL=)
endif()

if(NOT DEFINED OPENCV_DIR)
    set(OPENCV_DIR "D:/thirdparty/opencv" CACHE PATH "OpenCV build root")
endif()
find_package(OpenCV REQUIRED PATHS "${OPENCV_DIR}" NO_DEFAULT_PATH)
message(STATUS "[ocr_infer] OpenCV ${OpenCV_VERSION} found at ${OPENCV_DIR}")

set(OCR_INFER_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(REPO_ROOT "${OCR_INFER_DIR}/../..")
set(TRTFORGE_GUI_DIR "${REPO_ROOT}/4-qt-studio")

if(NOT DEFINED PADDLE_LIB)
    set(PADDLE_LIB "${TRTFORGE_GUI_DIR}/bin/paddle_inference" CACHE PATH "Paddle Inference package root")
endif()
if(NOT DEFINED PADDLEOCR_CPP_INFER_DIR)
    set(PADDLEOCR_CPP_INFER_DIR "${REPO_ROOT}/ocr/PaddleOCR-main/deploy/cpp_infer" CACHE PATH "PaddleOCR cpp_infer root")
endif()
if(NOT DEFINED TRTFORGE_BIN_DIR)
    set(TRTFORGE_BIN_DIR "${TRTFORGE_GUI_DIR}/bin")
endif()

message(STATUS "[ocr_infer] PADDLE_LIB             : ${PADDLE_LIB}")
message(STATUS "[ocr_infer] PADDLEOCR_CPP_INFER_DIR: ${PADDLEOCR_CPP_INFER_DIR}")

set(THIRD_PARTY_PATH "${OCR_INFER_DIR}/third_party")
function(download_and_decompress url filename decompress_dir)
    if(NOT EXISTS "${filename}" AND NOT EXISTS "${decompress_dir}")
        message(STATUS "[ocr_infer] Downloading ${url}")
        file(DOWNLOAD ${url} "${filename}.tmp" SHOW_PROGRESS)
        file(RENAME "${filename}.tmp" ${filename})
    endif()
    if(NOT EXISTS ${decompress_dir})
        file(MAKE_DIRECTORY ${decompress_dir})
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${filename}
                        WORKING_DIRECTORY ${decompress_dir})
    endif()
endfunction()

set(PACKAGE_LIST abseil-cpp clipper_ver6.4.2 nlohmann)
foreach(PKG ${PACKAGE_LIST})
    set(PKG_URL "https://paddle-model-ecology.bj.bcebos.com/paddlex/cpp/libs/${PKG}.tgz")
    set(PKG_TGZ_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PKG}.tgz")
    set(PKG_DST_PATH "${THIRD_PARTY_PATH}/${PKG}")
    download_and_decompress(${PKG_URL} ${PKG_TGZ_PATH} ${PKG_DST_PATH})
endforeach()

add_subdirectory("${THIRD_PARTY_PATH}/clipper_ver6.4.2/cpp"
                 "${CMAKE_CURRENT_BINARY_DIR}/clipper" EXCLUDE_FROM_ALL)

# 关闭 Abseil 自带的 CTest 仪表盘目标（Continuous / Nightly 等），减少 VS 工程数量
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory("${THIRD_PARTY_PATH}/abseil-cpp" "${CMAKE_CURRENT_BINARY_DIR}/abseil-cpp" EXCLUDE_FROM_ALL)

set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_MSVC_SHARED_RT ON CACHE BOOL "" FORCE)
add_subdirectory("${THIRD_PARTY_PATH}/yaml-cpp" "${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp" EXCLUDE_FROM_ALL)

file(GLOB_RECURSE PADDLEOCR_SRC "${PADDLEOCR_CPP_INFER_DIR}/src/*.cc")
list(FILTER PADDLEOCR_SRC EXCLUDE REGEX ".*/src/utils/args\\.cc$")

add_library(ocr_infer SHARED
    src/ocr_infer_c.cpp
    ${PADDLEOCR_SRC}
)

target_compile_definitions(ocr_infer PRIVATE
    OCR_INFER_EXPORTS
    YAML_CPP_STATIC_DEFINE
)

if(MSVC)
    target_compile_options(ocr_infer PRIVATE "/FI${CMAKE_CURRENT_SOURCE_DIR}/compat/msvc_posix.h")
endif()

target_include_directories(ocr_infer
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/compat"
        "${PADDLEOCR_CPP_INFER_DIR}"
        "${PADDLEOCR_CPP_INFER_DIR}/src"
        "${PADDLE_LIB}/paddle/include"
        "${PADDLE_LIB}/third_party/install/protobuf/include"
        "${PADDLE_LIB}/third_party/install/glog/include"
        "${PADDLE_LIB}/third_party/install/gflags/include"
        "${PADDLE_LIB}/third_party/install/xxhash/include"
        "${PADDLE_LIB}/third_party/install/zlib/include"
        "${PADDLE_LIB}/third_party/install/onnxruntime/include"
        "${PADDLE_LIB}/third_party/install/paddle2onnx/include"
        "${PADDLE_LIB}/third_party/install/openvino/include"
        "${PADDLE_LIB}/third_party/install/tbb/include"
        "${PADDLE_LIB}/third_party/install/mklml/include"
        "${PADDLE_LIB}/third_party/install/onednn/include"
        "${PADDLE_LIB}/third_party/boost"
        "${PADDLE_LIB}/third_party/eigen3"
        "${OpenCV_INCLUDE_DIRS}"
        "${POLYCLIPPING_INCLUDE_DIR}"
)

target_link_directories(ocr_infer PRIVATE
    "${PADDLE_LIB}/paddle/lib"
    "${PADDLE_LIB}/third_party/install/zlib/lib"
    "${PADDLE_LIB}/third_party/install/protobuf/lib"
    "${PADDLE_LIB}/third_party/install/glog/lib"
    "${PADDLE_LIB}/third_party/install/gflags/lib"
    "${PADDLE_LIB}/third_party/install/xxhash/lib"
    "${PADDLE_LIB}/third_party/install/mklml/lib"
    "${PADDLE_LIB}/third_party/install/onednn/lib"
)

set(PADDLE_INFERENCE_LIB "${PADDLE_LIB}/paddle/lib/paddle_inference.lib")
target_link_libraries(ocr_infer PRIVATE
    "${PADDLE_INFERENCE_LIB}"
    ${OpenCV_LIBS}
    polyclipping
    absl::statusor
    yaml-cpp::yaml-cpp
    glog
    gflags_static
    libprotobuf
    xxhash
    mklml
    libiomp5md
    mkldnn
)

if(WIN32)
    target_link_libraries(ocr_infer PRIVATE shlwapi)
endif()

if(MSVC)
    target_link_options(ocr_infer PRIVATE /NODEFAULTLIB:LIBCMT /NODEFAULTLIB:LIBCPMT)
endif()

set_target_properties(ocr_infer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TRTFORGE_BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TRTFORGE_BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TRTFORGE_BIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRTFORGE_BIN_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${TRTFORGE_BIN_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${TRTFORGE_BIN_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${TRTFORGE_BIN_DIR}"
)

option(OCR_INFER_BUILD_EXAMPLES "Build ocr_basic example executable" OFF)
if(OCR_INFER_BUILD_EXAMPLES)
    add_executable(ocr_basic examples/cpp_basic/main.cpp)
    target_include_directories(ocr_basic PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(ocr_basic PRIVATE ocr_infer ${OpenCV_LIBS})
    set_target_properties(ocr_basic PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TRTFORGE_BIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TRTFORGE_BIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TRTFORGE_BIN_DIR}"
    )
endif()
