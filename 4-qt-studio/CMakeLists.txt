cmake_minimum_required(VERSION 3.21)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" CACHE STRING "" FORCE)

project(TRTForge LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# 基本设置
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/W3> $<$<COMPILE_LANGUAGE:CXX>:/MP>)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -DNOMINMAX -DUNICODE -D_UNICODE)
endif()

# ---------------------------------------------------------------------------
# Qt6 (位置可被 CMakePresets / 环境变量覆盖)
# ---------------------------------------------------------------------------
if(NOT DEFINED QT_DIR_HINT)
    set(QT_DIR_HINT "D:/QT/6.8.3/msvc2022_64" CACHE PATH "Qt6 installation prefix")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${QT_DIR_HINT}")

find_package(Qt6 6.5 COMPONENTS Core Gui Widgets REQUIRED)
message(STATUS "[TRTForge] Qt6 found: ${Qt6_DIR}")

# ---------------------------------------------------------------------------
# CUDA / TensorRT / cuDNN
# 默认布局：CUDA 11.8 + TensorRT 8.5+ + cuDNN 8.6+
#   include : <CUDA>/include       （TRT/cuDNN 头一并放此）
#   lib     : <CUDA>/lib           （TRT 顶层 lib；cuDNN 在 lib/x64）
#   bin     : <CUDA>/bin           （所有运行时 DLL，包括 TRT + cuDNN）
# 实际版本会在配置阶段从 NvInferVersion.h / cudnn_version.h 解析并打印
# ---------------------------------------------------------------------------
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    if(DEFINED ENV{CUDA_PATH})
        set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_PATH}" CACHE PATH "CUDA root")
    else()
        set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8" CACHE PATH "CUDA root")
    endif()
endif()

set(TRT_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include" CACHE PATH "TensorRT include dir")
# 优先使用 lib，再回退到 lib/x64
if(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/nvinfer.lib")
    set(TRT_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib")
elseif(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvinfer.lib")
    set(TRT_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
else()
    set(TRT_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib")
endif()

if(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/cudart.lib")
    set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib")
else()
    set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
endif()

if(NOT EXISTS "${TRT_INCLUDE_DIR}/NvInfer.h")
    message(FATAL_ERROR "NvInfer.h not found under ${TRT_INCLUDE_DIR}, please pass -DCUDA_TOOLKIT_ROOT_DIR=...")
endif()

# 解析 TensorRT / cuDNN 版本号
function(_parse_version_macros _file _prefix _out)
    if(NOT EXISTS "${_file}")
        set(${_out} "(unknown)" PARENT_SCOPE)
        return()
    endif()
    file(STRINGS "${_file}" _lines REGEX "^#define[ \t]+${_prefix}_[A-Z]+[ \t]+[0-9]+")
    set(_v "")
    foreach(_part MAJOR MINOR PATCH)
        foreach(_l ${_lines})
            if(_l MATCHES "^#define[ \t]+${_prefix}_${_part}[ \t]+([0-9]+)")
                if(_v STREQUAL "")
                    set(_v "${CMAKE_MATCH_1}")
                else()
                    set(_v "${_v}.${CMAKE_MATCH_1}")
                endif()
                break()
            endif()
        endforeach()
    endforeach()
    set(${_out} "${_v}" PARENT_SCOPE)
endfunction()

_parse_version_macros("${TRT_INCLUDE_DIR}/NvInferVersion.h" "NV_TENSORRT" TRT_VERSION_STR)
_parse_version_macros("${TRT_INCLUDE_DIR}/cudnn_version.h"  "CUDNN"        CUDNN_VERSION_STR)

message(STATUS "[TRTForge] TensorRT include : ${TRT_INCLUDE_DIR}")
message(STATUS "[TRTForge] TensorRT lib dir : ${TRT_LIB_DIR}")
message(STATUS "[TRTForge] CUDA lib dir     : ${CUDA_LIB_DIR}")
message(STATUS "[TRTForge] TensorRT version : ${TRT_VERSION_STR}")
message(STATUS "[TRTForge] cuDNN version    : ${CUDNN_VERSION_STR}")

# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
if(NOT DEFINED OPENCV_DIR)
    set(OPENCV_DIR "D:/thirdparty/opencv" CACHE PATH "OpenCV build root")
endif()
find_package(OpenCV REQUIRED PATHS "${OPENCV_DIR}" NO_DEFAULT_PATH)
message(STATUS "[TRTForge] OpenCV ${OpenCV_VERSION} found at ${OpenCV_DIR}")

# ---------------------------------------------------------------------------
# 子工程：推理 DLL
# ocr_infer 依赖 Abseil / Clipper / yaml-cpp 等第三方库，add_subdirectory 会把
# 大量第三方目标一并写进 VS 解决方案。默认将 OCR 作为独立工程单独编译，
# 主工程只链接 bin/ocr_infer.dll 与 bin/ocr_infer.lib，保持 TRTForge 解决方案精简。
# ---------------------------------------------------------------------------
option(TRTFORGE_BUILD_OCR_INFER
    "Build ocr_infer inside TRTForge solution (pulls Abseil third-party targets into VS)"
    OFF)

set(TRTFORGE_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(TRTFORGE_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin")

# TRTFORGE_INFER_BLOCK_BEGIN
add_subdirectory("${TRTFORGE_REPO_ROOT}/3-cpp-tensorrt/yolo_infer" "${CMAKE_BINARY_DIR}/yolo_infer")
add_subdirectory("${TRTFORGE_REPO_ROOT}/3-cpp-tensorrt/anomalib_infer" "${CMAKE_BINARY_DIR}/anomalib_infer")

if(TRTFORGE_BUILD_OCR_INFER)
    add_subdirectory("${TRTFORGE_REPO_ROOT}/ocr/ocr_infer" "${CMAKE_BINARY_DIR}/ocr_infer")
else()
    set(_ocr_infer_dll "${CMAKE_CURRENT_SOURCE_DIR}/bin/ocr_infer.dll")
    set(_ocr_infer_lib "${CMAKE_CURRENT_SOURCE_DIR}/bin/ocr_infer.lib")
    if(NOT EXISTS "${_ocr_infer_lib}")
        set(_ocr_infer_lib "${CMAKE_CURRENT_SOURCE_DIR}/bin/Release/ocr_infer.lib")
    endif()
    if(NOT EXISTS "${_ocr_infer_lib}" OR NOT EXISTS "${_ocr_infer_dll}")
        message(WARNING
            "[TRTForge] TRTFORGE_BUILD_OCR_INFER=OFF but prebuilt ocr_infer not found.\n"
            "  expected dll: ${_ocr_infer_dll}\n"
            "  expected lib: ${_ocr_infer_lib}\n"
            "Build ocr_infer separately first, then re-configure TRTForge.")
    endif()
    add_library(ocr_infer SHARED IMPORTED GLOBAL)
    set_target_properties(ocr_infer PROPERTIES
        IMPORTED_IMPLIB "${_ocr_infer_lib}"
        IMPORTED_LOCATION "${_ocr_infer_dll}"
    )
endif()
# TRTFORGE_INFER_BLOCK_END

# ---------------------------------------------------------------------------
# 源码
# ---------------------------------------------------------------------------
set(TRTFORGE_HEADERS
    include/MainWindow.h
    include/engine/EngineBuilderTab.h
    include/engine/OnnxInputInspector.h
    include/BuilderWorker.h
    include/TRTLogger.h
    include/TRTProgressMonitor.h
    include/common/AppFileLogger.h
    include/common/AppEventFormatter.h
    include/common/AppEvents.h
    include/common/AppSettings.h
    include/BuildOptions.h
    include/EngineTester.h
    include/InferenceTab.h
    include/inference/AnomalyInferenceAdapter.h
    include/inference/InferenceResultPresenter.h
    include/inference/OcrInferenceAdapter.h
    include/inference/YoloInferenceAdapter.h
    include/InferWorker.h
    include/AnomalyInferWorker.h
    include/OcrInferWorker.h
    include/CaptureWorker.h
    include/LabelStore.h
    include/AnnotationProject.h
    include/AnnotationCanvas.h
    include/AnnotationConfirmDialog.h
    include/AnnotationTab.h
    include/DatasetExporter.h
    include/dataset/DatasetSplitPolicy.h
    include/dataset/DatasetYamlWriter.h
    include/YoloTrainController.h
    include/training/TrainEventParser.h
    include/training/TrainingArtifactModel.h
    include/TrainingTab.h
)

set(TRTFORGE_SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/engine/EngineBuilderTab.cpp
    src/engine/OnnxInputInspector.cpp
    src/BuilderWorker.cpp
    src/TRTLogger.cpp
    src/TRTProgressMonitor.cpp
    src/common/AppFileLogger.cpp
    src/common/AppEventFormatter.cpp
    src/common/AppSettings.cpp
    src/EngineTester.cpp
    src/InferenceTab.cpp
    src/inference/AnomalyInferenceAdapter.cpp
    src/inference/InferenceResultPresenter.cpp
    src/inference/OcrInferenceAdapter.cpp
    src/inference/YoloInferenceAdapter.cpp
    src/InferWorker.cpp
    src/AnomalyInferWorker.cpp
    src/OcrInferWorker.cpp
    src/CaptureWorker.cpp
    src/LabelStore.cpp
    src/annotation/AnnotationProject.cpp
    src/annotation/AnnotationCanvas.cpp
    src/annotation/AnnotationConfirmDialog.cpp
    src/annotation/AnnotationTab.cpp
    src/dataset/DatasetExporter.cpp
    src/dataset/DatasetSplitPolicy.cpp
    src/dataset/DatasetYamlWriter.cpp
    src/training/YoloTrainController.cpp
    src/training/TrainEventParser.cpp
    src/training/TrainingArtifactModel.cpp
    src/training/TrainingTab.cpp
)

set(TRTFORGE_RESOURCES
    resources/app.qrc
)

add_executable(TRTForge WIN32
    ${TRTFORGE_HEADERS}
    ${TRTFORGE_SOURCES}
    ${TRTFORGE_RESOURCES}
)

target_include_directories(TRTForge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TRTFORGE_REPO_ROOT}/3-cpp-tensorrt/yolo_infer/include
    ${TRTFORGE_REPO_ROOT}/3-cpp-tensorrt/anomalib_infer/include
    ${TRTFORGE_REPO_ROOT}/ocr/ocr_infer/include
    ${TRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(TRTForge PRIVATE
    ${TRT_LIB_DIR}
    ${CUDA_LIB_DIR}
)

target_link_libraries(TRTForge PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart
    ${OpenCV_LIBS}
    yolo_infer
    anomalib_infer
    ocr_infer
)

if(WIN32)
    target_link_libraries(TRTForge PRIVATE strmiids ole32 oleaut32)
endif()

# ---------------------------------------------------------------------------
# 输出目录：放在 4-qt-studio/bin 下，避免污染仓库根
# ---------------------------------------------------------------------------
set_target_properties(TRTForge PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

# 保留 VS 解决方案中的 INSTALL 目标，方便用户按固定项目列表识别主工程。
# 实际安装包仍由 installer/ 脚本负责，默认编译 TRTForge 不会触发该目标。
install(CODE "message(STATUS \"TRTForge install target is reserved; use installer scripts for packaging.\")")

# ---------------------------------------------------------------------------
# 自动 windeployqt：编译完拷贝 Qt 依赖 dll 到输出目录
# ---------------------------------------------------------------------------
get_target_property(_qmake_exe Qt6::qmake IMPORTED_LOCATION)
get_filename_component(_qt_bin_dir "${_qmake_exe}" DIRECTORY)
find_program(WINDEPLOYQT_EXE windeployqt HINTS "${_qt_bin_dir}")
if(WINDEPLOYQT_EXE)
    add_custom_command(TARGET TRTForge POST_BUILD
        COMMAND "${WINDEPLOYQT_EXE}"
            --no-translations --no-compiler-runtime --no-system-d3d-compiler
            "$<TARGET_FILE:TRTForge>"
        COMMENT "[TRTForge] Running windeployqt..."
    )
endif()

# ---------------------------------------------------------------------------
# 拷贝 OpenCV 运行库到输出目录（world + 可选 ffmpeg，适配多版本预编译包）
# ---------------------------------------------------------------------------
set(OPENCV_DLLS_TO_COPY "")
file(GLOB _opencv_bin_dirs "${OPENCV_DIR}/x64/vc*/bin")
foreach(_obin ${_opencv_bin_dirs})
    file(GLOB _ocv_glob
        "${_obin}/opencv_core*.dll"
        "${_obin}/opencv_dnn*.dll"
        "${_obin}/opencv_imgcodecs*.dll"
        "${_obin}/opencv_imgproc*.dll"
        "${_obin}/opencv_videoio*.dll"
        "${_obin}/opencv_videoio_ffmpeg*.dll"
    )
    foreach(_dll ${_ocv_glob})
        get_filename_component(_ocv_name "${_dll}" NAME)
        if(_ocv_name MATCHES ".*d\\.dll$")
            continue()
        endif()
        list(APPEND OPENCV_DLLS_TO_COPY "${_dll}")
    endforeach()
endforeach()

if(OPENCV_DLLS_TO_COPY)
    list(REMOVE_DUPLICATES OPENCV_DLLS_TO_COPY)
    list(LENGTH OPENCV_DLLS_TO_COPY _ocv_n)
    message(STATUS "[TRTForge] Will copy ${_ocv_n} OpenCV DLL(s) to output dir")
    add_custom_command(TARGET TRTForge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS_TO_COPY} "$<TARGET_FILE_DIR:TRTForge>"
        COMMENT "[TRTForge] Copying OpenCV runtime DLLs..."
    )
else()
    message(WARNING "[TRTForge] No OpenCV opencv_world*.dll found under ${OPENCV_DIR}/x64/vc*/bin. Set OPENCV_DIR and rebuild.")
endif()

# 拷贝训练入口脚本到 bin/training/scripts（安装包与绿色目录均与 TRTForge.exe 同根）
add_custom_command(TARGET TRTForge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:TRTForge>/training/scripts"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TRTFORGE_REPO_ROOT}/1-python-yolo/launcher/tf_yolo_launcher.py"
        "${TRTFORGE_REPO_ROOT}/2-python-anomaly/launcher/tf_anomalib_launcher.py"
        "$<TARGET_FILE_DIR:TRTForge>/training/scripts"
    COMMENT "[TRTForge] Copying training launcher scripts..."
)

# 拷贝 configs 目录到 bin
add_custom_command(TARGET TRTForge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/configs"
        "$<TARGET_FILE_DIR:TRTForge>/configs"
    COMMENT "[TRTForge] Copying configs..."
)

# ---------------------------------------------------------------------------
# 拷贝 TensorRT / cuDNN / CUDA 运行时 DLL 到 bin，保证 exe 自包含
# 仅拷贝 <CUDA>/bin 下实际存在的，缺失的会被忽略（不报错）
# ---------------------------------------------------------------------------
set(_runtime_dll_names
    # TensorRT 8.5+ 完整一组
    nvinfer.dll
    nvinfer_plugin.dll
    nvinfer_dispatch.dll
    nvinfer_lean.dll
    nvinfer_vc_plugin.dll
    nvinfer_builder_resource.dll
    nvonnxparser.dll
    # cuDNN 8.x （nvinfer 在构建/推理时会动态加载）
    cudnn64_8.dll
    cudnn_ops_infer64_8.dll
    cudnn_ops_train64_8.dll
    cudnn_cnn_infer64_8.dll
    cudnn_cnn_train64_8.dll
    cudnn_adv_infer64_8.dll
    cudnn_adv_train64_8.dll
    # CUDA 运行时
    cudart64_110.dll
    cublas64_11.dll
    cublasLt64_11.dll
)

set(_runtime_dlls_to_copy "")
foreach(_n ${_runtime_dll_names})
    if(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/bin/${_n}")
        list(APPEND _runtime_dlls_to_copy "${CUDA_TOOLKIT_ROOT_DIR}/bin/${_n}")
    endif()
endforeach()

if(_runtime_dlls_to_copy)
    list(LENGTH _runtime_dlls_to_copy _n_dll)
    message(STATUS "[TRTForge] Will copy ${_n_dll} runtime DLLs from ${CUDA_TOOLKIT_ROOT_DIR}/bin to output dir")
    add_custom_command(TARGET TRTForge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_runtime_dlls_to_copy}
            "$<TARGET_FILE_DIR:TRTForge>"
        COMMENT "[TRTForge] Copying TensorRT/cuDNN/CUDA runtime DLLs..."
    )
endif()

# PaddleOCR CPU/MKLDNN 运行时依赖。paddle_inference 目录会随 bin 一起打包，
# 但 Windows 加载 mkldnn.dll 时不会自动搜索嵌套的 mklml/lib 目录，因此把
# mklml.dll 与 libiomp5md.dll 同步放到 TRTForge.exe 同目录。
set(_paddle_runtime_dlls_to_copy "")
set(_paddle_runtime_root "${CMAKE_CURRENT_SOURCE_DIR}/bin/paddle_inference")
foreach(_n mklml.dll libiomp5md.dll)
    if(EXISTS "${_paddle_runtime_root}/third_party/install/mklml/lib/${_n}")
        list(APPEND _paddle_runtime_dlls_to_copy
            "${_paddle_runtime_root}/third_party/install/mklml/lib/${_n}")
    endif()
endforeach()

if(_paddle_runtime_dlls_to_copy)
    add_custom_command(TARGET TRTForge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_paddle_runtime_dlls_to_copy}
            "$<TARGET_FILE_DIR:TRTForge>"
        COMMENT "[TRTForge] Copying Paddle MKL runtime DLLs..."
    )
endif()

message(STATUS "[TRTForge] Output dir : ${CMAKE_CURRENT_SOURCE_DIR}/bin")
