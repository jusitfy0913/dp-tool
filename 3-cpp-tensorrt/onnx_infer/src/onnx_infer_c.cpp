/*
 * onnx_infer 的 C ABI 实现。
 *
 * 这一层的职责只有三件事：
 *   1. 把 C 参数转成内部类型，并把异常挡在 C 边界之内；
 *   2. 管理句柄生命周期，结果内存随句柄存活；
 *   3. 用 thread_local 的错误串把失败原因交给调用方。
 * 任何推理细节都下沉到 YoloOnnx / AnomalibOnnx，避免这里再出现第二套逻辑。
 */

#include "onnx_infer_c.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

#include "anomalib_onnx.hpp"
#include "ort_session.hpp"
#include "yolo_onnx.hpp"

namespace {

enum class HandleKind { Yolo, Anomalib };

struct Handle {
    HandleKind kind = HandleKind::Yolo;
    std::unique_ptr<oi::YoloOnnx> yolo;
    std::unique_ptr<oi::AnomalibOnnx> anomalib;
};

oi::YoloOnnx* as_yolo(oi_handle handle) {
    if (!handle) return nullptr;
    auto* impl = static_cast<Handle*>(handle);
    return impl->kind == HandleKind::Yolo ? impl->yolo.get() : nullptr;
}

oi::AnomalibOnnx* as_anomalib(oi_handle handle) {
    if (!handle) return nullptr;
    auto* impl = static_cast<Handle*>(handle);
    return impl->kind == HandleKind::Anomalib ? impl->anomalib.get() : nullptr;
}

void copy_text(char* dst, size_t size, const std::string& text) {
    if (!dst || size == 0) return;
    std::snprintf(dst, size, "%s", text.c_str());
}

}  // namespace

extern "C" {

oi_handle oi_create_yolo(const char* onnx_path_utf8, int task, float conf_thr, float nms_thr) {
    oi::set_last_error("");
    try {
        if (!onnx_path_utf8 || !*onnx_path_utf8) {
            oi::set_last_error("ONNX 模型路径为空。");
            return nullptr;
        }

        oi::Runtime& runtime = oi::Runtime::get();
        if (!runtime.available()) {
            oi::set_last_error("ONNX Runtime 不可用：" + runtime.error());
            return nullptr;
        }

        const std::wstring path = oi::utf8_to_wide(onnx_path_utf8);
        if (path.empty()) {
            oi::set_last_error("ONNX 模型路径不是合法的 UTF-8 字符串。");
            return nullptr;
        }

        auto handle = std::make_unique<Handle>();
        handle->kind = HandleKind::Yolo;
        handle->yolo = std::make_unique<oi::YoloOnnx>();

        std::string err;
        if (!handle->yolo->open(path, task, conf_thr, nms_thr, err)) {
            oi::set_last_error(err);
            return nullptr;
        }
        return handle.release();
    } catch (const std::exception& e) {
        oi::set_last_error(std::string("创建 YOLO 句柄时发生异常：") + e.what());
        return nullptr;
    } catch (...) {
        oi::set_last_error("创建 YOLO 句柄时发生未知异常。");
        return nullptr;
    }
}

oi_handle oi_create_anomalib(const char* onnx_path_utf8, const char* metadata_path_utf8,
                             const char* model_type_utf8, float image_threshold_override,
                             float pixel_threshold_override) {
    oi::set_last_error("");
    try {
        if (!onnx_path_utf8 || !*onnx_path_utf8) {
            oi::set_last_error("ONNX 模型路径为空。");
            return nullptr;
        }
        if (!metadata_path_utf8 || !*metadata_path_utf8) {
            oi::set_last_error("metadata.json 路径为空。Anomalib 必须先导出 metadata。");
            return nullptr;
        }

        oi::Runtime& runtime = oi::Runtime::get();
        if (!runtime.available()) {
            oi::set_last_error("ONNX Runtime 不可用：" + runtime.error());
            return nullptr;
        }

        const std::wstring path = oi::utf8_to_wide(onnx_path_utf8);
        if (path.empty()) {
            oi::set_last_error("ONNX 模型路径不是合法的 UTF-8 字符串。");
            return nullptr;
        }

        auto handle = std::make_unique<Handle>();
        handle->kind = HandleKind::Anomalib;
        handle->anomalib = std::make_unique<oi::AnomalibOnnx>();

        std::string err;
        if (!handle->anomalib->open(path, metadata_path_utf8, model_type_utf8,
                                    image_threshold_override, pixel_threshold_override, err)) {
            oi::set_last_error(err);
            return nullptr;
        }
        return handle.release();
    } catch (const std::exception& e) {
        oi::set_last_error(std::string("创建 Anomalib 句柄时发生异常：") + e.what());
        return nullptr;
    } catch (...) {
        oi::set_last_error("创建 Anomalib 句柄时发生未知异常。");
        return nullptr;
    }
}

int oi_infer_yolo(oi_handle handle, const unsigned char* bgr, int width, int height, int stride,
                  oi_box** out_boxes, int* out_count, oi_frame_stat* out_stat) {
    if (out_boxes) *out_boxes = nullptr;
    if (out_count) *out_count = 0;
    if (out_stat) std::memset(out_stat, 0, sizeof(*out_stat));

    if (!out_boxes || !out_count) {
        oi::set_last_error("oi_infer_yolo 的 out_boxes / out_count 不能为空。");
        return OI_ERR_INVALID_ARGUMENT;
    }

    try {
        oi::YoloOnnx* yolo = as_yolo(handle);
        if (!yolo) {
            oi::set_last_error("句柄无效，或不是 YOLO 句柄。");
            return OI_ERR_INVALID_ARGUMENT;
        }

        std::string err;
        if (!yolo->infer(bgr, width, height, stride, out_stat, err)) {
            oi::set_last_error(err);
            return OI_ERR_INFERENCE;
        }

        const std::vector<oi_box>& boxes = yolo->boxes();
        *out_count = static_cast<int>(boxes.size());
        *out_boxes = boxes.empty() ? nullptr : const_cast<oi_box*>(boxes.data());
        return OI_OK;
    } catch (const std::exception& e) {
        oi::set_last_error(std::string("YOLO 推理时发生异常：") + e.what());
        return OI_ERR_INFERENCE;
    } catch (...) {
        oi::set_last_error("YOLO 推理时发生未知异常。");
        return OI_ERR_INFERENCE;
    }
}

int oi_infer_anomalib(oi_handle handle, const unsigned char* rgb, int width, int height, int stride,
                      const oi_anomaly_result** out_result, oi_frame_stat* out_stat) {
    if (out_result) *out_result = nullptr;
    if (out_stat) std::memset(out_stat, 0, sizeof(*out_stat));

    if (!out_result) {
        oi::set_last_error("oi_infer_anomalib 的 out_result 不能为空。");
        return OI_ERR_INVALID_ARGUMENT;
    }

    try {
        oi::AnomalibOnnx* anomalib = as_anomalib(handle);
        if (!anomalib) {
            oi::set_last_error("句柄无效，或不是 Anomalib 句柄。");
            return OI_ERR_INVALID_ARGUMENT;
        }

        std::string err;
        if (!anomalib->infer(rgb, width, height, stride, out_stat, err)) {
            oi::set_last_error(err);
            return OI_ERR_INFERENCE;
        }

        *out_result = anomalib->result();
        return OI_OK;
    } catch (const std::exception& e) {
        oi::set_last_error(std::string("Anomalib 推理时发生异常：") + e.what());
        return OI_ERR_INFERENCE;
    } catch (...) {
        oi::set_last_error("Anomalib 推理时发生未知异常。");
        return OI_ERR_INFERENCE;
    }
}

int oi_get_info(oi_handle handle, oi_info* out_info) {
    if (!out_info) {
        oi::set_last_error("oi_get_info 的 out_info 不能为空。");
        return OI_ERR_INVALID_ARGUMENT;
    }
    std::memset(out_info, 0, sizeof(*out_info));

    if (!handle) {
        oi::set_last_error("句柄为空。");
        return OI_ERR_INVALID_ARGUMENT;
    }

    auto* impl = static_cast<Handle*>(handle);
    copy_text(out_info->ort_version, sizeof(out_info->ort_version), oi::Runtime::get().ort_version());

    if (impl->kind == HandleKind::Yolo && impl->yolo) {
        out_info->task = impl->yolo->task();
        out_info->input_w = impl->yolo->input_w();
        out_info->input_h = impl->yolo->input_h();
        out_info->num_classes = impl->yolo->num_classes();
        out_info->num_keypoints = impl->yolo->num_keypoints();
        copy_text(out_info->provider, sizeof(out_info->provider), impl->yolo->provider());
        return OI_OK;
    }
    if (impl->kind == HandleKind::Anomalib && impl->anomalib) {
        out_info->task = -1;
        out_info->input_w = impl->anomalib->input_w();
        out_info->input_h = impl->anomalib->input_h();
        out_info->num_classes = 0;
        out_info->num_keypoints = 0;
        copy_text(out_info->provider, sizeof(out_info->provider), impl->anomalib->provider());
        return OI_OK;
    }

    oi::set_last_error("句柄内部状态无效。");
    return OI_ERR_INVALID_ARGUMENT;
}

void oi_destroy(oi_handle handle) {
    if (!handle) return;
    try {
        delete static_cast<Handle*>(handle);
    } catch (...) {
        // 析构不应抛异常，这里兜底避免跨越 C 边界。
    }
}

const char* oi_last_error(void) { return oi::last_error(); }

}  // extern "C"
