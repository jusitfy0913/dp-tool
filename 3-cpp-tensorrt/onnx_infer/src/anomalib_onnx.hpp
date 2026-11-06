#pragma once

/*
 * Anomalib 的 ONNX 推理实现（PatchCore / PaDiM / EfficientAd）。
 *
 * 与 anomalib_infer.dll 的关系：
 *   - 预处理尺寸、归一化方式、阈值来源、判定边界与渲染完全一致；
 *   - 渲染与归一化直接复用 anomalib_post.hpp，保证两条后端不会各说各话。
 *
 * 输出解析不绑定具体模型：先按输出名找 score / anomaly_map，再按元素数兜底，
 * 因为三个模型导出的输出名与维度并不统一（score 可能是 [1] 也可能是 [1,1]）。
 */

#include <string>
#include <vector>

#include "onnx_infer_c.h"
#include "ort_session.hpp"

namespace oi {

class AnomalibOnnx {
public:
    bool open(const std::wstring& model_path, const char* metadata_path_utf8,
              const char* model_type_utf8, float image_threshold_override,
              float pixel_threshold_override, std::string& err);

    /* 输入 RGB888，与 ai_infer_rgb 一致。 */
    bool infer(const unsigned char* rgb, int width, int height, int stride,
               oi_frame_stat* stat, std::string& err);

    const oi_anomaly_result* result() const { return &result_; }
    const std::string& provider() const { return session_.provider(); }
    const std::string& model_type() const { return model_type_; }

    int input_w() const { return input_w_; }
    int input_h() const { return input_h_; }

private:
    Session session_;

    std::string model_type_;
    std::string normalization_ = "none";
    float image_threshold_ = 0.5f;
    float pixel_threshold_ = 0.5f;
    float score_min_ = 0.0f;
    float score_max_ = 1.0f;
    float map_min_ = 0.0f;
    float map_max_ = 1.0f;

    int input_w_ = 256;
    int input_h_ = 256;

    /* 结果缓冲由句柄持有，oi_anomaly_result 里的指针指向它们。 */
    std::vector<unsigned char> source_rgb_;
    std::vector<unsigned char> overlay_rgb_;
    std::vector<unsigned char> mask_rgb_;
    std::vector<float> normalized_map_;
    oi_anomaly_result result_{};
};

}  // namespace oi
