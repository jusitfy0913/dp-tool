#pragma once

/*
 * 整图分类的 ONNX 推理实现（v1.1.0 新增）。
 *
 * 与 Python 侧 `6-web/backend/classifyinfer.py` 的分工：
 *   - 这边只负责「图 → 原始分数」，预处理逐行对齐 Ultralytics 的
 *     classify_transforms（短边等比缩放 → 中心裁剪 → BGR2RGB → ÷255 → NCHW）；
 *   - softmax、top-k、置信度阈值、类别名映射全部留在 Python。
 *
 * 为什么不在 C++ 里算概率：
 *   classifyinfer.interpret 已经有一套「先看是不是概率、不是才 softmax」的口径，
 *   在这里再写一遍 softmax，两条路径迟早会在边界样本上给出不同的类别。
 *   分数是事实，概率是解释，解释只应该有一份。
 */

#include <string>
#include <vector>

#include "onnx_infer_c.h"
#include "ort_session.hpp"

namespace oi {

class ClassifyOnnx {
public:
    /* 打开分类模型；失败时 err 里是可以直接展示给用户的原因。 */
    bool open(const std::wstring& model_path, std::string& err);

    /* 输入 BGR888。结果保存在句柄内，直到下一次 infer 或销毁。 */
    bool infer(const unsigned char* bgr, int width, int height, int stride,
               oi_frame_stat* stat, std::string& err);

    const oi_classify_result* result() const { return &result_; }
    const std::string& provider() const { return session_.provider(); }

    int input_w() const { return input_w_; }
    int input_h() const { return input_h_; }
    int num_classes() const { return num_classes_; }
    const std::string& names() const { return names_; }

private:
    /* 把任意尺寸的 BGR 图变成 NCHW float32（3×H×W），顺序与 Python 侧一致。 */
    bool preprocess(const unsigned char* bgr, int width, int height, int stride,
                    std::vector<float>& tensor, std::string& err) const;

    Session session_;

    int input_w_ = 0;
    int input_h_ = 0;
    int num_classes_ = 0;

    /* scores_ 必须独立于 result_ 存活：result_.scores 指向它，拷贝结果结构体时
     * 指针仍然有效（调用方按约定只读、且在下一次推理前用完）。 */
    std::vector<float> scores_;
    /* 模型 metadata 的 names 原文；result_.names_utf8 指向它。 */
    std::string names_;
    oi_classify_result result_{};
};

}  // namespace oi
