#pragma once

/*
 * YOLO 的 ONNX 推理实现。
 *
 * 与 yolo_infer.dll 的分工：
 *   - 那边跑 TensorRT 引擎，CUDA 内核做 letterbox / decode / NMS；
 *   - 这边跑 ONNX，用 OpenCV 在 CPU 上做同样的几何变换与解码。
 *
 * 两者必须给出可比较的结果，所以仿射矩阵、padding 值、坐标还原公式都从
 * yolo.cu 的 AffineMatrix 与 decode kernel 一比一移植，而不是自己另定一套。
 */

#include <cstdint>
#include <string>
#include <vector>

#include "onnx_infer_c.h"
#include "ort_session.hpp"

namespace oi {

class YoloOnnx {
public:
    bool open(const std::wstring& model_path, int task, float conf, float nms, std::string& err);

    /* 输入 BGR888，与 yi_infer_bgr 一致。结果保存在句柄内，直到下一次 infer。 */
    bool infer(const unsigned char* bgr, int width, int height, int stride,
               oi_frame_stat* stat, std::string& err);

    const std::vector<oi_box>& boxes() const { return boxes_; }
    const std::string& provider() const { return session_.provider(); }

    int task() const { return task_; }
    int input_w() const { return input_w_; }
    int input_h() const { return input_h_; }
    int num_classes() const { return num_classes_; }
    int num_keypoints() const { return num_keypoints_; }

private:
    /* 网络输出布局：transposed=true 表示 [1, 候选框, 通道]，false 表示 [1, 通道, 候选框]。 */
    struct OutputLayout {
        bool transposed = true;
        int64_t rows = 0;   // 候选框数量
        int64_t cols = 0;   // 每个候选框的通道数
    };

    /* NMS 之前的候选框，保留 row 以便回到原始张量取附属数据（关键点/掩码系数）。 */
    struct RawBox {
        float left = 0, top = 0, right = 0, bottom = 0;
        float score = 0;
        int label = 0;
        int row = 0;
    };

    bool resolve_layout(std::string& err);
    bool decode(const std::vector<std::vector<float>>& outputs,
                int src_w, int src_h, const float* d2i, const float* i2d, std::string& err);

    /* t() 按布局取第 row 个候选框的第 col 个通道。 */
    static float at(const float* data, const OutputLayout& layout, int64_t row, int64_t col);
    /* 逐类 NMS，返回 RawBox 下标（按分数降序）。 */
    std::vector<int> nms_by_class(const std::vector<RawBox>& candidates) const;

    Session session_;

    int task_ = OI_DETECT;
    float conf_ = 0.25f;
    float nms_ = 0.45f;
    int input_w_ = 0;
    int input_h_ = 0;
    int num_classes_ = 0;
    int num_keypoints_ = 0;

    int det_index_ = 0;
    int proto_index_ = -1;
    int proto_dim_ = 0;
    int proto_h_ = 0;
    int proto_w_ = 0;
    OutputLayout det_layout_;

    /* 每个框的附属数据。用独立容器是为了让 oi_box 里的指针在拷贝后仍然有效。 */
    std::vector<std::vector<unsigned char>> masks_;
    std::vector<std::vector<float>> kpts_;

    std::vector<oi_box> boxes_;
};

}  // namespace oi
