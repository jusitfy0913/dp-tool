/*
 * YOLO ONNX 推理。
 *
 * 关键取舍：
 *   1. 预处理用 cv::warpAffine + 仿射矩阵，而不是先 resize 再补边。
 *      这样与 yolo.cu 的 AffineMatrix 完全同源，坐标还原误差只有插值实现层面的差异。
 *   2. 解码前不强制转置输出张量，而是用 at() 按布局取值，避免为大模型额外拷一份内存。
 *   3. 掩码只对通过 NMS 的框计算。proto 与系数的点积是 O(通道数 × ROI 面积)，
 *      对 8400 个候选框全算会让后处理耗时超过推理本身。
 */

#include "yolo_onnx.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace oi {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

std::string shape_text(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) oss << ",";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

/* 与 yolo.cu 中的 AffineMatrix 完全一致的构造方式。 */
struct AffineMatrix {
    float i2d[6] = {1, 0, 0, 0, 1, 0};  // 原图 -> 网络输入
    float d2i[6] = {1, 0, 0, 0, 1, 0};  // 网络输入 -> 原图

    void compute(float from_w, float from_h, float to_w, float to_h) {
        const float scale = std::min(to_w / from_w, to_h / from_h);
        i2d[0] = scale;
        i2d[1] = 0.0f;
        i2d[2] = -scale * from_w * 0.5f + to_w * 0.5f + scale * 0.5f - 0.5f;
        i2d[3] = 0.0f;
        i2d[4] = scale;
        i2d[5] = -scale * from_h * 0.5f + to_h * 0.5f + scale * 0.5f - 0.5f;

        cv::Mat forward(2, 3, CV_32F, i2d);
        cv::Mat inverse;
        cv::invertAffineTransform(forward, inverse);
        std::memcpy(d2i, inverse.ptr<float>(), sizeof(d2i));
    }

    cv::Mat matrix() const {
        cv::Mat m(2, 3, CV_32F);
        std::memcpy(m.ptr<float>(), i2d, sizeof(i2d));
        return m;
    }
};

inline void affine_project(const float* m, float x, float y, float& ox, float& oy) {
    ox = m[0] * x + m[1] * y + m[2];
    oy = m[3] * x + m[4] * y + m[5];
}

inline float sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

inline int clamp_int(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

/* 判断去掉 batch 维之后是否为二维检测输出。 */
bool strip_batch(const std::vector<int64_t>& shape, std::vector<int64_t>& out) {
    out = shape;
    if (!out.empty() && out[0] == 1) {
        out.erase(out.begin());
    }
    return true;
}

}  // namespace

bool YoloOnnx::open(const std::wstring& model_path, int task, float conf, float nms,
                    std::string& err) {
    task_ = task;
    conf_ = conf;
    nms_ = nms;

    if (task_ < OI_DETECT || task_ > OI_OBB) {
        err = "任务类型无效，只支持 detect / segment / pose / obb。";
        return false;
    }
    if (conf_ <= 0.0f || conf_ > 1.0f) {
        err = "置信度阈值必须在 (0, 1] 之间。";
        return false;
    }
    if (nms_ <= 0.0f || nms_ > 1.0f) {
        err = "NMS 阈值必须在 (0, 1] 之间。";
        return false;
    }

    if (!session_.open(model_path, err)) {
        return false;
    }

    const auto& inputs = session_.inputs();
    if (inputs.size() != 1) {
        err = "YOLO ONNX 应只有 1 个输入，实际有 " + std::to_string(inputs.size()) + " 个。";
        return false;
    }

    const TensorInfo& input = inputs[0];
    if (input.shape.size() != 4) {
        err = "YOLO ONNX 输入应为 4 维 NCHW，实际形状 " + shape_text(input.shape) + "。";
        return false;
    }
    if (input.shape[1] != 3) {
        err = "YOLO ONNX 输入的通道维应为 3（NCHW），实际形状 " + shape_text(input.shape) +
              "。如果这是 NHWC 导出的模型，请改用 NCHW 重新导出。";
        return false;
    }
    if (input.shape[2] <= 0 || input.shape[3] <= 0) {
        err = "YOLO ONNX 的输入尺寸是动态的 " + shape_text(input.shape) +
              "。本 DLL 只接受固定尺寸导出，请用 imgsz=640（或 960）重新导出 ONNX。";
        return false;
    }

    input_h_ = static_cast<int>(input.shape[2]);
    input_w_ = static_cast<int>(input.shape[3]);

    return resolve_layout(err);
}

bool YoloOnnx::resolve_layout(std::string& err) {
    const auto& outputs = session_.outputs();

    // 找出唯一的二维输出作为检测头；分割模型的原型输出是三维，不会命中。
    det_index_ = -1;
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::vector<int64_t> dims;
        strip_batch(outputs[i].shape, dims);
        if (dims.size() != 2) {
            continue;
        }
        if (dims[0] <= 0 || dims[1] <= 0) {
            err = "检测输出 " + outputs[i].name + " 含动态维度 " + shape_text(outputs[i].shape) +
                  "，无法确定候选框数量与通道数。";
            return false;
        }
        det_index_ = static_cast<int>(i);
        // 通道数通常远小于候选框数量（例如 84 vs 8400），据此判断是否转置。
        if (dims[0] <= dims[1]) {
            det_layout_.transposed = false;
            det_layout_.cols = dims[0];
            det_layout_.rows = dims[1];
        } else {
            det_layout_.transposed = true;
            det_layout_.cols = dims[1];
            det_layout_.rows = dims[0];
        }
        break;
    }

    if (det_index_ < 0) {
        err = "找不到二维检测输出，输出列表为：";
        for (const TensorInfo& info : outputs) {
            err += " " + info.name + shape_text(info.shape);
        }
        err += "。请确认这是 Ultralytics 导出的 YOLO 模型。";
        return false;
    }

    // 是否存在分割原型输出（去掉 batch 后三维、且第一维是小通道数）。
    int proto_candidate = -1;
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (static_cast<int>(i) == det_index_) {
            continue;
        }
        std::vector<int64_t> dims;
        strip_batch(outputs[i].shape, dims);
        if (dims.size() == 3) {
            if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) {
                err = "分割原型输出 " + outputs[i].name + " 含动态维度 " +
                      shape_text(outputs[i].shape) + "。";
                return false;
            }
            proto_candidate = static_cast<int>(i);
            break;
        }
    }

    if (task_ != OI_SEGMENT && proto_candidate >= 0) {
        err = "模型带有分割原型输出，看起来是分割模型。请把任务切换为 segment。";
        return false;
    }

    const int64_t cols = det_layout_.cols;

    switch (task_) {
        case OI_DETECT: {
            num_classes_ = static_cast<int>(cols - 4);
            if (num_classes_ <= 0) {
                err = "检测输出通道数为 " + std::to_string(cols) + "，不足以容纳 4 个框坐标。";
                return false;
            }
            if (cols == 56) {
                err = "该模型通道数为 56，符合姿态模型 5+3×17 的结构。请把任务切换为 pose。";
                return false;
            }
            if (num_classes_ > 1000) {
                err = "由通道数推导出的类别数为 " + std::to_string(num_classes_) +
                      "，数量异常。请确认任务类型与模型是否匹配。";
                return false;
            }
            break;
        }
        case OI_OBB: {
            num_classes_ = static_cast<int>(cols - 5);
            if (num_classes_ <= 0) {
                err = "旋转框输出通道数为 " + std::to_string(cols) +
                      "，不足以容纳 4 个坐标、类别与角度。";
                return false;
            }
            break;
        }
        case OI_POSE: {
            const int64_t rest = cols - 5;
            if (rest <= 0 || rest % 3 != 0) {
                err = "姿态模型通道数为 " + std::to_string(cols) +
                      "，不符合 5+3k 结构（4 框坐标 + 置信度 + k 个关键点三元组）。";
                return false;
            }
            num_keypoints_ = static_cast<int>(rest / 3);
            num_classes_ = 1;
            break;
        }
        case OI_SEGMENT: {
            if (proto_candidate < 0) {
                err = "找不到分割原型输出。该模型更像纯检测模型，请把任务切换为 detect。";
                return false;
            }
            std::vector<int64_t> dims;
            strip_batch(outputs[proto_candidate].shape, dims);
            proto_index_ = proto_candidate;
            proto_dim_ = static_cast<int>(dims[0]);
            proto_h_ = static_cast<int>(dims[1]);
            proto_w_ = static_cast<int>(dims[2]);

            num_classes_ = static_cast<int>(cols - 4 - proto_dim_);
            if (num_classes_ <= 0) {
                err = "检测通道数 " + std::to_string(cols) + " 减去原型通道 " +
                      std::to_string(proto_dim_) + " 后不足以容纳类别数。";
                return false;
            }
            break;
        }
        default:
            err = "未知任务类型。";
            return false;
    }

    return true;
}

float YoloOnnx::at(const float* data, const OutputLayout& layout, int64_t row, int64_t col) {
    return layout.transposed ? data[row * layout.cols + col] : data[col * layout.rows + row];
}

std::vector<int> YoloOnnx::nms_by_class(const std::vector<RawBox>& candidates) const {
    std::vector<int> kept;
    for (int label = 0; label < num_classes_; ++label) {
        std::vector<cv::Rect2d> rects;
        std::vector<float> scores;
        std::vector<int> local;
        for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
            if (candidates[i].label != label) {
                continue;
            }
            const double w = std::max(1e-3, static_cast<double>(candidates[i].right - candidates[i].left));
            const double h = std::max(1e-3, static_cast<double>(candidates[i].bottom - candidates[i].top));
            rects.emplace_back(candidates[i].left, candidates[i].top, w, h);
            scores.push_back(candidates[i].score);
            local.push_back(i);
        }
        if (rects.empty()) {
            continue;
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(rects, scores, conf_, nms_, keep);
        for (int index : keep) {
            kept.push_back(local[index]);
        }
    }
    return kept;
}

bool YoloOnnx::infer(const unsigned char* bgr, int width, int height, int stride,
                     oi_frame_stat* stat, std::string& err) {
    if (!bgr || width <= 0 || height <= 0) {
        err = "输入图像无效。";
        return false;
    }
    if (stride <= 0) {
        stride = width * 3;
    }
    if (stride < width * 3) {
        err = "输入图像的 stride 小于 width*3，数据长度不足。";
        return false;
    }

    const Clock::time_point t0 = Clock::now();

    AffineMatrix affine;
    affine.compute(static_cast<float>(width), static_cast<float>(height),
                   static_cast<float>(input_w_), static_cast<float>(input_h_));

    cv::Mat src(height, width, CV_8UC3, const_cast<unsigned char*>(bgr), static_cast<size_t>(stride));
    cv::Mat warped;
    cv::warpAffine(src, warped, affine.matrix(), cv::Size(input_w_, input_h_),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat rgb;
    cv::cvtColor(warped, rgb, cv::COLOR_BGR2RGB);

    cv::Mat floats;
    rgb.convertTo(floats, CV_32FC3, 1.0 / 255.0);

    std::vector<float> tensor(static_cast<size_t>(3) * input_w_ * input_h_);
    {
        std::vector<cv::Mat> planes;
        cv::split(floats, planes);
        const size_t plane_size = static_cast<size_t>(input_w_) * input_h_;
        for (int c = 0; c < 3; ++c) {
            std::memcpy(tensor.data() + c * plane_size, planes[c].ptr<float>(), plane_size * sizeof(float));
        }
    }

    const Clock::time_point t1 = Clock::now();

    std::vector<std::vector<float>> outputs;
    if (!session_.run({tensor}, outputs, err)) {
        return false;
    }

    const Clock::time_point t2 = Clock::now();

    if (!decode(outputs, width, height, affine.d2i, affine.i2d, err)) {
        return false;
    }

    const Clock::time_point t3 = Clock::now();

    if (stat) {
        stat->src_w = width;
        stat->src_h = height;
        stat->count = static_cast<int>(boxes_.size());
        stat->preprocess_ms = elapsed_ms(t0, t1);
        stat->inference_ms = elapsed_ms(t1, t2);
        stat->postprocess_ms = elapsed_ms(t2, t3);
        stat->total_ms = elapsed_ms(t0, t3);
    }
    return true;
}

bool YoloOnnx::decode(const std::vector<std::vector<float>>& outputs, int src_w, int src_h,
                      const float* d2i, const float* i2d, std::string& err) {
    boxes_.clear();
    masks_.clear();
    kpts_.clear();

    if (det_index_ < 0 || static_cast<size_t>(det_index_) >= outputs.size()) {
        err = "检测输出索引无效。";
        return false;
    }

    const std::vector<float>& det = outputs[det_index_];
    const int64_t rows = det_layout_.rows;
    const int64_t cols = det_layout_.cols;
    if (static_cast<int64_t>(det.size()) < rows * cols) {
        err = "检测输出元素数 " + std::to_string(det.size()) + " 少于期望的 " +
              std::to_string(rows * cols) + "。";
        return false;
    }

    std::vector<RawBox> candidates;
    candidates.reserve(512);

    // 分割模型的系数在类别之后，先记录每行的起始偏移。
    const int64_t mask_coeff_offset = (task_ == OI_SEGMENT) ? (4 + num_classes_) : 0;

    for (int64_t row = 0; row < rows; ++row) {
        float score = 0.0f;
        int label = 0;

        if (task_ == OI_POSE) {
            score = at(det.data(), det_layout_, row, 4);
            label = 0;
        } else {
            score = at(det.data(), det_layout_, row, 4);
            for (int c = 1; c < num_classes_; ++c) {
                const float value = at(det.data(), det_layout_, row, 4 + c);
                if (value > score) {
                    score = value;
                    label = c;
                }
            }
        }

        if (score < conf_) {
            continue;
        }

        RawBox box;
        box.score = score;
        box.label = label;
        box.row = static_cast<int>(row);

        if (task_ == OI_OBB) {
            // OBB 的 (cx,cy) 需要经 d2i 还原，w/h 按统一缩放比还原，角度保持原值。
            const float cx_in = at(det.data(), det_layout_, row, 0);
            const float cy_in = at(det.data(), det_layout_, row, 1);
            const float w_in = at(det.data(), det_layout_, row, 2);
            const float h_in = at(det.data(), det_layout_, row, 3);
            const float angle = at(det.data(), det_layout_, row, 4 + num_classes_);

            float cx = 0.0f, cy = 0.0f;
            affine_project(d2i, cx_in, cy_in, cx, cy);
            const float scale = d2i[0];
            const float ow = w_in * scale;
            const float oh = h_in * scale;

            const float cos_a = std::fabs(std::cos(angle));
            const float sin_a = std::fabs(std::sin(angle));
            const float aabb_w = ow * cos_a + oh * sin_a;
            const float aabb_h = ow * sin_a + oh * cos_a;

            box.left = cx - aabb_w * 0.5f;
            box.top = cy - aabb_h * 0.5f;
            box.right = cx + aabb_w * 0.5f;
            box.bottom = cy + aabb_h * 0.5f;
        } else {
            const float cx = at(det.data(), det_layout_, row, 0);
            const float cy = at(det.data(), det_layout_, row, 1);
            const float w = at(det.data(), det_layout_, row, 2);
            const float h = at(det.data(), det_layout_, row, 3);

            box.left = cx - w * 0.5f;
            box.top = cy - h * 0.5f;
            box.right = cx + w * 0.5f;
            box.bottom = cy + h * 0.5f;
            affine_project(d2i, box.left, box.top, box.left, box.top);
            affine_project(d2i, box.right, box.bottom, box.right, box.bottom);

            // 裁掉 letterbox 溢出到图像外的部分，避免掩码尺寸为负。
            box.left = std::max(0.0f, box.left);
            box.top = std::max(0.0f, box.top);
            box.right = std::min(static_cast<float>(src_w - 1), box.right);
            box.bottom = std::min(static_cast<float>(src_h - 1), box.bottom);
            if (box.right <= box.left || box.bottom <= box.top) {
                continue;
            }
        }

        candidates.push_back(box);
    }

    // 旋转框用 RotatedRect 的 NMS，避免两个斜框的 AABB 重叠但本体不重叠被误删。
    std::vector<int> kept;
    if (task_ == OI_OBB) {
        std::vector<cv::RotatedRect> shapes;
        std::vector<float> scores;
        std::vector<int> local;
        for (int c = 0; c < num_classes_; ++c) {
            shapes.clear();
            scores.clear();
            local.clear();
            for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
                if (candidates[i].label != c) {
                    continue;
                }
                const int64_t row = candidates[i].row;
                const float cx_in = at(det.data(), det_layout_, row, 0);
                const float cy_in = at(det.data(), det_layout_, row, 1);
                const float w_in = at(det.data(), det_layout_, row, 2);
                const float h_in = at(det.data(), det_layout_, row, 3);
                const float angle = at(det.data(), det_layout_, row, 4 + num_classes_);

                float cx = 0.0f, cy = 0.0f;
                affine_project(d2i, cx_in, cy_in, cx, cy);
                const float scale = d2i[0];
                shapes.emplace_back(cv::Point2f(cx, cy),
                                    cv::Size2f(std::max(1e-3f, w_in * scale),
                                               std::max(1e-3f, h_in * scale)),
                                    angle * 180.0f / static_cast<float>(CV_PI));
                scores.push_back(candidates[i].score);
                local.push_back(i);
            }
            if (shapes.empty()) {
                continue;
            }
            std::vector<int> keep;
            cv::dnn::NMSBoxes(shapes, scores, conf_, nms_, keep);
            for (int index : keep) {
                kept.push_back(local[index]);
            }
        }
    } else {
        kept = nms_by_class(candidates);
    }

    // 按分数降序，页面渲染与统计都依赖这个顺序。
    std::sort(kept.begin(), kept.end(), [&](int a, int b) {
        return candidates[a].score > candidates[b].score;
    });

    const std::vector<float>* proto = nullptr;
    if (task_ == OI_SEGMENT && proto_index_ >= 0 && static_cast<size_t>(proto_index_) < outputs.size()) {
        proto = &outputs[proto_index_];
        const int64_t need = static_cast<int64_t>(proto_dim_) * proto_h_ * proto_w_;
        if (static_cast<int64_t>(proto->size()) < need) {
            err = "分割原型张量元素数不足，期望 " + std::to_string(need) + "，实际 " +
                  std::to_string(proto->size()) + "。";
            return false;
        }
    }

    boxes_.reserve(kept.size());
    masks_.reserve(kept.size());
    kpts_.reserve(kept.size());

    for (int index : kept) {
        const RawBox& raw = candidates[index];
        oi_box box{};
        std::memset(&box, 0, sizeof(box));
        box.left = raw.left;
        box.top = raw.top;
        box.right = raw.right;
        box.bottom = raw.bottom;
        box.confidence = raw.score;
        box.class_label = raw.label;

        if (task_ == OI_POSE) {
            box.num_kpts = num_keypoints_;
            std::vector<float> points(static_cast<size_t>(num_keypoints_) * 3, 0.0f);
            for (int k = 0; k < num_keypoints_; ++k) {
                const float kx_in = at(det.data(), det_layout_, raw.row, 5 + k * 3 + 0);
                const float ky_in = at(det.data(), det_layout_, raw.row, 5 + k * 3 + 1);
                const float kv = at(det.data(), det_layout_, raw.row, 5 + k * 3 + 2);
                float kx = 0.0f, ky = 0.0f;
                affine_project(d2i, kx_in, ky_in, kx, ky);
                points[static_cast<size_t>(k) * 3 + 0] = kx;
                points[static_cast<size_t>(k) * 3 + 1] = ky;
                points[static_cast<size_t>(k) * 3 + 2] = kv;
            }
            kpts_.push_back(std::move(points));
            box.kpts = kpts_.back().data();
        }

        if (task_ == OI_OBB) {
            const float cx_in = at(det.data(), det_layout_, raw.row, 0);
            const float cy_in = at(det.data(), det_layout_, raw.row, 1);
            const float w_in = at(det.data(), det_layout_, raw.row, 2);
            const float h_in = at(det.data(), det_layout_, raw.row, 3);
            float cx = 0.0f, cy = 0.0f;
            affine_project(d2i, cx_in, cy_in, cx, cy);
            box.is_obb = 1;
            box.cx = cx;
            box.cy = cy;
            box.ow = w_in * d2i[0];
            box.oh = h_in * d2i[0];
            box.angle = at(det.data(), det_layout_, raw.row, 4 + num_classes_);
        }

        if (task_ == OI_SEGMENT && proto) {
            // 框在原图坐标，先经 i2d 回到网络空间，再按比例落到原型分辨率。
            auto to_proto = [&](float x, float y, int& px, int& py) {
                float nx = 0.0f, ny = 0.0f;
                affine_project(i2d, x, y, nx, ny);
                px = clamp_int(static_cast<int>(std::floor(nx * proto_w_ / input_w_)), 0, proto_w_ - 1);
                py = clamp_int(static_cast<int>(std::floor(ny * proto_h_ / input_h_)), 0, proto_h_ - 1);
            };

            int px0 = 0, py0 = 0, px1 = 0, py1 = 0;
            to_proto(raw.left, raw.top, px0, py0);
            to_proto(raw.right, raw.bottom, px1, py1);
            if (px1 < px0) std::swap(px0, px1);
            if (py1 < py0) std::swap(py0, py1);

            const int roi_w = px1 - px0 + 1;
            const int roi_h = py1 - py0 + 1;

            cv::Mat roi(roi_h, roi_w, CV_32FC1);
            for (int y = 0; y < roi_h; ++y) {
                float* dst = roi.ptr<float>(y);
                const int64_t proto_y = py0 + y;
                for (int x = 0; x < roi_w; ++x) {
                    const int64_t proto_x = px0 + x;
                    float sum = 0.0f;
                    for (int c = 0; c < proto_dim_; ++c) {
                        const float coeff = at(det.data(), det_layout_, raw.row, mask_coeff_offset + c);
                        const float value =
                            (*proto)[(static_cast<size_t>(c) * proto_h_ + proto_y) * proto_w_ + proto_x];
                        sum += coeff * value;
                    }
                    dst[x] = sigmoid(sum);
                }
            }

            cv::Mat binary;
            cv::threshold(roi, binary, 0.5, 255.0, cv::THRESH_BINARY);
            binary.convertTo(binary, CV_8UC1);

            const int bx = clamp_int(static_cast<int>(std::floor(raw.left)), 0, std::max(0, src_w - 1));
            const int by = clamp_int(static_cast<int>(std::floor(raw.top)), 0, std::max(0, src_h - 1));
            int bw = static_cast<int>(std::ceil(raw.right)) - bx;
            int bh = static_cast<int>(std::ceil(raw.bottom)) - by;
            bw = std::max(1, std::min(bw, src_w - bx));
            bh = std::max(1, std::min(bh, src_h - by));

            cv::Mat resized;
            cv::resize(binary, resized, cv::Size(bw, bh), 0, 0, cv::INTER_NEAREST);

            std::vector<unsigned char> mask(static_cast<size_t>(bw) * bh, 0);
            for (int y = 0; y < bh; ++y) {
                std::memcpy(mask.data() + static_cast<size_t>(y) * bw, resized.ptr<unsigned char>(y),
                            static_cast<size_t>(bw));
            }

            masks_.push_back(std::move(mask));
            box.has_mask = 1;
            box.mask_x = bx;
            box.mask_y = by;
            box.mask_w = bw;
            box.mask_h = bh;
            box.mask_data = masks_.back().data();
        }

        boxes_.push_back(box);
    }

    return true;
}

}  // namespace oi
