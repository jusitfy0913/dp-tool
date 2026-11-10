/*
 * 整图分类的 ONNX 推理实现。
 *
 * 预处理是这一层唯一的"容易出错"的地方：它必须与 Python 侧
 * `classifyinfer.preprocess` 逐行等价，否则同一个模型在 DLL 路径和
 * onnxruntime 降级路径上会给出不同的分数，而页面上看不出是哪一边错了。
 * 因此下面每一步都标了对应的 Python 语句。
 */

#include "classify_onnx.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace oi {
namespace {

using Clock = std::chrono::steady_clock;

/* 分类输入在形状信息缺失时的默认边长，与 classifyinfer._input_hw 的 224 一致。 */
constexpr int kDefaultSide = 224;

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

/*
 * Python 的 round() 是「四舍六入五成双」，std::lround 是「五入」。
 * 中心裁剪的偏移正好可能落在 x.5 上（边长差为奇数时），两个取整口径会差一个像素，
 * 而差一个像素就足以让分类分数偏移。这里显式复刻 Python 的行为。
 */
int py_round(double value) {
    const double base = std::floor(value);
    const double frac = value - base;
    const int low = static_cast<int>(base);
    if (frac > 0.5) return low + 1;
    if (frac < 0.5) return low;
    return (low % 2 == 0) ? low : low + 1;  // 正好 .5：取偶数，与 Python 一致
}

}  // namespace

bool ClassifyOnnx::open(const std::wstring& model_path, std::string& err) {
    if (!session_.open(model_path, err)) {
        return false;
    }

    const auto& inputs = session_.inputs();
    if (inputs.size() != 1) {
        err = "分类 ONNX 应只有 1 个输入，实际有 " + std::to_string(inputs.size()) + " 个。";
        return false;
    }

    const TensorInfo& input = inputs[0];
    if (input.shape.size() != 4) {
        err = "分类 ONNX 输入应为 4 维 NCHW，实际形状 " + shape_text(input.shape) + "。";
        return false;
    }
    if (input.shape[1] != 3) {
        err = "分类 ONNX 输入的通道维应为 3（NCHW），实际形状 " + shape_text(input.shape) + "。";
        return false;
    }
    if (input.shape[0] > 0 && input.shape[0] != 1) {
        err = "分类 ONNX 的 batch 维固定为 " + std::to_string(input.shape[0]) +
              "，本 DLL 一次只处理一张图，请按 batch=1 重新导出。";
        return false;
    }

    // 动态维按 224 处理，与 Python 侧 `_input_hw` 的默认值一致；
    // 静态维直接用模型声明值，不做任何猜测。
    input_h_ = input.shape[2] > 0 ? static_cast<int>(input.shape[2]) : kDefaultSide;
    input_w_ = input.shape[3] > 0 ? static_cast<int>(input.shape[3]) : kDefaultSide;
    if (input_h_ <= 0 || input_w_ <= 0) {
        err = "分类 ONNX 的输入尺寸无法确定：" + shape_text(input.shape) + "。";
        return false;
    }

    // 把所有动态维固定下来：Session::run 需要具体形状才能造 OrtValue。
    if (!session_.set_input_dimensions(0, {1, 3, input_h_, input_w_}, err)) {
        return false;
    }

    // 输出：常见是 [1, C] 或 [1, C, 1, 1]，去掉 batch 后各维乘积就是类别数。
    const auto& outputs = session_.outputs();
    if (outputs.empty()) {
        err = "分类 ONNX 没有输出。";
        return false;
    }
    std::vector<int64_t> dims = outputs[0].shape;
    if (!dims.empty() && dims[0] == 1) {
        dims.erase(dims.begin());
    }
    int64_t product = 1;
    bool dynamic = false;
    for (int64_t dim : dims) {
        if (dim <= 0) {
            dynamic = true;
            break;
        }
        product *= dim;
    }
    // 全静态时先把类别数定下来，动态输出留到第一次推理再确定。
    if (!dynamic && product > 0 && product <= 100000) {
        num_classes_ = static_cast<int>(product);
    }

    // 模型自带的类别名：Ultralytics 会写进 custom metadata，顺序就是输出顺序。
    // 取不到就留空，由调用方退回标签文件。
    names_ = session_.custom_metadata("names");
    return true;
}

bool ClassifyOnnx::preprocess(const unsigned char* bgr, int width, int height, int stride,
                              std::vector<float>& tensor, std::string& err) const {
    const cv::Mat src(height, width, CV_8UC3, const_cast<unsigned char*>(bgr),
                      static_cast<size_t>(stride));

    cv::Mat cropped;
    if (input_h_ == input_w_) {
        // 对应 classifyinfer.preprocess 的 `if size_height == size_width` 分支。
        const int target = input_h_;
        const int short_side = std::min(width, height);
        cv::Mat resized;
        if (short_side != target) {
            const double scale = static_cast<double>(target) / static_cast<double>(short_side);
            const int new_w = std::max(static_cast<int>(width * scale), target);
            const int new_h = std::max(static_cast<int>(height * scale), target);
            cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        } else {
            resized = src;
        }

        const int top = py_round((resized.rows - target) / 2.0);
        const int left = py_round((resized.cols - target) / 2.0);
        const int crop_w = std::min(target, resized.cols - left);
        const int crop_h = std::min(target, resized.rows - top);
        if (left < 0 || top < 0 || crop_w <= 0 || crop_h <= 0) {
            err = "中心裁剪区域超出缩放后的图像，分类中止。";
            return false;
        }
        cropped = resized(cv::Rect(left, top, crop_w, crop_h)).clone();
        if (crop_w != target || crop_h != target) {
            // Python 切片越界时不会补边，这里补一次 resize 保证输入尺寸正确。
            cv::resize(cropped, cropped, cv::Size(target, target), 0, 0, cv::INTER_LINEAR);
        }
    } else {
        // 非正方形输入：直接缩放，与 Python 的 else 分支一致。
        cv::resize(src, cropped, cv::Size(input_w_, input_h_), 0, 0, cv::INTER_LINEAR);
    }

    if (cropped.empty()) {
        err = "缩放后没有可用像素，分类中止。";
        return false;
    }

    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);

    cv::Mat floats;
    rgb.convertTo(floats, CV_32FC3, 1.0 / 255.0);

    const size_t plane_size = static_cast<size_t>(input_w_) * static_cast<size_t>(input_h_);
    tensor.assign(plane_size * 3, 0.0f);
    std::vector<cv::Mat> planes;
    cv::split(floats, planes);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(tensor.data() + c * plane_size, planes[c].ptr<float>(), plane_size * sizeof(float));
    }
    return true;
}

bool ClassifyOnnx::infer(const unsigned char* bgr, int width, int height, int stride,
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

    std::vector<float> tensor;
    if (!preprocess(bgr, width, height, stride, tensor, err)) {
        return false;
    }

    const Clock::time_point t1 = Clock::now();

    std::vector<std::vector<float>> outputs;
    if (!session_.run({tensor}, outputs, err)) {
        return false;
    }

    const Clock::time_point t2 = Clock::now();

    if (outputs.empty() || outputs[0].empty()) {
        err = "分类模型没有输出分数。";
        return false;
    }

    const std::vector<float>& raw = outputs[0];
    const int count = static_cast<int>(raw.size());
    if (num_classes_ == 0) {
        num_classes_ = count;
    } else if (num_classes_ != count) {
        err = "分类输出长度 " + std::to_string(count) + " 与创建句柄时推断的类别数 " +
              std::to_string(num_classes_) + " 不一致。";
        return false;
    }

    // 只挑最大值，不做 softmax：概率化由调用方统一处理。
    int best = 0;
    for (int i = 1; i < count; ++i) {
        if (raw[static_cast<size_t>(i)] > raw[static_cast<size_t>(best)]) {
            best = i;
        }
    }

    scores_.assign(raw.begin(), raw.end());
    result_.num_classes = count;
    result_.top1 = best;
    result_.top1_score = scores_[static_cast<size_t>(best)];
    result_.scores = scores_.data();
    result_.input_w = input_w_;
    result_.input_h = input_h_;
    result_.names_utf8 = names_.empty() ? nullptr : names_.c_str();

    const Clock::time_point t3 = Clock::now();

    if (stat) {
        stat->src_w = width;
        stat->src_h = height;
        stat->count = count;
        stat->preprocess_ms = elapsed_ms(t0, t1);
        stat->inference_ms = elapsed_ms(t1, t2);
        stat->postprocess_ms = elapsed_ms(t2, t3);
        stat->total_ms = elapsed_ms(t0, t3);
    }
    return true;
}

}  // namespace oi
