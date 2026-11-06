/*
 * Anomalib ONNX 推理。
 *
 * metadata.json 的字段名与 fallback 规则（pred_scores / anomaly_maps / transform）
 * 必须与 anomalib_infer 的解析口径一致，否则同一个模型在两条后端上阈值不同。
 * 这里刻意保留同样的优先级顺序：运行时覆盖 > metadata 显式字段 > 嵌套对象 > 顶层 min/max。
 */

#include "anomalib_onnx.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "anomalib_post.hpp"

namespace oi {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

struct Metadata {
    std::string modelType = "Auto";
    std::string normalization = "none";
    std::string outputContract;
    float imageThreshold = 0.5f;
    float pixelThreshold = 0.5f;
    float scoreMin = 0.0f;
    float scoreMax = 1.0f;
    float mapMin = 0.0f;
    float mapMax = 1.0f;
    int inputH = 256;
    int inputW = 256;
};

std::string lower_copy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

std::string regex_escape(const std::string& text) {
    static const std::regex special(R"([-[\]{}()*+?.,\^$|#\s])");
    return std::regex_replace(text, special, R"(\$&)");
}

bool json_number(const std::string& json, const std::string& key, float& out) {
    const std::regex re("\"" + regex_escape(key) +
                        R"("\s*:\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))");
    std::smatch match;
    if (!std::regex_search(json, match, re)) return false;
    out = std::stof(match[1].str());
    return true;
}

bool json_string(const std::string& json, const std::string& key, std::string& out) {
    const std::regex re("\"" + regex_escape(key) + R"json("\s*:\s*"([^"]*)")json");
    std::smatch match;
    if (!std::regex_search(json, match, re)) return false;
    out = match[1].str();
    return true;
}

bool json_array2_ints(const std::string& json, const std::string& key, int& a, int& b) {
    const std::regex re("\"" + regex_escape(key) + R"("\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+))");
    std::smatch match;
    if (!std::regex_search(json, match, re)) return false;
    a = std::stoi(match[1].str());
    b = std::stoi(match[2].str());
    return true;
}

/* 从 JSON 中截出某个对象的原文，用于读 pred_scores / anomaly_maps 里的 min/max。 */
bool extract_object(const std::string& json, const std::string& key, std::string& out) {
    const std::regex key_re("\"" + regex_escape(key) + R"("\s*:\s*\{)");
    std::smatch match;
    if (!std::regex_search(json, match, key_re)) return false;

    const size_t open = static_cast<size_t>(match.position(0) + match.length(0) - 1);
    int depth = 0;
    bool in_string = false;
    bool escape = false;

    for (size_t i = open; i < json.size(); ++i) {
        const char c = json[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) {
                out = json.substr(open, i - open + 1);
                return true;
            }
        }
    }
    return false;
}

bool json_nested_number(const std::string& json, const std::string& parent, const std::string& key,
                        float& out) {
    std::string object;
    return extract_object(json, parent, object) && json_number(object, key, out);
}

bool find_transform_size(const std::string& json, int& h, int& w) {
    std::string transform;
    if (extract_object(json, "transform", transform)) {
        float height = 0.0f;
        float width = 0.0f;
        if (json_number(transform, "height", height) && json_number(transform, "width", width)) {
            h = static_cast<int>(height);
            w = static_cast<int>(width);
            return true;
        }
    }
    return false;
}

bool read_text_file(const char* utf8_path, std::string& out, std::string& err) {
    if (!utf8_path || !*utf8_path) {
        err = "metadata 路径为空。";
        return false;
    }
    std::ifstream file(std::filesystem::u8path(utf8_path), std::ios::binary);
    if (!file) {
        err = std::string("无法读取 metadata：") + utf8_path;
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    out = buffer.str();
    if (out.empty()) {
        err = std::string("metadata 文件为空：") + utf8_path;
        return false;
    }
    return true;
}

bool load_metadata(const char* path, const std::string& model_type_hint, Metadata& meta,
                   std::string& err) {
    std::string json;
    if (!read_text_file(path, json, err)) return false;

    if (!json_string(json, "model_type", meta.modelType) || meta.modelType.empty()) {
        meta.modelType = model_type_hint.empty() ? "Auto" : model_type_hint;
    }

    json_string(json, "normalization", meta.normalization);
    meta.normalization = lower_copy(meta.normalization);
    json_string(json, "output_contract", meta.outputContract);
    meta.outputContract = lower_copy(meta.outputContract);

    if (meta.normalization.empty()) {
        if (meta.outputContract == "anomalib_post_processed_0_1") {
            meta.normalization = "none";
        } else {
            meta.normalization =
                lower_copy(meta.modelType).find("efficient") != std::string::npos ? "none" : "imagenet";
        }
    }

    const bool ok_image = json_number(json, "image_threshold", meta.imageThreshold);
    const bool ok_pixel = json_number(json, "pixel_threshold", meta.pixelThreshold);
    bool ok_score_min = json_nested_number(json, "pred_scores", "min", meta.scoreMin);
    bool ok_score_max = json_nested_number(json, "pred_scores", "max", meta.scoreMax);
    bool ok_map_min = json_nested_number(json, "anomaly_maps", "min", meta.mapMin);
    bool ok_map_max = json_nested_number(json, "anomaly_maps", "max", meta.mapMax);
    if (!ok_score_min) ok_score_min = json_number(json, "pred_scores.min", meta.scoreMin);
    if (!ok_score_max) ok_score_max = json_number(json, "pred_scores.max", meta.scoreMax);
    if (!ok_map_min) ok_map_min = json_number(json, "anomaly_maps.min", meta.mapMin);
    if (!ok_map_max) ok_map_max = json_number(json, "anomaly_maps.max", meta.mapMax);

    float flat_min = 0.0f;
    float flat_max = 1.0f;
    const bool ok_flat_min = json_number(json, "min", flat_min);
    const bool ok_flat_max = json_number(json, "max", flat_max);
    if (!ok_score_min && ok_flat_min) meta.scoreMin = flat_min;
    if (!ok_score_max && ok_flat_max) meta.scoreMax = flat_max;
    if (!ok_map_min && ok_flat_min) meta.mapMin = flat_min;
    if (!ok_map_max && ok_flat_max) meta.mapMax = flat_max;

    if (!find_transform_size(json, meta.inputH, meta.inputW)) {
        int h = 0;
        int w = 0;
        if (json_array2_ints(json, "input_size", h, w) || json_array2_ints(json, "image_size", h, w)) {
            meta.inputH = h;
            meta.inputW = w;
        }
    }

    if (!ok_image || !ok_pixel) {
        err = "metadata 缺少 image_threshold 或 pixel_threshold 字段。";
        return false;
    }
    if (meta.inputH <= 0 || meta.inputW <= 0) {
        err = "metadata 的输入尺寸无效。";
        return false;
    }
    return true;
}

void assign_mat_bytes(const cv::Mat& mat, std::vector<unsigned char>& out) {
    cv::Mat compact = mat.isContinuous() ? mat : mat.clone();
    out.assign(compact.data, compact.data + compact.total() * compact.elemSize());
}

}  // namespace

bool AnomalibOnnx::open(const std::wstring& model_path, const char* metadata_path_utf8,
                        const char* model_type_utf8, float image_threshold_override,
                        float pixel_threshold_override, std::string& err) {
    Metadata meta;
    if (!load_metadata(metadata_path_utf8, model_type_utf8 ? model_type_utf8 : "Auto", meta, err)) {
        return false;
    }

    model_type_ = meta.modelType;
    normalization_ = meta.normalization;
    image_threshold_ = meta.imageThreshold;
    pixel_threshold_ = meta.pixelThreshold;
    score_min_ = meta.scoreMin;
    score_max_ = meta.scoreMax;
    map_min_ = meta.mapMin;
    map_max_ = meta.mapMax;
    input_h_ = meta.inputH;
    input_w_ = meta.inputW;

    // 与 ai_create 相同：>=0 覆盖，<0 沿用 metadata。
    if (image_threshold_override >= 0.0f) image_threshold_ = image_threshold_override;
    if (pixel_threshold_override >= 0.0f) pixel_threshold_ = pixel_threshold_override;

    if (!session_.open(model_path, err)) {
        return false;
    }

    const auto& inputs = session_.inputs();
    if (inputs.size() != 1) {
        err = "Anomalib ONNX 应只有 1 个输入，实际有 " + std::to_string(inputs.size()) + " 个。";
        return false;
    }
    const TensorInfo& input = inputs[0];
    if (input.shape.size() != 4) {
        err = "Anomalib ONNX 输入应为 4 维 NCHW。";
        return false;
    }
    if (input.shape[1] != 3) {
        err = "Anomalib ONNX 输入通道维应为 3，实际第 2 维为 " + std::to_string(input.shape[1]) + "。";
        return false;
    }

    // 宽高可能是动态的，此时以 metadata 的 input_size 为准。
    int height = input.shape[2] > 0 ? static_cast<int>(input.shape[2]) : input_h_;
    int width = input.shape[3] > 0 ? static_cast<int>(input.shape[3]) : input_w_;
    if (height <= 0 || width <= 0) {
        err = "无法确定 Anomalib 的输入尺寸：模型声明为动态且 metadata 未给出有效 input_size。";
        return false;
    }
    input_h_ = height;
    input_w_ = width;

    std::vector<int64_t> dims = input.shape;
    dims[0] = 1;
    dims[1] = 3;
    dims[2] = height;
    dims[3] = width;
    if (!session_.set_input_dimensions(0, dims, err)) {
        return false;
    }

    if (session_.outputs().empty()) {
        err = "Anomalib ONNX 没有输出张量。";
        return false;
    }
    return true;
}

bool AnomalibOnnx::infer(const unsigned char* rgb, int width, int height, int stride,
                         oi_frame_stat* stat, std::string& err) {
    if (!rgb || width <= 0 || height <= 0) {
        err = "输入图像无效。";
        return false;
    }
    if (stride <= 0) stride = width * 3;
    if (stride < width * 3) {
        err = "输入图像的 stride 小于 width*3，数据长度不足。";
        return false;
    }

    const Clock::time_point t0 = Clock::now();

    cv::Mat rgb_view(height, width, CV_8UC3, const_cast<unsigned char*>(rgb), static_cast<size_t>(stride));
    cv::Mat rgb_mat = rgb_view.clone();

    cv::Mat resized;
    cv::resize(rgb_mat, resized, cv::Size(input_w_, input_h_), 0, 0, cv::INTER_LINEAR);
    cv::Mat floats;
    resized.convertTo(floats, CV_32FC3, 1.0 / 255.0);
    if (normalization_ == "imagenet") {
        const cv::Scalar mean(0.485, 0.456, 0.406);
        const cv::Scalar stddev(0.229, 0.224, 0.225);
        cv::subtract(floats, mean, floats);
        cv::divide(floats, stddev, floats);
    }

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

    // ---- 选择分数与异常图输出 -------------------------------------------------
    const auto& infos = session_.outputs();
    int score_index = -1;
    int map_index = -1;

    for (size_t i = 0; i < outputs.size(); ++i) {
        const std::string name = lower_copy(i < infos.size() ? infos[i].name : std::string());
        if (name.find("score") != std::string::npos) {
            score_index = static_cast<int>(i);
        }
        if (name.find("anomaly_map") != std::string::npos ||
            (name.find("map") != std::string::npos && name.find("mask") == std::string::npos)) {
            map_index = static_cast<int>(i);
        }
        // 元素数最大的那个必定是异常图：单元素张量不可能是逐像素图。
        if (map_index < 0 || outputs[i].size() > outputs[map_index].size()) {
            map_index = static_cast<int>(i);
        }
        if (outputs[i].size() == 1 && score_index < 0 && name.find("label") == std::string::npos) {
            score_index = static_cast<int>(i);
        }
    }

    if (map_index < 0) map_index = 0;
    if (score_index == map_index) score_index = -1;

    const std::vector<float>& map_data = outputs[map_index];

    int map_h = input_h_;
    int map_w = input_w_;
    if (map_index < static_cast<int>(infos.size()) && infos[map_index].shape.size() >= 2) {
        const auto& shape = infos[map_index].shape;
        map_h = static_cast<int>(shape[shape.size() - 2]);
        map_w = static_cast<int>(shape[shape.size() - 1]);
    }
    if (map_h <= 0 || map_w <= 0 ||
        static_cast<size_t>(map_h) * map_w > map_data.size()) {
        const int approx = static_cast<int>(std::sqrt(static_cast<double>(map_data.size())));
        if (approx <= 0) {
            err = "无法解析异常图输出尺寸。";
            return false;
        }
        map_h = approx;
        map_w = approx;
    }
    if (static_cast<size_t>(map_h) * map_w > map_data.size()) {
        err = "异常图尺寸与输出元素数不匹配。";
        return false;
    }

    cv::Mat raw_map(map_h, map_w, CV_32FC1, const_cast<float*>(map_data.data()));

    float raw_score = 0.0f;
    if (score_index >= 0 && !outputs[score_index].empty()) {
        raw_score = outputs[score_index][0];
    } else {
        double max_value = 0.0;
        cv::minMaxLoc(raw_map, nullptr, &max_value);
        raw_score = static_cast<float>(max_value);
    }

    anomalib_post::RenderConfig config;
    config.imageThreshold = image_threshold_;
    config.pixelThreshold = pixel_threshold_;
    config.mapMin = map_min_;
    config.mapMax = map_max_;
    config.scoreMin = score_min_;
    config.scoreMax = score_max_;
    const anomalib_post::RenderOutput rendered =
        anomalib_post::render(rgb_mat, raw_map, raw_score, config);

    cv::Mat map_compact = rendered.normalized_map.isContinuous() ? rendered.normalized_map
                                                                : rendered.normalized_map.clone();
    const auto* map_ptr = reinterpret_cast<const float*>(map_compact.data);
    normalized_map_.assign(map_ptr, map_ptr + map_compact.total());

    assign_mat_bytes(rgb_mat, source_rgb_);
    assign_mat_bytes(rendered.overlay_rgb, overlay_rgb_);
    assign_mat_bytes(rendered.mask_rgb, mask_rgb_);

    result_ = oi_anomaly_result{};
    result_.source_rgb = source_rgb_.data();
    result_.source_w = width;
    result_.source_h = height;
    result_.source_stride = width * 3;
    result_.overlay_rgb = overlay_rgb_.data();
    result_.overlay_w = width;
    result_.overlay_h = height;
    result_.overlay_stride = width * 3;
    result_.mask_rgb = mask_rgb_.data();
    result_.mask_w = width;
    result_.mask_h = height;
    result_.mask_stride = width * 3;
    result_.normalized_map = normalized_map_.data();
    result_.map_w = map_w;
    result_.map_h = map_h;
    result_.raw_score = raw_score;
    result_.normalized_score = rendered.normalized_score;
    result_.is_anomaly = rendered.is_anomaly ? 1 : 0;

    const Clock::time_point t3 = Clock::now();

    if (stat) {
        stat->src_w = width;
        stat->src_h = height;
        stat->count = 1;
        stat->preprocess_ms = elapsed_ms(t0, t1);
        stat->inference_ms = elapsed_ms(t1, t2);
        stat->postprocess_ms = elapsed_ms(t2, t3);
        stat->total_ms = elapsed_ms(t0, t3);
    }
    return true;
}

}  // namespace oi
