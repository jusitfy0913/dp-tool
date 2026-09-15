#include "ocr_infer_c.h"

#include "src/pipelines/ocr/pipeline.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

static thread_local std::string g_last_error;

namespace {

struct OcrLineStorage {
    oi_text_line view{};
    std::vector<float> xs;
    std::vector<float> ys;
    std::string text;
};

struct OcrContext {
    std::unique_ptr<_OCRPipeline> pipeline;
    std::mutex mutex;
    oi_result result{};
    std::vector<OcrLineStorage> line_storage;
    std::vector<oi_text_line> line_views;
    std::vector<unsigned char> overlay_rgb;
    std::string raw_json;
    std::string device;
    std::string precision;
    std::string run_mode;
    bool mkldnn_enabled = false;
    int cpu_threads = 8;
    int rec_batch_size = 6;
};

void set_error(const std::string& msg)
{
    g_last_error = msg;
}

std::string safe_str(const char* value)
{
    return value ? std::string(value) : std::string();
}

bool file_exists(const std::string& path)
{
    std::ifstream f(path.c_str(), std::ios::binary);
    return f.good();
}

void copy_cstr(char* dst, size_t dst_size, const std::string& value)
{
    if (!dst || dst_size == 0) return;
    const size_t n = std::min(dst_size - 1, value.size());
    std::copy(value.data(), value.data() + n, dst);
    dst[n] = '\0';
}

std::string resolve_run_mode(const std::string& device,
                             const std::string& precision,
                             bool enable_mkldnn)
{
    if (device.rfind("gpu", 0) == 0) {
        return precision == "fp16" ? "paddle_fp16" : "paddle";
    }
    return enable_mkldnn ? "mkldnn" : "paddle";
}

std::string join_path(const std::string& dir, const std::string& name)
{
    if (dir.empty()) return name;
    const char last = dir.back();
    if (last == '/' || last == '\\') return dir + name;
    return dir + "\\" + name;
}

std::string read_global_model_name(const std::string& dir)
{
    const std::string yml_path = join_path(dir, "inference.yml");
    std::ifstream in(yml_path.c_str());
    if (!in.good()) {
        return {};
    }
    std::string line;
    bool in_global = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line == "Global:") {
            in_global = true;
            continue;
        }
        if (!in_global) {
            continue;
        }
        if (!line.empty() && line[0] != ' ' && line[0] != '\t') {
            break;
        }
        const std::string key = "model_name:";
        const auto pos = line.find(key);
        if (pos == std::string::npos) {
            continue;
        }
        std::string value = line.substr(pos + key.size());
        const auto vstart = value.find_first_not_of(" \t");
        if (vstart != std::string::npos) {
            value = value.substr(vstart);
        }
        if (!value.empty() && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.size() - 2);
        }
        return value;
    }
    return {};
}

bool validate_model_dir(const std::string& dir, const char* label, std::string& error)
{
    if (dir.empty()) {
        error = std::string(label) + " model dir is empty";
        return false;
    }
    const bool has_model = file_exists(join_path(dir, "inference.pdmodel")) ||
                           file_exists(join_path(dir, "inference.json"));
    if (!has_model) {
        error = std::string(label) + " model dir missing inference.pdmodel or inference.json: " + dir;
        return false;
    }
    if (!file_exists(join_path(dir, "inference.pdiparams"))) {
        error = std::string(label) + " model dir missing inference.pdiparams: " + dir;
        return false;
    }
    if (!file_exists(join_path(dir, "inference.yml")) &&
        !file_exists(join_path(dir, "inference.yaml"))) {
        error = std::string(label) + " model dir missing inference.yml or inference.yaml: " + dir;
        return false;
    }
    return true;
}

std::string json_escape(const std::string& value)
{
    std::ostringstream os;
    for (char c : value) {
        switch (c) {
        case '\\': os << "\\\\"; break;
        case '"': os << "\\\""; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                os << "\\u" << std::hex << static_cast<int>(c);
            } else {
                os << c;
            }
        }
    }
    return os.str();
}

std::string build_raw_json(const OCRPipelineResult& r)
{
    std::ostringstream os;
    os << "{";
    os << "\"rec_texts\":[";
    for (size_t i = 0; i < r.rec_texts.size(); ++i) {
        if (i) os << ",";
        os << "\"" << json_escape(r.rec_texts[i]) << "\"";
    }
    os << "],\"rec_scores\":[";
    for (size_t i = 0; i < r.rec_scores.size(); ++i) {
        if (i) os << ",";
        os << r.rec_scores[i];
    }
    os << "],\"textline_orientation_angles\":[";
    for (size_t i = 0; i < r.textline_orientation_angles.size(); ++i) {
        if (i) os << ",";
        os << r.textline_orientation_angles[i];
    }
    os << "],\"rec_polys\":[";
    for (size_t i = 0; i < r.rec_polys.size(); ++i) {
        if (i) os << ",";
        os << "[";
        for (size_t j = 0; j < r.rec_polys[i].size(); ++j) {
            if (j) os << ",";
            os << "[" << r.rec_polys[i][j].x << "," << r.rec_polys[i][j].y << "]";
        }
        os << "]";
    }
    os << "]}";
    return os.str();
}

void draw_overlay(const OCRPipelineResult& r, OcrContext* ctx)
{
    cv::Mat source = r.doc_preprocessor_res.output_image.empty()
                         ? r.doc_preprocessor_res.input_image
                         : r.doc_preprocessor_res.output_image;
    if (source.empty()) {
        ctx->overlay_rgb.clear();
        ctx->result.overlay_rgb = nullptr;
        ctx->result.overlay_w = 0;
        ctx->result.overlay_h = 0;
        ctx->result.overlay_stride = 0;
        return;
    }

    cv::Mat overlay = source.clone();
    for (size_t i = 0; i < r.rec_polys.size(); ++i) {
        std::vector<cv::Point> pts;
        pts.reserve(r.rec_polys[i].size());
        for (const auto& p : r.rec_polys[i]) {
            pts.emplace_back(cvRound(p.x), cvRound(p.y));
        }
        if (pts.size() >= 2) {
            const cv::Scalar color(0, 200, 255);
            cv::polylines(overlay, pts, true, color, 2, cv::LINE_AA);
            cv::putText(overlay, std::to_string(i + 1), pts.front(),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv::LINE_AA);
        }
    }

    cv::Mat rgb;
    cv::cvtColor(overlay, rgb, cv::COLOR_BGR2RGB);
    ctx->overlay_rgb.assign(rgb.data, rgb.data + rgb.total() * rgb.elemSize());
    ctx->result.overlay_rgb = ctx->overlay_rgb.empty() ? nullptr : ctx->overlay_rgb.data();
    ctx->result.overlay_w = rgb.cols;
    ctx->result.overlay_h = rgb.rows;
    ctx->result.overlay_stride = static_cast<int>(rgb.step);
}

int fill_result(OcrContext* ctx, const OCRPipelineResult& r, oi_frame_stat* out_stat,
                double io_ms, double pipeline_ms, double result_ms, double total_ms)
{
    ctx->line_storage.clear();
    ctx->line_views.clear();
    ctx->raw_json = build_raw_json(r);

    const size_t n = std::min(r.rec_texts.size(),
                              std::min(r.rec_scores.size(), r.rec_polys.size()));
    ctx->line_storage.reserve(n);
    ctx->line_views.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        OcrLineStorage line;
        line.view.index = static_cast<int>(i);
        line.view.point_count = static_cast<int>(r.rec_polys[i].size());
        line.view.rec_score = r.rec_scores[i];
        line.view.textline_angle =
            i < r.textline_orientation_angles.size() ? r.textline_orientation_angles[i] : -1;
        line.text = r.rec_texts[i];
        line.xs.reserve(r.rec_polys[i].size());
        line.ys.reserve(r.rec_polys[i].size());
        for (const auto& p : r.rec_polys[i]) {
            line.xs.push_back(p.x);
            line.ys.push_back(p.y);
        }
        line.view.xs = line.xs.empty() ? nullptr : line.xs.data();
        line.view.ys = line.ys.empty() ? nullptr : line.ys.data();
        line.view.text_utf8 = line.text.c_str();
        ctx->line_storage.push_back(std::move(line));
    }
    for (auto& line : ctx->line_storage) {
        line.view.xs = line.xs.empty() ? nullptr : line.xs.data();
        line.view.ys = line.ys.empty() ? nullptr : line.ys.data();
        line.view.text_utf8 = line.text.c_str();
        ctx->line_views.push_back(line.view);
    }

    draw_overlay(r, ctx);

    const cv::Mat source = r.doc_preprocessor_res.output_image.empty()
                               ? r.doc_preprocessor_res.input_image
                               : r.doc_preprocessor_res.output_image;
    ctx->result.src_w = source.cols;
    ctx->result.src_h = source.rows;
    ctx->result.line_count = static_cast<int>(ctx->line_views.size());
    ctx->result.lines = ctx->line_views.empty() ? nullptr : ctx->line_views.data();
    ctx->result.raw_json_utf8 = ctx->raw_json.c_str();

    if (out_stat) {
        out_stat->src_w = ctx->result.src_w;
        out_stat->src_h = ctx->result.src_h;
        out_stat->count = ctx->result.line_count;
        out_stat->preprocess_ms = io_ms;
        out_stat->inference_ms = pipeline_ms;
        out_stat->postprocess_ms = result_ms;
        out_stat->total_ms = total_ms;
        out_stat->io_ms = io_ms;
        out_stat->pipeline_ms = pipeline_ms;
        out_stat->result_ms = result_ms;
        out_stat->mkldnn_enabled = ctx->mkldnn_enabled ? 1 : 0;
        out_stat->cpu_threads = ctx->cpu_threads;
        out_stat->rec_batch_size = ctx->rec_batch_size;
        copy_cstr(out_stat->device_utf8, sizeof(out_stat->device_utf8), ctx->device);
        copy_cstr(out_stat->precision_utf8, sizeof(out_stat->precision_utf8), ctx->precision);
        copy_cstr(out_stat->run_mode_utf8, sizeof(out_stat->run_mode_utf8), ctx->run_mode);
    }
    return OI_OK;
}

OCRPipelineParams to_pipeline_params(const oi_config* config)
{
    OCRPipelineParams params;
    const std::string det_dir = safe_str(config->det_model_dir_utf8);
    const std::string rec_dir = safe_str(config->rec_model_dir_utf8);
    params.text_detection_model_dir = det_dir;
    params.text_recognition_model_dir = rec_dir;
    if (const std::string det_name = read_global_model_name(det_dir); !det_name.empty()) {
        params.text_detection_model_name = det_name;
    }
    if (const std::string rec_name = read_global_model_name(rec_dir); !rec_name.empty()) {
        params.text_recognition_model_name = rec_name;
    }
    params.text_det_thresh = config->det_thresh > 0 ? config->det_thresh : 0.3f;
    params.text_det_box_thresh = config->det_box_thresh > 0 ? config->det_box_thresh : 0.6f;
    params.text_det_unclip_ratio = config->det_unclip_ratio > 0 ? config->det_unclip_ratio : 1.5f;
    params.text_rec_score_thresh = config->rec_score_thresh >= 0 ? config->rec_score_thresh : 0.0f;
    params.text_recognition_batch_size = config->rec_batch_size > 0 ? config->rec_batch_size : 6;
    params.device = safe_str(config->device_utf8).empty() ? "cpu" : safe_str(config->device_utf8);
    params.precision = safe_str(config->precision_utf8).empty() ? "fp32" : safe_str(config->precision_utf8);
    params.enable_mkldnn = config->enable_mkldnn != 0;
    params.cpu_threads = config->cpu_threads > 0 ? config->cpu_threads : 8;
    params.thread_num = 1;

    const std::string textline_dir = safe_str(config->textline_orientation_model_dir_utf8);
    const bool use_textline =
        config->use_textline_orientation != 0 && !textline_dir.empty();
    params.use_textline_orientation = use_textline;
    if (use_textline) {
        params.textline_orientation_model_dir = textline_dir;
        if (const std::string line_name = read_global_model_name(textline_dir);
            !line_name.empty()) {
            params.textline_orientation_model_name = line_name;
        }
    }

    const std::string doc_orient_dir = safe_str(config->doc_orientation_model_dir_utf8);
    const bool use_doc_orient =
        config->use_doc_orientation_classify != 0 && !doc_orient_dir.empty();
    params.use_doc_orientation_classify = use_doc_orient;
    if (use_doc_orient) {
        params.doc_orientation_classify_model_dir = doc_orient_dir;
        if (const std::string name = read_global_model_name(doc_orient_dir); !name.empty()) {
            params.doc_orientation_classify_model_name = name;
        }
    }

    const std::string doc_unwarp_dir = safe_str(config->doc_unwarping_model_dir_utf8);
    const bool use_doc_unwarp = config->use_doc_unwarping != 0 && !doc_unwarp_dir.empty();
    params.use_doc_unwarping = use_doc_unwarp;
    if (use_doc_unwarp) {
        params.doc_unwarping_model_dir = doc_unwarp_dir;
        if (const std::string name = read_global_model_name(doc_unwarp_dir); !name.empty()) {
            params.doc_unwarping_model_name = name;
        }
    }
    return params;
}

int run_file(OcrContext* ctx, const char* image_path_utf8,
             const oi_result** out_result, oi_frame_stat* out_stat)
{
    if (!ctx || !ctx->pipeline) {
        set_error("OCR handle is not initialized");
        return OI_ERR_INVALID_ARGUMENT;
    }
    const std::string path = safe_str(image_path_utf8);
    if (path.empty()) {
        set_error("image path is empty");
        return OI_ERR_INVALID_ARGUMENT;
    }
    if (!file_exists(path)) {
        set_error("image file not found: " + path);
        return OI_ERR_INVALID_ARGUMENT;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        ctx->pipeline->Predict(std::vector<std::string>{path});
    } catch (const std::exception& e) {
        set_error(std::string("OCR inference exception: ") + e.what());
        return OI_ERR_INFERENCE;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    const auto results = ctx->pipeline->PipelineResult();
    if (results.empty()) {
        set_error("OCR pipeline returned no result");
        return OI_ERR_INFERENCE;
    }
    const double pipeline_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const int rc = fill_result(ctx, results.front(), out_stat,
                               0.0, pipeline_ms, 0.0, pipeline_ms);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (out_stat) {
        out_stat->result_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        out_stat->postprocess_ms = out_stat->result_ms;
        out_stat->total_ms = pipeline_ms + out_stat->result_ms;
    }
    if (out_result) *out_result = &ctx->result;
    return rc;
}

std::string temp_image_path()
{
#ifdef _WIN32
    char temp_dir[MAX_PATH] = {0};
    GetTempPathA(MAX_PATH, temp_dir);
    char temp_file[MAX_PATH] = {0};
    GetTempFileNameA(temp_dir, "oi_", 0, temp_file);
    std::string path = temp_file;
    std::remove(path.c_str());
    return path + ".png";
#else
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return "/tmp/oi_" + std::to_string(now) + ".png";
#endif
}

} // namespace

extern "C" {

OI_API oi_handle oi_create(const oi_config* config)
{
    if (!config) {
        set_error("null OCR config");
        return nullptr;
    }

    std::string error;
    const std::string det_dir = safe_str(config->det_model_dir_utf8);
    const std::string rec_dir = safe_str(config->rec_model_dir_utf8);
    if (!validate_model_dir(det_dir, "detection", error) ||
        !validate_model_dir(rec_dir, "recognition", error)) {
        set_error(error);
        return nullptr;
    }
    if (config->use_textline_orientation &&
        !validate_model_dir(safe_str(config->textline_orientation_model_dir_utf8),
                            "textline orientation", error)) {
        set_error(error);
        return nullptr;
    }
    if (config->use_doc_orientation_classify &&
        !validate_model_dir(safe_str(config->doc_orientation_model_dir_utf8),
                            "document orientation", error)) {
        set_error(error);
        return nullptr;
    }
    if (config->use_doc_unwarping &&
        !validate_model_dir(safe_str(config->doc_unwarping_model_dir_utf8),
                            "document unwarping", error)) {
        set_error(error);
        return nullptr;
    }

    try {
        auto* ctx = new OcrContext();
        ctx->device = safe_str(config->device_utf8).empty() ? "cpu" : safe_str(config->device_utf8);
        ctx->precision = safe_str(config->precision_utf8).empty() ? "fp32" : safe_str(config->precision_utf8);
        ctx->mkldnn_enabled = config->enable_mkldnn != 0 && ctx->device.rfind("cpu", 0) == 0;
        ctx->run_mode = resolve_run_mode(ctx->device, ctx->precision, config->enable_mkldnn != 0);
        ctx->cpu_threads = config->cpu_threads > 0 ? config->cpu_threads : 8;
        ctx->rec_batch_size = config->rec_batch_size > 0 ? config->rec_batch_size : 6;
        ctx->pipeline.reset(new _OCRPipeline(to_pipeline_params(config)));
        return static_cast<oi_handle>(ctx);
    } catch (const std::exception& e) {
        set_error(std::string("oi_create exception: ") + e.what());
        return nullptr;
    }
}

OI_API int oi_infer_file(oi_handle h,
                         const char* image_path_utf8,
                         const oi_result** out_result,
                         oi_frame_stat* out_stat)
{
    if (!h) {
        set_error("null OCR handle");
        return OI_ERR_INVALID_ARGUMENT;
    }
    auto* ctx = static_cast<OcrContext*>(h);
    std::lock_guard<std::mutex> lock(ctx->mutex);
    return run_file(ctx, image_path_utf8, out_result, out_stat);
}

OI_API int oi_infer_bgr(oi_handle h,
                        const unsigned char* bgr,
                        int width,
                        int height,
                        int stride,
                        const oi_result** out_result,
                        oi_frame_stat* out_stat)
{
    if (!h || !bgr || width <= 0 || height <= 0) {
        set_error("invalid BGR input");
        return OI_ERR_INVALID_ARGUMENT;
    }
    auto* ctx = static_cast<OcrContext*>(h);
    std::lock_guard<std::mutex> lock(ctx->mutex);

    const int line_size = stride > 0 ? stride : width * 3;
    cv::Mat bgr_view(height, width, CV_8UC3, const_cast<unsigned char*>(bgr),
                     static_cast<size_t>(line_size));
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        ctx->pipeline->Predict(std::vector<cv::Mat>{bgr_view.clone()});
    } catch (const std::exception& e) {
        set_error(std::string("OCR inference exception: ") + e.what());
        return OI_ERR_INFERENCE;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    const auto results = ctx->pipeline->PipelineResult();
    if (results.empty()) {
        set_error("OCR pipeline returned no result");
        return OI_ERR_INFERENCE;
    }
    const double pipeline_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const int rc = fill_result(ctx, results.front(), out_stat,
                               0.0, pipeline_ms, 0.0, pipeline_ms);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (out_stat) {
        out_stat->result_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        out_stat->postprocess_ms = out_stat->result_ms;
        out_stat->total_ms = pipeline_ms + out_stat->result_ms;
    }
    if (out_result) *out_result = &ctx->result;
    return rc;
}

OI_API void oi_destroy(oi_handle h)
{
    if (h) {
        auto* ctx = static_cast<OcrContext*>(h);
        delete ctx;
    }
}

OI_API const char* oi_last_error(void)
{
    return g_last_error.c_str();
}

} // extern "C"
