#include "anomalib_infer_c.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace {

thread_local std::string g_last_error;

void setError(const std::string& msg)
{
    g_last_error = msg;
}

std::string lowerCopy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

template <typename T>
void safeDestroy(T*& p)
{
    if (p) {
        delete p;
        p = nullptr;
    }
}

int64_t volume(const Dims& d)
{
    if (d.nbDims <= 0) return 0;
    return std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>());
}

size_t elementSize(DataType t)
{
    switch (t) {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF:  return 2;
        case DataType::kINT8:  return 1;
        case DataType::kINT32: return 4;
        case DataType::kBOOL:  return 1;
        default:               return 0;
    }
}

float halfToFloat(uint16_t h)
{
    const uint32_t sign = (h >> 15) & 0x1u;
    const uint32_t exponent = (h >> 10) & 0x1Fu;
    const uint32_t mantissa = h & 0x3FFu;
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) {
            f = sign << 31;
        } else {
            uint32_t e = 1, m = mantissa;
            while (!(m & 0x400u)) {
                m <<= 1;
                --e;
            }
            m &= 0x3FFu;
            f = (sign << 31) | ((e + 127u - 15u) << 23) | (m << 13);
        }
    } else if (exponent == 31) {
        f = (sign << 31) | 0x7F800000u | (mantissa << 13);
    } else {
        f = (sign << 31) | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

void convertHalfToFloat(const void* src, float* dst, int64_t count)
{
    const auto* in = reinterpret_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < count; ++i) {
        dst[i] = halfToFloat(in[i]);
    }
}

void convertInt32ToFloat(const void* src, float* dst, int64_t count)
{
    const auto* in = reinterpret_cast<const int32_t*>(src);
    for (int64_t i = 0; i < count; ++i) {
        dst[i] = static_cast<float>(in[i]);
    }
}

float clamp01(float v)
{
    return std::max(0.0f, std::min(1.0f, v));
}

float normalizeWithRange(float value, float threshold, float minVal, float maxVal)
{
    const float denom = maxVal - minVal;
    if (std::abs(denom) < 1.0e-6f) return clamp01(value);
    return clamp01(((value - threshold) / denom) + 0.5f);
}

std::filesystem::path utf8Path(const char* path)
{
    return std::filesystem::u8path(path ? path : "");
}

bool readBinaryFile(const char* path, std::vector<char>& out, std::string& err)
{
    std::ifstream file(utf8Path(path), std::ios::binary);
    if (!file) {
        err = std::string("无法读取文件: ") + (path ? path : "");
        return false;
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        err = std::string("文件为空: ") + (path ? path : "");
        return false;
    }
    file.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    file.read(out.data(), size);
    if (!file) {
        err = std::string("读取文件失败: ") + (path ? path : "");
        return false;
    }
    return true;
}

bool readTextFile(const char* path, std::string& out, std::string& err)
{
    std::vector<char> bytes;
    if (!readBinaryFile(path, bytes, err)) return false;
    out.assign(bytes.begin(), bytes.end());
    return true;
}

std::string regexEscape(const std::string& s)
{
    static const std::regex re(R"([-[\]{}()*+?.,\^$|#\s])");
    return std::regex_replace(s, re, R"(\$&)");
}

bool jsonNumber(const std::string& json, const std::string& key, float& out)
{
    const std::regex re("\"" + regexEscape(key) + R"("\s*:\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))");
    std::smatch m;
    if (!std::regex_search(json, m, re)) return false;
    out = std::stof(m[1].str());
    return true;
}

bool jsonString(const std::string& json, const std::string& key, std::string& out)
{
    const std::regex re("\"" + regexEscape(key) + R"json("\s*:\s*"([^"]*)")json");
    std::smatch m;
    if (!std::regex_search(json, m, re)) return false;
    out = m[1].str();
    return true;
}

bool jsonArray2Ints(const std::string& json, const std::string& key, int& a, int& b)
{
    const std::regex re("\"" + regexEscape(key) + R"("\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+))");
    std::smatch m;
    if (!std::regex_search(json, m, re)) return false;
    a = std::stoi(m[1].str());
    b = std::stoi(m[2].str());
    return true;
}

bool extractObject(const std::string& json, const std::string& key, std::string& out)
{
    const std::regex keyRe("\"" + regexEscape(key) + R"("\s*:\s*\{)");
    std::smatch m;
    if (!std::regex_search(json, m, keyRe)) return false;
    size_t open = static_cast<size_t>(m.position(0) + m.length(0) - 1);
    int depth = 0;
    bool inString = false;
    bool escape = false;
    for (size_t i = open; i < json.size(); ++i) {
        const char c = json[i];
        if (inString) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                inString = false;
            }
            continue;
        }
        if (c == '"') {
            inString = true;
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

bool jsonNestedNumber(const std::string& json, const std::string& parent, const std::string& key, float& out)
{
    std::string obj;
    return extractObject(json, parent, obj) && jsonNumber(obj, key, out);
}

bool findTransformSize(const std::string& json, int& h, int& w)
{
    std::string transform;
    if (extractObject(json, "transform", transform)) {
        float hf = 0.0f;
        float wf = 0.0f;
        if (jsonNumber(transform, "height", hf) && jsonNumber(transform, "width", wf)) {
            h = static_cast<int>(hf);
            w = static_cast<int>(wf);
            return true;
        }
    }
    return false;
}

void copyCString(char* dst, size_t dstSize, const std::string& src)
{
    if (!dst || dstSize == 0) return;
    std::snprintf(dst, dstSize, "%s", src.c_str());
}

class TrtLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING && msg) {
            last_message = msg;
        }
    }

    std::string last_message;
};

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

struct TensorBuffer {
    int bindingIndex = -1;
    std::string name;
    bool isInput = false;
    Dims dims{};
    DataType dtype = DataType::kFLOAT;
    size_t bytes = 0;
    void* device = nullptr;
    std::vector<float> host;
    std::vector<unsigned char> rawHost;
    int64_t elemCount = 0;
};

class AnomalyEngine {
public:
    ~AnomalyEngine()
    {
        unload();
    }

    bool load(const char* enginePath, const char* metadataPath, const char* modelType,
              float imageThresholdOverride, float pixelThresholdOverride)
    {
        unload();
        std::string err;
        if (!loadMetadata(metadataPath, modelType ? modelType : "Auto", err)) {
            setError("metadata 加载失败: " + err);
            return false;
        }
        if (imageThresholdOverride >= 0.0f) meta_.imageThreshold = imageThresholdOverride;
        if (pixelThresholdOverride >= 0.0f) meta_.pixelThreshold = pixelThresholdOverride;
        if (!prepareEngine(enginePath, err)) {
            setError("异常检测引擎加载失败: " + err);
            unload();
            return false;
        }
        return true;
    }

    void unload()
    {
        for (TensorBuffer& b : buffers_) {
            if (b.device) {
                cudaFree(b.device);
                b.device = nullptr;
            }
        }
        buffers_.clear();
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        safeDestroy(context_);
        safeDestroy(engine_);
        safeDestroy(runtime_);
        inputBinding_ = -1;
        result_ = {};
        sourceRgb_.clear();
        overlayRgb_.clear();
        maskRgb_.clear();
        normalizedMap_.clear();
    }

    const Metadata& metadata() const
    {
        return meta_;
    }

    bool inferRgb(const unsigned char* rgb, int width, int height, int stride,
                  const ai_result** outResult, ai_frame_stat* outStat)
    {
        if (!context_ || buffers_.empty()) {
            setError("推理句柄尚未加载 engine");
            return false;
        }
        if (!rgb || width <= 0 || height <= 0 || stride < width * 3 || !outResult) {
            setError("输入图像参数无效");
            return false;
        }

        const auto totalStart = std::chrono::steady_clock::now();
        auto stageStart = totalStart;

        cv::Mat rgbView(height, width, CV_8UC3, const_cast<unsigned char*>(rgb), static_cast<size_t>(stride));
        cv::Mat rgbMat = rgbView.clone();
        cv::Mat resized;
        cv::resize(rgbMat, resized, cv::Size(meta_.inputW, meta_.inputH), 0, 0, cv::INTER_LINEAR);
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
        if (meta_.normalization == "imagenet") {
            const cv::Scalar mean(0.485, 0.456, 0.406);
            const cv::Scalar stdv(0.229, 0.224, 0.225);
            resized = (resized - mean) / stdv;
        }
        cv::Mat blob = cv::dnn::blobFromImage(resized);
        TensorBuffer& input = buffers_[inputBinding_];
        cudaError_t cerr = cudaMemcpyAsync(input.device, blob.ptr<float>(), input.bytes, cudaMemcpyHostToDevice, stream_);
        if (cerr != cudaSuccess) {
            setError(std::string("输入拷贝到 GPU 失败: ") + cudaGetErrorString(cerr));
            return false;
        }
        const double preMs = elapsedMs(stageStart);

        stageStart = std::chrono::steady_clock::now();
        std::vector<void*> bindings(buffers_.size(), nullptr);
        for (const TensorBuffer& b : buffers_) {
            bindings[b.bindingIndex] = b.device;
        }
        if (!context_->enqueueV2(bindings.data(), stream_, nullptr)) {
            setError("TensorRT enqueueV2 执行失败");
            return false;
        }
        cudaStreamSynchronize(stream_);
        const double inferMs = elapsedMs(stageStart);

        stageStart = std::chrono::steady_clock::now();
        std::vector<TensorBuffer*> outputs;
        for (TensorBuffer& b : buffers_) {
            if (!b.isInput) {
                if (b.dtype == DataType::kFLOAT) {
                    cerr = cudaMemcpyAsync(b.host.data(), b.device, b.bytes, cudaMemcpyDeviceToHost, stream_);
                } else {
                    cerr = cudaMemcpyAsync(b.rawHost.data(), b.device, b.bytes, cudaMemcpyDeviceToHost, stream_);
                }
                if (cerr != cudaSuccess) {
                    setError(std::string("输出拷贝到 CPU 失败: ") + cudaGetErrorString(cerr));
                    return false;
                }
                outputs.push_back(&b);
            }
        }
        cudaStreamSynchronize(stream_);

        for (TensorBuffer* b : outputs) {
            if (b->dtype == DataType::kHALF) {
                convertHalfToFloat(b->rawHost.data(), b->host.data(), b->elemCount);
            } else if (b->dtype == DataType::kINT32) {
                convertInt32ToFloat(b->rawHost.data(), b->host.data(), b->elemCount);
            }
        }
        if (outputs.empty()) {
            setError("engine 没有输出 tensor");
            return false;
        }

        TensorBuffer* mapOutput = nullptr;
        TensorBuffer* scoreOutput = nullptr;
        for (TensorBuffer* output : outputs) {
            const std::string name = lowerCopy(output->name);
            if (name.find("score") != std::string::npos) {
                scoreOutput = output;
            }
            if (name.find("anomaly_map") != std::string::npos ||
                (name.find("map") != std::string::npos && name.find("mask") == std::string::npos)) {
                mapOutput = output;
            }
            if (!mapOutput || output->elemCount > mapOutput->elemCount) {
                mapOutput = output;
            }
            if (output->elemCount == 1 && !scoreOutput && name.find("label") == std::string::npos) {
                scoreOutput = output;
            }
        }
        if (!mapOutput) mapOutput = outputs.front();
        if (scoreOutput == mapOutput) scoreOutput = nullptr;

        int mapH = meta_.inputH;
        int mapW = meta_.inputW;
        if (mapOutput->dims.nbDims >= 2) {
            mapH = mapOutput->dims.d[mapOutput->dims.nbDims - 2];
            mapW = mapOutput->dims.d[mapOutput->dims.nbDims - 1];
        }
        if (mapH <= 0 || mapW <= 0 || mapH * mapW > static_cast<int>(mapOutput->host.size())) {
            const int approx = static_cast<int>(std::sqrt(static_cast<double>(mapOutput->host.size())));
            mapH = approx;
            mapW = approx;
        }
        if (mapH <= 0 || mapW <= 0 || mapH * mapW > static_cast<int>(mapOutput->host.size())) {
            setError("无法解析 anomaly map 输出尺寸");
            return false;
        }

        cv::Mat rawMap(mapH, mapW, CV_32FC1, mapOutput->host.data());
        rawMap = rawMap.clone();
        float rawScore = 0.0f;
        if (scoreOutput && !scoreOutput->host.empty()) {
            rawScore = scoreOutput->host[0];
        } else {
            double maxValue = 0.0;
            cv::minMaxLoc(rawMap, nullptr, &maxValue);
            rawScore = static_cast<float>(maxValue);
        }

        cv::Mat normalizedMap = ((rawMap - meta_.pixelThreshold) / (meta_.mapMax - meta_.mapMin)) + 0.5f;
        cv::threshold(normalizedMap, normalizedMap, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(normalizedMap, normalizedMap, 0.0, 0.0, cv::THRESH_TOZERO);
        cv::Mat normalizedMapCompact = normalizedMap.isContinuous() ? normalizedMap : normalizedMap.clone();
        const auto* mapPtr = reinterpret_cast<const float*>(normalizedMapCompact.data);
        normalizedMap_.assign(mapPtr, mapPtr + normalizedMapCompact.total());

        cv::Mat displayMap;
        cv::resize(normalizedMap, displayMap, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
        cv::Mat map8u;
        displayMap.convertTo(map8u, CV_8UC1, 255.0);
        cv::Mat heatBgr;
        cv::applyColorMap(map8u, heatBgr, cv::COLORMAP_JET);

        cv::Mat originalBgr;
        cv::cvtColor(rgbMat, originalBgr, cv::COLOR_RGB2BGR);
        cv::Mat overlayBgr;
        cv::addWeighted(heatBgr, 0.45, originalBgr, 0.55, 0, overlayBgr);

        cv::Mat mask;
        cv::threshold(displayMap, mask, 0.5, 255.0, cv::THRESH_BINARY);
        mask.convertTo(mask, CV_8UC1);
        cv::Mat maskOverlayBgr = originalBgr.clone();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (!contours.empty()) {
            cv::drawContours(maskOverlayBgr, contours, -1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }

        const float normalizedScore = normalizeWithRange(rawScore, meta_.imageThreshold, meta_.scoreMin, meta_.scoreMax);
        const bool isAnomaly = normalizedScore >= 0.5f;
        const std::string status = std::string("score ") + formatScore(normalizedScore) + (isAnomaly ? "  NG" : "  OK");
        const cv::Scalar statusColor = isAnomaly ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 220, 0);
        cv::putText(overlayBgr, status, cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);
        cv::putText(maskOverlayBgr, status, cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);

        cv::Mat overlayRgb;
        cv::Mat maskRgb;
        cv::cvtColor(overlayBgr, overlayRgb, cv::COLOR_BGR2RGB);
        cv::cvtColor(maskOverlayBgr, maskRgb, cv::COLOR_BGR2RGB);

        assignMatBytes(rgbMat, sourceRgb_);
        assignMatBytes(overlayRgb, overlayRgb_);
        assignMatBytes(maskRgb, maskRgb_);

        result_.source_rgb = sourceRgb_.data();
        result_.source_w = width;
        result_.source_h = height;
        result_.source_stride = width * 3;
        result_.overlay_rgb = overlayRgb_.data();
        result_.overlay_w = width;
        result_.overlay_h = height;
        result_.overlay_stride = width * 3;
        result_.mask_rgb = maskRgb_.data();
        result_.mask_w = width;
        result_.mask_h = height;
        result_.mask_stride = width * 3;
        result_.normalized_map = normalizedMap_.data();
        result_.map_w = mapW;
        result_.map_h = mapH;
        result_.raw_score = rawScore;
        result_.normalized_score = normalizedScore;
        result_.is_anomaly = isAnomaly ? 1 : 0;

        const double postMs = elapsedMs(stageStart);
        const double totalMs = elapsedMs(totalStart);
        if (outStat) {
            outStat->src_w = width;
            outStat->src_h = height;
            outStat->preprocess_ms = preMs;
            outStat->inference_ms = inferMs;
            outStat->postprocess_ms = postMs;
            outStat->total_ms = totalMs;
        }
        *outResult = &result_;
        return true;
    }

private:
    static double elapsedMs(const std::chrono::steady_clock::time_point& start)
    {
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    static std::string formatScore(float score)
    {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(3);
        oss << score;
        return oss.str();
    }

    static void assignMatBytes(const cv::Mat& mat, std::vector<unsigned char>& out)
    {
        cv::Mat compact = mat.isContinuous() ? mat : mat.clone();
        out.assign(compact.data, compact.data + compact.total() * compact.elemSize());
    }

    bool loadMetadata(const char* path, const std::string& modelType, std::string& err)
    {
        std::string json;
        if (!readTextFile(path, json, err)) return false;

        Metadata meta;
        if (!jsonString(json, "model_type", meta.modelType) || meta.modelType.empty()) {
            meta.modelType = modelType.empty() ? "Auto" : modelType;
        }
        jsonString(json, "normalization", meta.normalization);
        meta.normalization = lowerCopy(meta.normalization);
        jsonString(json, "output_contract", meta.outputContract);
        meta.outputContract = lowerCopy(meta.outputContract);

        if (meta.normalization.empty()) {
            if (meta.outputContract == "anomalib_post_processed_0_1") {
                meta.normalization = "none";
            } else {
                meta.normalization = lowerCopy(meta.modelType).find("efficient") != std::string::npos
                    ? "none"
                    : "imagenet";
            }
        }

        const bool okImage = jsonNumber(json, "image_threshold", meta.imageThreshold);
        const bool okPixel = jsonNumber(json, "pixel_threshold", meta.pixelThreshold);
        bool okScoreMin = jsonNestedNumber(json, "pred_scores", "min", meta.scoreMin);
        bool okScoreMax = jsonNestedNumber(json, "pred_scores", "max", meta.scoreMax);
        bool okMapMin = jsonNestedNumber(json, "anomaly_maps", "min", meta.mapMin);
        bool okMapMax = jsonNestedNumber(json, "anomaly_maps", "max", meta.mapMax);
        if (!okScoreMin) okScoreMin = jsonNumber(json, "pred_scores.min", meta.scoreMin);
        if (!okScoreMax) okScoreMax = jsonNumber(json, "pred_scores.max", meta.scoreMax);
        if (!okMapMin) okMapMin = jsonNumber(json, "anomaly_maps.min", meta.mapMin);
        if (!okMapMax) okMapMax = jsonNumber(json, "anomaly_maps.max", meta.mapMax);

        float flatMin = 0.0f;
        float flatMax = 1.0f;
        const bool okFlatMin = jsonNumber(json, "min", flatMin);
        const bool okFlatMax = jsonNumber(json, "max", flatMax);
        if (!okScoreMin && okFlatMin) meta.scoreMin = flatMin;
        if (!okScoreMax && okFlatMax) meta.scoreMax = flatMax;
        if (!okMapMin && okFlatMin) meta.mapMin = flatMin;
        if (!okMapMax && okFlatMax) meta.mapMax = flatMax;

        if (!findTransformSize(json, meta.inputH, meta.inputW)) {
            int h = 0;
            int w = 0;
            if (jsonArray2Ints(json, "input_size", h, w) || jsonArray2Ints(json, "image_size", h, w)) {
                meta.inputH = h;
                meta.inputW = w;
            }
        }

        if (!okImage || !okPixel) {
            err = "缺少 image_threshold 或 pixel_threshold 字段";
            return false;
        }
        if (meta.inputH <= 0 || meta.inputW <= 0) {
            meta.inputH = 256;
            meta.inputW = 256;
        }
        if (meta.scoreMax <= meta.scoreMin) meta.scoreMax = meta.scoreMin + 1.0f;
        if (meta.mapMax <= meta.mapMin) meta.mapMax = meta.mapMin + 1.0f;
        meta_ = meta;
        return true;
    }

    bool prepareEngine(const char* path, std::string& err)
    {
        std::vector<char> engineBytes;
        if (!readBinaryFile(path, engineBytes, err)) return false;

        cudaError_t cerr = cudaSetDevice(0);
        if (cerr != cudaSuccess) {
            err = std::string("cudaSetDevice(0) 失败: ") + cudaGetErrorString(cerr);
            return false;
        }

        initLibNvInferPlugins(&logger_, "");
        runtime_ = createInferRuntime(logger_);
        if (!runtime_) {
            err = "createInferRuntime 失败";
            return false;
        }
        engine_ = runtime_->deserializeCudaEngine(engineBytes.data(), engineBytes.size());
        if (!engine_) {
            err = "deserializeCudaEngine 失败，请确认 engine 与当前 TensorRT/CUDA 版本匹配";
            if (!logger_.last_message.empty()) err += ": " + logger_.last_message;
            return false;
        }
        context_ = engine_->createExecutionContext();
        if (!context_) {
            err = "createExecutionContext 失败";
            return false;
        }
        cerr = cudaStreamCreate(&stream_);
        if (cerr != cudaSuccess) {
            err = std::string("cudaStreamCreate 失败: ") + cudaGetErrorString(cerr);
            return false;
        }

        const int nb = engine_->getNbBindings();
        if (nb < 2) {
            err = "异常检测 engine 至少需要 1 个输入和 1 个输出";
            return false;
        }
        buffers_.clear();
        buffers_.resize(static_cast<size_t>(nb));

        for (int i = 0; i < nb; ++i) {
            TensorBuffer b;
            b.bindingIndex = i;
            b.name = engine_->getBindingName(i) ? engine_->getBindingName(i) : "";
            b.isInput = engine_->bindingIsInput(i);
            b.dtype = engine_->getBindingDataType(i);
            Dims dims = context_->getBindingDimensions(i);
            if (b.isInput) {
                inputBinding_ = i;
                bool hasDynamic = false;
                for (int k = 0; k < dims.nbDims; ++k) {
                    if (dims.d[k] < 0) hasDynamic = true;
                }
                if (hasDynamic) {
                    Dims inputDims{};
                    inputDims.nbDims = 4;
                    inputDims.d[0] = 1;
                    inputDims.d[1] = 3;
                    inputDims.d[2] = meta_.inputH;
                    inputDims.d[3] = meta_.inputW;
                    if (!context_->setBindingDimensions(i, inputDims)) {
                        err = "设置动态输入尺寸失败";
                        return false;
                    }
                    dims = context_->getBindingDimensions(i);
                } else if (dims.nbDims == 4) {
                    meta_.inputH = dims.d[2];
                    meta_.inputW = dims.d[3];
                }
            }

            b.dims = dims;
            const int64_t elems = volume(dims);
            const size_t elemBytes = elementSize(b.dtype);
            if (elems <= 0 || elemBytes == 0) {
                err = "无效 tensor 尺寸或类型，binding=" + std::to_string(i);
                return false;
            }
            b.bytes = static_cast<size_t>(elems) * elemBytes;
            cerr = cudaMalloc(&b.device, b.bytes);
            if (cerr != cudaSuccess) {
                err = "cudaMalloc 失败，binding=" + std::to_string(i) + ": " + cudaGetErrorString(cerr);
                return false;
            }
            b.elemCount = elems;
            if (!b.isInput) {
                b.host.resize(static_cast<size_t>(elems));
                if (b.dtype != DataType::kFLOAT) {
                    b.rawHost.resize(b.bytes);
                }
            }
            buffers_[static_cast<size_t>(i)] = std::move(b);
        }

        if (inputBinding_ < 0) {
            err = "未找到输入 tensor";
            return false;
        }
        if (buffers_[static_cast<size_t>(inputBinding_)].dtype != DataType::kFLOAT) {
            err = "当前 Anomalib DLL 仅支持 FP32 输入 tensor";
            return false;
        }
        return true;
    }

    Metadata meta_;
    TrtLogger logger_;
    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::vector<TensorBuffer> buffers_;
    int inputBinding_ = -1;

    std::vector<unsigned char> sourceRgb_;
    std::vector<unsigned char> overlayRgb_;
    std::vector<unsigned char> maskRgb_;
    std::vector<float> normalizedMap_;
    ai_result result_{};
};

void fillMetadata(const Metadata& src, ai_metadata* dst)
{
    if (!dst) return;
    std::memset(dst, 0, sizeof(ai_metadata));
    copyCString(dst->model_type, sizeof(dst->model_type), src.modelType);
    copyCString(dst->normalization, sizeof(dst->normalization), src.normalization);
    copyCString(dst->output_contract, sizeof(dst->output_contract), src.outputContract);
    dst->image_threshold = src.imageThreshold;
    dst->pixel_threshold = src.pixelThreshold;
    dst->score_min = src.scoreMin;
    dst->score_max = src.scoreMax;
    dst->map_min = src.mapMin;
    dst->map_max = src.mapMax;
    dst->input_h = src.inputH;
    dst->input_w = src.inputW;
}

} // namespace

extern "C" {

AI_API ai_handle ai_create(const char* engine_path_utf8,
                           const char* metadata_path_utf8,
                           const char* model_type_utf8,
                           float image_threshold_override,
                           float pixel_threshold_override)
{
    g_last_error.clear();
    if (!engine_path_utf8 || !metadata_path_utf8) {
        setError("engine_path_utf8 和 metadata_path_utf8 不能为空");
        return nullptr;
    }
    auto engine = std::make_unique<AnomalyEngine>();
    if (!engine->load(engine_path_utf8, metadata_path_utf8, model_type_utf8,
                      image_threshold_override, pixel_threshold_override)) {
        return nullptr;
    }
    return engine.release();
}

AI_API int ai_get_metadata(ai_handle h, ai_metadata* out_metadata)
{
    if (!h || !out_metadata) {
        setError("ai_get_metadata 参数无效");
        return AI_ERR_INVALID_ARGUMENT;
    }
    const auto* engine = reinterpret_cast<AnomalyEngine*>(h);
    fillMetadata(engine->metadata(), out_metadata);
    return AI_OK;
}

AI_API int ai_infer_rgb(ai_handle h,
                        const unsigned char* rgb,
                        int width,
                        int height,
                        int stride,
                        const ai_result** out_result,
                        ai_frame_stat* out_stat)
{
    if (!h) {
        setError("ai_handle 为空");
        return AI_ERR_INVALID_ARGUMENT;
    }
    auto* engine = reinterpret_cast<AnomalyEngine*>(h);
    return engine->inferRgb(rgb, width, height, stride, out_result, out_stat)
        ? AI_OK
        : AI_ERR_INFERENCE;
}

AI_API void ai_destroy(ai_handle h)
{
    auto* engine = reinterpret_cast<AnomalyEngine*>(h);
    delete engine;
}

AI_API const char* ai_last_error(void)
{
    return g_last_error.c_str();
}

} // extern "C"
