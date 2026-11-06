#pragma once

/*
 * Anomalib 后处理的共享实现。
 *
 * 为什么单独抽出来：
 *   阈值归一化、判定边界、热力叠加权重、异常轮廓颜色这些规则，一旦在
 *   anomalib_infer（TensorRT）和 onnx_infer（ONNX Runtime）里各写一份，
 *   同一条产线换个后端就会得出不同结论，这种问题在验收阶段极难被发现。
 *
 * 因此这里只放纯函数与纯几何渲染，不碰任何推理框架的类型：
 *   两个 DLL 各自把张量整理成 cv::Mat，再调用同一份 render()。
 */

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace anomalib_post {

inline float clamp01(float value) { return std::max(0.0f, std::min(1.0f, value)); }

/*
 * 把原始分数映射到 [0,1]：
 *   threshold 处正好是 0.5，minVal/maxVal 决定两侧的斜率。
 *   minVal 与 maxVal 相等时退化为直接裁剪，避免除零。
 */
inline float normalizeWithRange(float value, float threshold, float minVal, float maxVal) {
    const float denom = maxVal - minVal;
    if (std::abs(denom) < 1.0e-6f) {
        return clamp01(value);
    }
    return clamp01(((value - threshold) / denom) + 0.5f);
}

inline std::string formatScore(float score) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << score;
    return oss.str();
}

/* 渲染所需的阈值参数，语义与 metadata.json 完全一致。 */
struct RenderConfig {
    float imageThreshold = 0.5f;
    float pixelThreshold = 0.5f;
    float mapMin = 0.0f;
    float mapMax = 1.0f;
    float scoreMin = 0.0f;
    float scoreMax = 1.0f;
};

struct RenderOutput {
    cv::Mat overlay_rgb;     // 热力图叠加，RGB888
    cv::Mat mask_rgb;        // 红色异常轮廓，RGB888
    cv::Mat normalized_map;  // 归一化异常图，CV_32FC1，保持原始 map 分辨率
    float normalized_score = 0.0f;
    bool is_anomaly = false;
};

/*
 * 生成三张展示图与归一化异常图。
 *
 * 参数：
 * - rgb：原图，CV_8UC3，RGB 通道顺序（与 ai_infer_rgb 的入参一致）。
 * - rawMap：原始异常图，CV_32FC1。
 * - rawScore：原始图像级分数；没有单独 score 输出时调用方用 map 的最大值代替。
 */
inline RenderOutput render(const cv::Mat& rgb, const cv::Mat& rawMap, float rawScore,
                           const RenderConfig& config) {
    RenderOutput output;

    cv::Mat normalizedMap = ((rawMap - config.pixelThreshold) / (config.mapMax - config.mapMin)) + 0.5f;
    cv::threshold(normalizedMap, normalizedMap, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(normalizedMap, normalizedMap, 0.0, 0.0, cv::THRESH_TOZERO);
    output.normalized_map = normalizedMap;

    cv::Mat displayMap;
    cv::resize(normalizedMap, displayMap, cv::Size(rgb.cols, rgb.rows), 0, 0, cv::INTER_LINEAR);

    cv::Mat map8u;
    displayMap.convertTo(map8u, CV_8UC1, 255.0);
    cv::Mat heatBgr;
    cv::applyColorMap(map8u, heatBgr, cv::COLORMAP_JET);

    cv::Mat originalBgr;
    cv::cvtColor(rgb, originalBgr, cv::COLOR_RGB2BGR);

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

    // 判定边界固定为 0.5：这是 metadata 归一化之后的口径，换后端也不能变。
    output.normalized_score =
        normalizeWithRange(rawScore, config.imageThreshold, config.scoreMin, config.scoreMax);
    output.is_anomaly = output.normalized_score >= 0.5f;

    const std::string status =
        std::string("score ") + formatScore(output.normalized_score) + (output.is_anomaly ? "  NG" : "  OK");
    const cv::Scalar statusColor = output.is_anomaly ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 220, 0);
    cv::putText(overlayBgr, status, cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);
    cv::putText(maskOverlayBgr, status, cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);

    cv::cvtColor(overlayBgr, output.overlay_rgb, cv::COLOR_BGR2RGB);
    cv::cvtColor(maskOverlayBgr, output.mask_rgb, cv::COLOR_BGR2RGB);
    return output;
}

}  // namespace anomalib_post
