#include "anomalib_infer_c.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>

namespace {

cv::Mat resultImageToMat(const unsigned char* data, int w, int h, int stride)
{
    if (!data || w <= 0 || h <= 0 || stride <= 0) return {};
    cv::Mat rgb(h, w, CV_8UC3, const_cast<unsigned char*>(data), static_cast<size_t>(stride));
    return rgb.clone();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "Usage: anomalib_infer_cpp_basic <engine.trt> <metadata.json> <image> <out_dir>\n";
        return 1;
    }

    const char* enginePath = argv[1];
    const char* metadataPath = argv[2];
    const char* imagePath = argv[3];
    const std::string outDir = argv[4];

    cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Failed to read image: " << imagePath << "\n";
        return 2;
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    ai_handle handle = ai_create(enginePath, metadataPath, "Auto", -1.0f, -1.0f);
    if (!handle) {
        std::cerr << "ai_create failed: " << ai_last_error() << "\n";
        return 3;
    }

    ai_metadata meta{};
    if (ai_get_metadata(handle, &meta) != AI_OK) {
        std::cerr << "ai_get_metadata failed: " << ai_last_error() << "\n";
        ai_destroy(handle);
        return 4;
    }
    std::cout << "model=" << meta.model_type
              << " input=" << meta.input_w << "x" << meta.input_h
              << " normalization=" << meta.normalization << "\n";

    const ai_result* result = nullptr;
    ai_frame_stat stat{};
    const int rc = ai_infer_rgb(handle,
                                rgb.data,
                                rgb.cols,
                                rgb.rows,
                                static_cast<int>(rgb.step),
                                &result,
                                &stat);
    if (rc != AI_OK || !result) {
        std::cerr << "ai_infer_rgb failed: " << ai_last_error() << "\n";
        ai_destroy(handle);
        return 5;
    }

    cv::Mat overlayRgb = resultImageToMat(result->overlay_rgb, result->overlay_w, result->overlay_h, result->overlay_stride);
    cv::Mat maskRgb = resultImageToMat(result->mask_rgb, result->mask_w, result->mask_h, result->mask_stride);
    cv::Mat overlayBgr;
    cv::Mat maskBgr;
    cv::cvtColor(overlayRgb, overlayBgr, cv::COLOR_RGB2BGR);
    cv::cvtColor(maskRgb, maskBgr, cv::COLOR_RGB2BGR);

    cv::imwrite(outDir + "/overlay.png", overlayBgr);
    cv::imwrite(outDir + "/mask.png", maskBgr);

    std::cout << "raw_score=" << result->raw_score
              << " normalized_score=" << result->normalized_score
              << " status=" << (result->is_anomaly ? "NG" : "OK")
              << " infer_ms=" << stat.inference_ms << "\n";

    ai_destroy(handle);
    return 0;
}
