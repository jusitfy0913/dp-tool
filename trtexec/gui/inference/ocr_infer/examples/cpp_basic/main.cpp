#include "ocr_infer_c.h"

#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: ocr_basic <det_model_dir> <rec_model_dir> <image_path> [line_orientation_model_dir] [--bgr] [--device=cpu|gpu:0] [--precision=fp32|fp16] [--no-mkldnn]\n";
        return 1;
    }

    oi_config cfg{};
    cfg.det_model_dir_utf8 = argv[1];
    cfg.rec_model_dir_utf8 = argv[2];
    cfg.device_utf8 = "cpu";
    cfg.precision_utf8 = "fp32";
    cfg.enable_mkldnn = 1;
    cfg.cpu_threads = 8;
    cfg.rec_batch_size = 6;
    cfg.det_thresh = 0.3f;
    cfg.det_box_thresh = 0.6f;
    cfg.det_unclip_ratio = 1.5f;
    cfg.rec_score_thresh = 0.0f;
    bool use_bgr = false;
    std::string device = "cpu";
    std::string precision = "fp32";
    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i] ? argv[i] : "";
        if (arg == "--bgr") {
            use_bgr = true;
        } else if (arg.rfind("--device=", 0) == 0) {
            device = arg.substr(std::string("--device=").size());
        } else if (arg.rfind("--precision=", 0) == 0) {
            precision = arg.substr(std::string("--precision=").size());
        } else if (arg == "--no-mkldnn") {
            cfg.enable_mkldnn = 0;
        } else if (!arg.empty()) {
            cfg.textline_orientation_model_dir_utf8 = argv[i];
            cfg.use_textline_orientation = 1;
        }
    }
    cfg.device_utf8 = device.c_str();
    cfg.precision_utf8 = precision.c_str();

    oi_handle handle = oi_create(&cfg);
    if (!handle) {
        std::cerr << "oi_create failed: " << oi_last_error() << "\n";
        return 2;
    }

    const oi_result* result = nullptr;
    oi_frame_stat stat{};
    int rc = OI_ERR_INFERENCE;
    if (use_bgr) {
        cv::Mat image = cv::imread(argv[3], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "failed to read image: " << argv[3] << "\n";
            oi_destroy(handle);
            return 3;
        }
        rc = oi_infer_bgr(handle, image.data, image.cols, image.rows,
                          static_cast<int>(image.step), &result, &stat);
    } else {
        rc = oi_infer_file(handle, argv[3], &result, &stat);
    }
    if (rc != OI_OK || !result) {
        std::cerr << "oi_infer_file failed: " << oi_last_error() << "\n";
        oi_destroy(handle);
        return 3;
    }

    std::cout << "lines: " << result->line_count
              << ", total_ms: " << stat.total_ms
              << ", device=" << stat.device_utf8
              << ", run_mode=" << stat.run_mode_utf8
              << ", mkldnn=" << stat.mkldnn_enabled
              << ", threads=" << stat.cpu_threads
              << ", rec_batch=" << stat.rec_batch_size
              << ", io_ms=" << stat.io_ms
              << ", pipeline_ms=" << stat.pipeline_ms
              << ", result_ms=" << stat.result_ms << "\n";
    for (int i = 0; i < result->line_count; ++i) {
        const oi_text_line& line = result->lines[i];
        std::cout << "#" << line.index + 1
                  << " score=" << line.rec_score
                  << " angle=" << line.textline_angle
                  << " text=" << (line.text_utf8 ? line.text_utf8 : "")
                  << " poly=";
        for (int j = 0; j < line.point_count; ++j) {
            std::cout << "(" << line.xs[j] << "," << line.ys[j] << ")";
        }
        std::cout << "\n";
    }

    if (result->raw_json_utf8) {
        std::cout << "raw_json=" << result->raw_json_utf8 << "\n";
    }

    oi_destroy(handle);
    return 0;
}
