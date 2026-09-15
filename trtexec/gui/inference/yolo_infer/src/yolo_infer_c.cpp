#include "yolo_infer_c.h"
#include "yolo.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <cstring>

static thread_local std::string g_last_error;

struct YoloContext {
    std::shared_ptr<yolo::Infer> infer;
    yi_task                      task;
    std::vector<yi_box>          boxes_buf;
    std::vector<std::shared_ptr<yolo::InstanceSegmentMap>> seg_hold;
    std::vector<cv::Mat>         mask_hold; // 持有 resize 后的掩膜内存
    std::vector<std::vector<float>> kpt_hold; // 持有每个 box 的关键点数据，供 yi_box.kpts 指向
};

static void set_error(const std::string& msg) { g_last_error = msg; }

extern "C" {

YI_API yi_handle yi_create(const char* engine_path, yi_task task,
                           float conf_thr, float nms_thr)
{
    try {
        yolo::Type type = yolo::Type::V8;
        if (task == YI_SEGMENT) type = yolo::Type::V8Seg;
        else if (task == YI_POSE) type = yolo::Type::V8Pose;
        else if (task == YI_OBB) type = yolo::Type::V8OBB;
        auto infer = yolo::load(engine_path, type, conf_thr, nms_thr);
        if (!infer) {
            set_error("Failed to load engine: " + std::string(engine_path));
            return nullptr;
        }
        auto* ctx  = new YoloContext();
        ctx->infer = infer;
        ctx->task  = task;
        return static_cast<yi_handle>(ctx);
    } catch (const std::exception& e) {
        set_error(std::string("yi_create exception: ") + e.what());
        return nullptr;
    }
}

YI_API int yi_infer_bgr(yi_handle h,
                        const unsigned char* bgr, int w, int height, int stride,
                        yi_box** out_boxes, int* out_count,
                        yi_frame_stat* out_stat)
{
    if (!h) { set_error("null handle"); return -1; }
    auto* ctx = static_cast<YoloContext*>(h);

    try {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 不再包装 cv::Mat：stride 直接透传给 yolo::Image，避免行对齐丢失。
        // stride <= 0 视为紧凑（width * 3）。
        const int line_size = (stride > 0) ? stride : (w * 3);

        auto t1 = std::chrono::high_resolution_clock::now();

        yolo::Image img(bgr, w, height, line_size);
        yolo::BoxArray results = ctx->infer->forward(img);

        auto t2 = std::chrono::high_resolution_clock::now();

        ctx->boxes_buf.clear();
        ctx->seg_hold.clear();
        ctx->mask_hold.clear();
        ctx->kpt_hold.clear();
        ctx->boxes_buf.reserve(results.size());
        ctx->mask_hold.reserve(results.size());
        ctx->kpt_hold.reserve(results.size());

        const int img_w = w;
        const int img_h = height;

        for (auto& box : results) {
            yi_box ob;
            ob.left       = box.left;
            ob.top        = box.top;
            ob.right      = box.right;
            ob.bottom     = box.bottom;
            ob.confidence = box.confidence;
            ob.class_label = box.class_label;
            ob.has_mask  = 0;
            ob.mask_x = ob.mask_y = ob.mask_w = ob.mask_h = 0;
            ob.mask_data = nullptr;
            ob.num_kpts  = 0;
            ob.kpts      = nullptr;
            ob.is_obb    = 0;
            ob.cx = ob.cy = ob.ow = ob.oh = ob.angle = 0.f;

            if (ctx->task == YI_OBB) {
                ob.is_obb = 1;
                ob.cx     = box.cx;
                ob.cy     = box.cy;
                ob.ow     = box.ow;
                ob.oh     = box.oh;
                ob.angle  = box.angle;
            }

            // Pose keypoints: copy into ctx->kpt_hold then point yi_box.kpts to it.
            if (!box.kps.empty() && (box.kps.size() % 3) == 0) {
                ctx->kpt_hold.push_back(box.kps);
                ob.num_kpts = static_cast<int>(box.kps.size() / 3);
                ob.kpts     = ctx->kpt_hold.back().data();
            }

            if (box.seg && box.seg->data && box.seg->width > 0 && box.seg->height > 0) {
                // 原始（未裁）bbox
                float ox0 = box.left,  oy0 = box.top;
                float ox1 = box.right, oy1 = box.bottom;
                float ow  = ox1 - ox0, oh = oy1 - oy0;
                if (ow > 0 && oh > 0) {
                    // 可见 ROI（裁到画面内）
                    int vx0 = std::max(int(std::floor(ox0)), 0);
                    int vy0 = std::max(int(std::floor(oy0)), 0);
                    int vx1 = std::min(int(std::ceil (ox1)), img_w);
                    int vy1 = std::min(int(std::ceil (oy1)), img_h);
                    int vw = vx1 - vx0, vh = vy1 - vy0;
                    if (vw > 0 && vh > 0) {
                        cv::Mat segMask(box.seg->height, box.seg->width,
                                        CV_8U, box.seg->data);
                        // 把可见 ROI 的边界换算到 segMask 坐标（按比例 "先裁后缩"）
                        float sx = float(box.seg->width)  / ow;
                        float sy = float(box.seg->height) / oh;
                        int cx = int(std::round((vx0 - ox0) * sx));
                        int cy = int(std::round((vy0 - oy0) * sy));
                        int cw = std::max(1, int(std::round(vw * sx)));
                        int ch = std::max(1, int(std::round(vh * sy)));
                        cv::Rect segCrop(cx, cy, cw, ch);
                        segCrop &= cv::Rect(0, 0, segMask.cols, segMask.rows);
                        if (segCrop.width > 0 && segCrop.height > 0) {
                            cv::Mat resized;
                            cv::resize(segMask(segCrop), resized,
                                       cv::Size(vw, vh));
                            ctx->mask_hold.push_back(resized);
                            ob.has_mask  = 1;
                            ob.mask_x    = vx0;
                            ob.mask_y    = vy0;
                            ob.mask_w    = vw;
                            ob.mask_h    = vh;
                            ob.mask_data = ctx->mask_hold.back().data;
                            ctx->seg_hold.push_back(box.seg);
                        }
                    }
                }
            }
            ctx->boxes_buf.push_back(ob);
        }

        auto t3 = std::chrono::high_resolution_clock::now();

        if (out_boxes) *out_boxes = ctx->boxes_buf.empty() ? nullptr : ctx->boxes_buf.data();
        if (out_count) *out_count = static_cast<int>(ctx->boxes_buf.size());

        if (out_stat) {
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            out_stat->src_w = w;
            out_stat->src_h = height;
            out_stat->count = static_cast<int>(ctx->boxes_buf.size());
            out_stat->preprocess_ms  = ms(t0, t1);
            out_stat->inference_ms   = ms(t1, t2);
            out_stat->postprocess_ms = ms(t2, t3);
            out_stat->total_ms       = ms(t0, t3);
        }
        return 0;
    } catch (const std::exception& e) {
        set_error(std::string("yi_infer_bgr exception: ") + e.what());
        return -1;
    }
}

YI_API void yi_destroy(yi_handle h)
{
    if (h) {
        auto* ctx = static_cast<YoloContext*>(h);
        delete ctx;
    }
}

YI_API const char* yi_last_error(void)
{
    return g_last_error.c_str();
}

} // extern "C"
