#include "InferWorker.h"
#include <QDebug>
#include <QElapsedTimer>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace trtforge {

InferWorker::InferWorker(QObject* parent) : QObject(parent) {}

InferWorker::~InferWorker()
{
    unloadEngine();
}

bool InferWorker::loadEngine(const QString& enginePath, int task,
                             float confThr, float nmsThr)
{
    unloadEngine();
    m_handle = yi_create(enginePath.toUtf8().constData(),
                         static_cast<yi_task>(task),
                         confThr,
                         nmsThr);
    if (!m_handle) {
        emit engineLoadFailed(QStringLiteral("加载引擎失败: %1").arg(QString::fromUtf8(yi_last_error())));
        return false;
    }
    emit engineLoaded();
    return true;
}

void InferWorker::unloadEngine()
{
    if (m_handle) {
        yi_destroy(m_handle);
        m_handle = nullptr;
    }
}

void InferWorker::onFrame(QImage frame, qint64 /*timestampMs*/)
{
    if (!m_handle) return;

    bool expected = false;
    if (!m_busy.compare_exchange_strong(expected, true))
        return;

    // ---- CPU 端预处理（QImage 格式归一化 + RGB→BGR 通道交换）----
    QElapsedTimer timer;
    timer.start();

    QImage rgb = frame;
    if (rgb.format() != QImage::Format_RGB888)
        rgb = rgb.convertToFormat(QImage::Format_RGB888);

    int w = rgb.width();
    int h = rgb.height();

    // 旧实现：手写双 for + 另一张 QImage 完成 RGB→BGR；性能差，且依赖 QImage 的 4 字节对齐 bytesPerLine()，
    // 当 width*3 不能被 4 整除时（例如 810×1080：2430 → 2432），下游一旦把 stride 当作 width*3
    // 处理，就会逐行错位。
    // 现实现：
    //   1) rgbView 用 Qt 的 bytesPerLine() 作为 step，零拷贝读取；
    //   2) cv::cvtColor(RGB->BGR) 输出常规构建下 step == w*3 的紧凑 BGR；
    //   3) 把紧凑 BGR 与其 step 一并透传给 yi_infer_bgr，下游 yolo::Image 按 line_size 解释行步长，
    //      与 GPU letterbox 内核读取行字节数严格一致。
    cv::Mat rgbView(h, w, CV_8UC3, rgb.bits(), static_cast<size_t>(rgb.bytesPerLine()));
    cv::Mat bgr;
    cv::cvtColor(rgbView, bgr, cv::COLOR_RGB2BGR);

    double cpuPreMs = timer.nsecsElapsed() / 1.0e6;

    yi_box*       boxes = nullptr;
    int           count = 0;
    yi_frame_stat stat{};

    int rc = yi_infer_bgr(m_handle, bgr.data, w, h,
                          static_cast<int>(bgr.step),
                          &boxes, &count, &stat);

    if (rc != 0) {
        emit errorOccurred(QStringLiteral("推理失败: %1").arg(QString::fromUtf8(yi_last_error())));
        m_busy = false;
        return;
    }

    timer.restart();

    QVector<Detection> dets;
    dets.reserve(count);
    for (int i = 0; i < count; ++i) {
        const yi_box& b = boxes[i];
        Detection d;
        d.left       = b.left;
        d.top        = b.top;
        d.right      = b.right;
        d.bottom     = b.bottom;
        d.confidence = b.confidence;
        d.classLabel = b.class_label;
        d.hasMask    = b.has_mask != 0;
        d.maskX      = b.mask_x;
        d.maskY      = b.mask_y;
        d.maskW      = b.mask_w;
        d.maskH      = b.mask_h;
        if (d.hasMask && b.mask_data && b.mask_w > 0 && b.mask_h > 0) {
            d.maskData = QByteArray(reinterpret_cast<const char*>(b.mask_data),
                                   b.mask_w * b.mask_h);
        }
        if (b.num_kpts > 0 && b.kpts) {
            d.hasKpts = true;
            d.numKpts = b.num_kpts;
            d.kptsData = QByteArray(reinterpret_cast<const char*>(b.kpts),
                                    b.num_kpts * 3 * sizeof(float));
        }
        if (b.is_obb) {
            d.hasOBB   = true;
            d.obbCx    = b.cx;
            d.obbCy    = b.cy;
            d.obbOw    = b.ow;
            d.obbOh    = b.oh;
            d.obbAngle = b.angle;
        }
        dets.append(d);
    }

    double cpuPostMs = timer.nsecsElapsed() / 1.0e6;

    // 真实预/后处理 = DLL 内统计 + Qt 端 CPU 操作
    double preMs   = stat.preprocess_ms  + cpuPreMs;
    double postMs  = stat.postprocess_ms + cpuPostMs;
    double infMs   = stat.inference_ms;
    double totalMs = preMs + infMs + postMs;

    emit resultReady(frame, dets, preMs, infMs, postMs, totalMs);
    m_busy = false;
}

} // namespace trtforge
