#include "OcrInferWorker.h"

#include <QByteArray>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace trtforge {

namespace {

QString lastOcrError()
{
    return QString::fromUtf8(oi_last_error());
}

QImage rgbViewToImage(const unsigned char* data, int w, int h, int stride)
{
    if (!data || w <= 0 || h <= 0 || stride <= 0) return {};
    return QImage(data, w, h, stride, QImage::Format_RGB888).copy();
}

OcrResult toQtResult(const oi_result* src)
{
    OcrResult result;
    if (!src) return result;

    result.overlayImage = rgbViewToImage(src->overlay_rgb,
                                         src->overlay_w,
                                         src->overlay_h,
                                         src->overlay_stride);
    result.rawJson = QString::fromUtf8(src->raw_json_utf8 ? src->raw_json_utf8 : "");
    result.lines.reserve(src->line_count);
    for (int i = 0; i < src->line_count; ++i) {
        const oi_text_line& line = src->lines[i];
        OcrTextLine item;
        item.index = line.index;
        item.text = QString::fromUtf8(line.text_utf8 ? line.text_utf8 : "");
        item.score = line.rec_score;
        item.angle = line.textline_angle;
        for (int j = 0; j < line.point_count; ++j) {
            const float x = line.xs ? line.xs[j] : 0.0f;
            const float y = line.ys ? line.ys[j] : 0.0f;
            item.polygon.append(QPointF(x, y));
        }
        result.lines.append(item);
    }
    return result;
}

} // namespace

OcrInferWorker::OcrInferWorker(QObject* parent)
    : QObject(parent)
{
    qRegisterMetaType<trtforge::OcrTextLine>("trtforge::OcrTextLine");
    qRegisterMetaType<trtforge::OcrResult>("trtforge::OcrResult");
    qRegisterMetaType<trtforge::OcrConfig>("trtforge::OcrConfig");
}

OcrInferWorker::~OcrInferWorker()
{
    unloadEngine();
}

bool OcrInferWorker::loadEngine(OcrConfig config)
{
    unloadEngine();

    const QByteArray det = config.detModelDir.toUtf8();
    const QByteArray rec = config.recModelDir.toUtf8();
    const QByteArray lineOri = config.textlineOrientationModelDir.toUtf8();
    const QByteArray docOri = config.docOrientationModelDir.toUtf8();
    const QByteArray docUnwarp = config.docUnwarpingModelDir.toUtf8();
    const QByteArray device = config.device.toUtf8();
    const QByteArray precision = config.precision.toUtf8();

    oi_config c{};
    c.det_model_dir_utf8 = det.constData();
    c.rec_model_dir_utf8 = rec.constData();
    c.textline_orientation_model_dir_utf8 = lineOri.isEmpty() ? nullptr : lineOri.constData();
    c.doc_orientation_model_dir_utf8 = docOri.isEmpty() ? nullptr : docOri.constData();
    c.doc_unwarping_model_dir_utf8 = docUnwarp.isEmpty() ? nullptr : docUnwarp.constData();
    c.device_utf8 = device.constData();
    c.precision_utf8 = precision.constData();
    c.enable_mkldnn = config.enableMkldnn ? 1 : 0;
    c.use_textline_orientation = config.useTextlineOrientation ? 1 : 0;
    c.use_doc_orientation_classify = config.useDocOrientationClassify ? 1 : 0;
    c.use_doc_unwarping = config.useDocUnwarping ? 1 : 0;
    c.cpu_threads = config.cpuThreads;
    c.rec_batch_size = config.recBatchSize;
    c.det_thresh = config.detThresh;
    c.det_box_thresh = config.detBoxThresh;
    c.det_unclip_ratio = config.detUnclipRatio;
    c.rec_score_thresh = config.recScoreThresh;

    m_handle = oi_create(&c);
    if (!m_handle) {
        emit engineLoadFailed(QStringLiteral("OCR DLL 加载失败：%1").arg(lastOcrError()));
        return false;
    }
    emit engineLoaded();
    return true;
}

void OcrInferWorker::unloadEngine()
{
    if (m_handle) {
        oi_destroy(m_handle);
        m_handle = nullptr;
    }
    m_busy = false;
}

void OcrInferWorker::onFrame(QImage frame, qint64 /*timestampMs*/)
{
    if (!m_handle) return;

    bool expected = false;
    if (!m_busy.compare_exchange_strong(expected, true)) return;

    QImage rgb = frame;
    if (rgb.format() != QImage::Format_RGB888) {
        rgb = rgb.convertToFormat(QImage::Format_RGB888);
    }
    if (rgb.isNull()) {
        emit errorOccurred(QStringLiteral("输入帧为空"));
        m_busy = false;
        return;
    }

    cv::Mat rgbView(rgb.height(), rgb.width(), CV_8UC3,
                    const_cast<uchar*>(rgb.constBits()),
                    static_cast<size_t>(rgb.bytesPerLine()));
    cv::Mat bgr;
    cv::cvtColor(rgbView, bgr, cv::COLOR_RGB2BGR);

    const oi_result* dllResult = nullptr;
    oi_frame_stat stat{};
    const int rc = oi_infer_bgr(m_handle,
                                bgr.data,
                                bgr.cols,
                                bgr.rows,
                                static_cast<int>(bgr.step),
                                &dllResult,
                                &stat);
    if (rc != OI_OK || !dllResult) {
        emit errorOccurred(QStringLiteral("OCR DLL 推理失败：%1").arg(lastOcrError()));
        m_busy = false;
        return;
    }

    OcrResult result = toQtResult(dllResult);
    result.device = QString::fromUtf8(stat.device_utf8);
    result.precision = QString::fromUtf8(stat.precision_utf8);
    result.runMode = QString::fromUtf8(stat.run_mode_utf8);
    result.mkldnnEnabled = stat.mkldnn_enabled != 0;
    result.cpuThreads = stat.cpu_threads;
    result.recBatchSize = stat.rec_batch_size;
    result.ioMs = stat.io_ms;
    result.pipelineMs = stat.pipeline_ms;
    result.resultMs = stat.result_ms;
    result.totalMs = stat.total_ms;

    emit resultReady(result,
                     stat.preprocess_ms,
                     stat.inference_ms,
                     stat.postprocess_ms,
                     stat.total_ms);
    m_busy = false;
}

} // namespace trtforge
