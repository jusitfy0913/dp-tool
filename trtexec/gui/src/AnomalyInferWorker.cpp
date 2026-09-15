#include "AnomalyInferWorker.h"

#include <QByteArray>

namespace trtforge {

namespace {

QString lastAiError()
{
    return QString::fromUtf8(ai_last_error());
}

QString fixedString(const char* s)
{
    return QString::fromUtf8(s ? s : "").trimmed();
}

QImage rgbViewToImage(const unsigned char* data, int w, int h, int stride)
{
    if (!data || w <= 0 || h <= 0 || stride <= 0) return {};
    return QImage(data, w, h, stride, QImage::Format_RGB888).copy();
}

AnomalyMetadata toQtMetadata(const ai_metadata& src)
{
    AnomalyMetadata meta;
    meta.modelType = fixedString(src.model_type);
    meta.normalization = fixedString(src.normalization);
    meta.outputContract = fixedString(src.output_contract);
    meta.imageThreshold = src.image_threshold;
    meta.pixelThreshold = src.pixel_threshold;
    meta.scoreMin = src.score_min;
    meta.scoreMax = src.score_max;
    meta.mapMin = src.map_min;
    meta.mapMax = src.map_max;
    meta.inputH = src.input_h;
    meta.inputW = src.input_w;
    return meta;
}

} // namespace

QString AnomalyMetadata::summary() const
{
    return QStringLiteral("%1 · 输入 %2×%3 · normalization=%4 · image_threshold=%5 · pixel_threshold=%6")
        .arg(modelType.isEmpty() ? QStringLiteral("Auto") : modelType)
        .arg(inputW).arg(inputH)
        .arg(normalization.isEmpty() ? QStringLiteral("auto") : normalization)
        .arg(QString::number(imageThreshold, 'f', 3))
        .arg(QString::number(pixelThreshold, 'f', 3));
}

AnomalyInferWorker::AnomalyInferWorker(QObject* parent)
    : QObject(parent)
{
    qRegisterMetaType<trtforge::AnomalyMetadata>("trtforge::AnomalyMetadata");
    qRegisterMetaType<trtforge::AnomalyResult>("trtforge::AnomalyResult");
}

AnomalyInferWorker::~AnomalyInferWorker()
{
    unloadEngine();
}

bool AnomalyInferWorker::loadEngine(const QString& enginePath,
                                    const QString& metadataPath,
                                    const QString& modelType,
                                    float imageThresholdOverride,
                                    float pixelThresholdOverride)
{
    unloadEngine();

    const QByteArray engineUtf8 = enginePath.toUtf8();
    const QByteArray metadataUtf8 = metadataPath.toUtf8();
    const QByteArray modelUtf8 = modelType.toUtf8();

    m_handle = ai_create(engineUtf8.constData(),
                         metadataUtf8.constData(),
                         modelUtf8.constData(),
                         imageThresholdOverride,
                         pixelThresholdOverride);
    if (!m_handle) {
        emit engineLoadFailed(QStringLiteral("异常检测 DLL 加载失败：%1").arg(lastAiError()));
        return false;
    }

    ai_metadata meta{};
    const int rc = ai_get_metadata(m_handle, &meta);
    if (rc != AI_OK) {
        const QString msg = QStringLiteral("异常检测 metadata 读取失败：%1").arg(lastAiError());
        unloadEngine();
        emit engineLoadFailed(msg);
        return false;
    }
    m_meta = toQtMetadata(meta);
    emit metadataReady(m_meta);
    emit engineLoaded();
    return true;
}

void AnomalyInferWorker::unloadEngine()
{
    if (m_handle) {
        ai_destroy(m_handle);
        m_handle = nullptr;
    }
    m_busy = false;
}

void AnomalyInferWorker::onFrame(QImage frame, qint64 /*timestampMs*/)
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

    const ai_result* dllResult = nullptr;
    ai_frame_stat stat{};
    const int rc = ai_infer_rgb(m_handle,
                                rgb.constBits(),
                                rgb.width(),
                                rgb.height(),
                                static_cast<int>(rgb.bytesPerLine()),
                                &dllResult,
                                &stat);
    if (rc != AI_OK || !dllResult) {
        emit errorOccurred(QStringLiteral("异常检测 DLL 推理失败：%1").arg(lastAiError()));
        m_busy = false;
        return;
    }

    AnomalyResult result;
    result.sourceImage = rgbViewToImage(dllResult->source_rgb,
                                        dllResult->source_w,
                                        dllResult->source_h,
                                        dllResult->source_stride);
    result.overlayImage = rgbViewToImage(dllResult->overlay_rgb,
                                         dllResult->overlay_w,
                                         dllResult->overlay_h,
                                         dllResult->overlay_stride);
    result.maskImage = rgbViewToImage(dllResult->mask_rgb,
                                      dllResult->mask_w,
                                      dllResult->mask_h,
                                      dllResult->mask_stride);
    result.displayImage = result.overlayImage;
    result.rawScore = dllResult->raw_score;
    result.normalizedScore = dllResult->normalized_score;
    result.isAnomaly = dllResult->is_anomaly != 0;
    result.statusText = result.isAnomaly ? QStringLiteral("NG") : QStringLiteral("OK");

    emit resultReady(result,
                     stat.preprocess_ms,
                     stat.inference_ms,
                     stat.postprocess_ms,
                     stat.total_ms);
    m_busy = false;
}

} // namespace trtforge
