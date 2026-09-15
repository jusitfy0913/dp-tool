#pragma once

#include <QObject>
#include <QImage>
#include <QString>
#include <atomic>

#include "anomalib_infer_c.h"

namespace trtforge {

struct AnomalyMetadata {
    QString modelType;
    QString normalization;
    QString outputContract;
    float imageThreshold = 0.5f;
    float pixelThreshold = 0.5f;
    float scoreMin = 0.0f;
    float scoreMax = 1.0f;
    float mapMin = 0.0f;
    float mapMax = 1.0f;
    int inputH = 256;
    int inputW = 256;

    QString summary() const;
};

struct AnomalyResult {
    QImage displayImage;       // legacy combined (unused in multi-canvas mode)
    QImage sourceImage;        // original input image
    QImage overlayImage;       // heatmap overlay on original
    QImage maskImage;          // prediction mask contours on original
    float rawScore = 0.0f;
    float normalizedScore = 0.0f;
    bool isAnomaly = false;
    QString statusText;
};

class AnomalyInferWorker : public QObject
{
    Q_OBJECT
public:
    explicit AnomalyInferWorker(QObject* parent = nullptr);
    ~AnomalyInferWorker() override;

    void unloadEngine();
    bool isLoaded() const { return m_handle != nullptr; }

public slots:
    bool loadEngine(const QString& enginePath,
                    const QString& metadataPath,
                    const QString& modelType,
                    float imageThresholdOverride = -1.0f,
                    float pixelThresholdOverride = -1.0f);
    void onFrame(QImage frame, qint64 timestampMs);

signals:
    void engineLoaded();
    void engineLoadFailed(QString msg);
    void metadataReady(trtforge::AnomalyMetadata meta);
    void resultReady(trtforge::AnomalyResult result,
                     double preprocessMs, double inferMs, double postMs, double totalMs);
    void errorOccurred(QString msg);

private:
    ai_handle m_handle = nullptr;
    AnomalyMetadata m_meta;
    std::atomic<bool> m_busy{false};
};

} // namespace trtforge

Q_DECLARE_METATYPE(trtforge::AnomalyMetadata)
Q_DECLARE_METATYPE(trtforge::AnomalyResult)
