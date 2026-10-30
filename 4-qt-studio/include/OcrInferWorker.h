#pragma once

#include <QObject>
#include <QImage>
#include <QPointF>
#include <QString>
#include <QVector>
#include <atomic>

#include "ocr_infer_c.h"

namespace trtforge {

struct OcrTextLine {
    int index = 0;
    QVector<QPointF> polygon;
    QString text;
    float score = 0.0f;
    int angle = -1;
};

struct OcrResult {
    QImage overlayImage;
    QVector<OcrTextLine> lines;
    QString rawJson;
    QString device;
    QString precision;
    QString runMode;
    bool mkldnnEnabled = false;
    int cpuThreads = 0;
    int recBatchSize = 0;
    double ioMs = 0.0;
    double pipelineMs = 0.0;
    double resultMs = 0.0;
    double totalMs = 0.0;
};

struct OcrConfig {
    QString detModelDir;
    QString recModelDir;
    QString textlineOrientationModelDir;
    QString docOrientationModelDir;
    QString docUnwarpingModelDir;
    QString device = QStringLiteral("cpu");
    QString precision = QStringLiteral("fp32");
    bool enableMkldnn = true;
    bool useTextlineOrientation = false;
    bool useDocOrientationClassify = false;
    bool useDocUnwarping = false;
    int cpuThreads = 8;
    int recBatchSize = 6;
    float detThresh = 0.3f;
    float detBoxThresh = 0.6f;
    float detUnclipRatio = 1.5f;
    float recScoreThresh = 0.0f;
};

class OcrInferWorker : public QObject
{
    Q_OBJECT
public:
    explicit OcrInferWorker(QObject* parent = nullptr);
    ~OcrInferWorker() override;

    void unloadEngine();
    bool isLoaded() const { return m_handle != nullptr; }

public slots:
    bool loadEngine(trtforge::OcrConfig config);
    void onFrame(QImage frame, qint64 timestampMs);

signals:
    void engineLoaded();
    void engineLoadFailed(QString msg);
    void resultReady(trtforge::OcrResult result,
                     double preprocessMs, double inferMs, double postMs, double totalMs);
    void errorOccurred(QString msg);

private:
    oi_handle m_handle = nullptr;
    std::atomic<bool> m_busy{false};
};

} // namespace trtforge

Q_DECLARE_METATYPE(trtforge::OcrTextLine)
Q_DECLARE_METATYPE(trtforge::OcrResult)
Q_DECLARE_METATYPE(trtforge::OcrConfig)
