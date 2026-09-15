#pragma once

#include <QObject>
#include <QImage>
#include <QMutex>
#include <QVector>
#include <atomic>

#include "yolo_infer_c.h"

namespace trtforge {

struct Detection {
    float left, top, right, bottom;
    float confidence;
    int   classLabel;
    bool  hasMask;
    int   maskX, maskY, maskW, maskH;
    QByteArray maskData;
    // Pose keypoints (only set when running YI_POSE engine)
    bool       hasKpts = false;
    int        numKpts = 0;
    QByteArray kptsData; // 连续 numKpts*3 floats
    // OBB (only set when running YI_OBB engine)
    bool  hasOBB   = false;
    float obbCx    = 0.f;
    float obbCy    = 0.f;
    float obbOw    = 0.f;
    float obbOh    = 0.f;
    float obbAngle = 0.f; // radians
};

class InferWorker : public QObject
{
    Q_OBJECT
public:
    explicit InferWorker(QObject* parent = nullptr);
    ~InferWorker() override;

    void unloadEngine();
    bool isLoaded() const { return m_handle != nullptr; }

public slots:
    bool loadEngine(const QString& enginePath, int task,
                    float confThr, float nmsThr);
    void onFrame(QImage frame, qint64 timestampMs);

signals:
    void engineLoaded();
    void engineLoadFailed(QString msg);
    void resultReady(QImage frame, QVector<trtforge::Detection> dets,
                     double preprocessMs, double inferMs, double postMs, double totalMs);
    void errorOccurred(QString msg);

private:
    yi_handle             m_handle = nullptr;
    std::atomic<bool>     m_busy{false};
};

} // namespace trtforge

Q_DECLARE_METATYPE(QVector<trtforge::Detection>)
