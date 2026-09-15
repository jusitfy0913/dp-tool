#include "inference/YoloInferenceAdapter.h"

#include "InferWorker.h"

#include <QMetaObject>

namespace trtforge {

void YoloInferenceAdapter::loadEngineAsync(InferWorker* worker,
                                           const QString& enginePath,
                                           int task,
                                           float confThreshold,
                                           float nmsThreshold)
{
    if (!worker) {
        return;
    }
    QMetaObject::invokeMethod(worker, "loadEngine", Qt::QueuedConnection,
                              Q_ARG(QString, enginePath),
                              Q_ARG(int, task),
                              Q_ARG(float, confThreshold),
                              Q_ARG(float, nmsThreshold));
}

} // namespace trtforge
