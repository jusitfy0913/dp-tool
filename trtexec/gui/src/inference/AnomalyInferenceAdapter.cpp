#include "inference/AnomalyInferenceAdapter.h"

#include "AnomalyInferWorker.h"

#include <QMetaObject>

namespace trtforge {

void AnomalyInferenceAdapter::loadEngineAsync(AnomalyInferWorker* worker,
                                              const QString& enginePath,
                                              const QString& metadataPath,
                                              const QString& modelType,
                                              float imageThreshold,
                                              float pixelThreshold)
{
    if (!worker) {
        return;
    }
    QMetaObject::invokeMethod(worker, "loadEngine", Qt::QueuedConnection,
                              Q_ARG(QString, enginePath),
                              Q_ARG(QString, metadataPath),
                              Q_ARG(QString, modelType),
                              Q_ARG(float, imageThreshold),
                              Q_ARG(float, pixelThreshold));
}

} // namespace trtforge
