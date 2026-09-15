#include "inference/OcrInferenceAdapter.h"

#include <QMetaObject>

namespace trtforge {

void OcrInferenceAdapter::loadEngineAsync(OcrInferWorker* worker, const OcrConfig& config)
{
    if (!worker) {
        return;
    }
    QMetaObject::invokeMethod(worker, "loadEngine", Qt::QueuedConnection,
                              Q_ARG(trtforge::OcrConfig, config));
}

} // namespace trtforge
