#pragma once

#include <QString>

namespace trtforge {

class AnomalyInferWorker;

class AnomalyInferenceAdapter
{
public:
    static void loadEngineAsync(AnomalyInferWorker* worker,
                                const QString& enginePath,
                                const QString& metadataPath,
                                const QString& modelType,
                                float imageThreshold,
                                float pixelThreshold);
};

} // namespace trtforge
