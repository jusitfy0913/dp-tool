#pragma once

#include <QString>

namespace trtforge {

class InferWorker;

class YoloInferenceAdapter
{
public:
    static void loadEngineAsync(InferWorker* worker,
                                const QString& enginePath,
                                int task,
                                float confThreshold,
                                float nmsThreshold);
};

} // namespace trtforge
