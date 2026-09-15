#pragma once

#include "OcrInferWorker.h"

namespace trtforge {

class OcrInferenceAdapter
{
public:
    static void loadEngineAsync(OcrInferWorker* worker, const OcrConfig& config);
};

} // namespace trtforge
