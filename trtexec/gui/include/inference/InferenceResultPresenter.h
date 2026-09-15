#pragma once

#include <QString>

namespace trtforge {

class InferenceResultPresenter
{
public:
    static QString anomalyStatusStyle(bool isAnomaly);
    static QString milliseconds(double value);
    static QString detectionCount(int count);
};

} // namespace trtforge
