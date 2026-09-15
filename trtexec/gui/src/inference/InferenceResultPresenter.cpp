#include "inference/InferenceResultPresenter.h"

namespace trtforge {

QString InferenceResultPresenter::anomalyStatusStyle(bool isAnomaly)
{
    return isAnomaly ? QStringLiteral("color:#FF4D6D;")
                     : QStringLiteral("color:#3AE675;");
}

QString InferenceResultPresenter::milliseconds(double value)
{
    return QString::number(value, 'f', 1) + QStringLiteral(" ms");
}

QString InferenceResultPresenter::detectionCount(int count)
{
    return QString::number(count);
}

} // namespace trtforge
