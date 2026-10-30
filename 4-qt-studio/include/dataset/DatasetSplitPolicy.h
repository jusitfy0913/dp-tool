#pragma once

#include "AnnotationProject.h"
#include "common/AppSettings.h"

#include <QVector>

namespace trtforge {

class DatasetSplitPolicy
{
public:
    static QVector<DatasetSplit> calculateSplits(int itemCount, const SplitRatio& ratio);
    static void applyToProject(AnnotationProject& project, const SplitRatio& ratio);
};

} // namespace trtforge
