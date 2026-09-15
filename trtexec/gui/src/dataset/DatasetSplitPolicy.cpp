#include "dataset/DatasetSplitPolicy.h"

#include <QHash>
#include <QString>
#include <algorithm>

namespace trtforge {

QVector<DatasetSplit> DatasetSplitPolicy::calculateSplits(int itemCount, const SplitRatio& ratio)
{
    QVector<DatasetSplit> result;
    if (itemCount <= 0 || !ratio.isValid()) {
        return result;
    }

    QVector<int> order;
    order.reserve(itemCount);
    for (int i = 0; i < itemCount; ++i) {
        order.append(i);
    }
    std::sort(order.begin(), order.end(), [](int a, int b) {
        return qHash(QString::number(a) + "_2026") < qHash(QString::number(b) + "_2026");
    });

    result.resize(itemCount);
    const int trainN = itemCount * ratio.train / 100;
    const int valN = itemCount * ratio.val / 100;
    for (int i = 0; i < itemCount; ++i) {
        result[order[i]] = i < trainN ? DatasetSplit::Train : (i < trainN + valN ? DatasetSplit::Val : DatasetSplit::Test);
    }
    return result;
}

void DatasetSplitPolicy::applyToProject(AnnotationProject& project, const SplitRatio& ratio)
{
    const QVector<DatasetSplit> splits = calculateSplits(project.images.size(), ratio);
    if (splits.size() != project.images.size()) {
        return;
    }

    for (int i = 0; i < project.images.size(); ++i) {
        project.images[i].split = splits[i];
    }
    project.dirty = true;
}

} // namespace trtforge
