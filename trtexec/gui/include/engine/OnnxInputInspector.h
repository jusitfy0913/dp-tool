#pragma once

#include <QString>
#include <QVector>

namespace trtforge {

struct OnnxInputInfo
{
    bool ok { false };
    QString error;
    QString name;
    QVector<int> dims;

    bool isFourDim() const { return dims.size() == 4; }
    int channel(int fallback = 3) const;
    int height(int fallback = 640) const;
    int width(int fallback = 640) const;
};

class OnnxInputInspector
{
public:
    static OnnxInputInfo inspect(const QString& onnxPath);
};

} // namespace trtforge
