#pragma once

#include <QJsonObject>
#include <QString>

namespace trtforge {

struct ParsedTrainEvent
{
    bool structured { false };
    QJsonObject event;
    QString text;
    bool isError { false };
};

class TrainEventParser
{
public:
    static ParsedTrainEvent parseLine(const QString& line, bool isError);
};

} // namespace trtforge
