#include "training/TrainEventParser.h"

#include <QJsonDocument>
#include <QJsonParseError>

namespace trtforge {

namespace {

constexpr const char* kJsonPrefix = "TRTFORGE_JSON ";

} // namespace

ParsedTrainEvent TrainEventParser::parseLine(const QString& line, bool isError)
{
    ParsedTrainEvent parsed;
    parsed.text = line;
    parsed.isError = isError;

    if (line.startsWith(QLatin1String(kJsonPrefix))) {
        const QByteArray json = line.mid(QString::fromLatin1(kJsonPrefix).size()).toUtf8();
        QJsonParseError err {};
        const QJsonDocument doc = QJsonDocument::fromJson(json, &err);
        if (err.error == QJsonParseError::NoError && doc.isObject()) {
            parsed.structured = true;
            parsed.event = doc.object();
            parsed.text.clear();
        }
    }

    return parsed;
}

} // namespace trtforge
