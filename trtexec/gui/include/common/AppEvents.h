#pragma once

#include <QJsonObject>
#include <QString>

namespace trtforge {

enum class AppLogLevel {
    Debug = 0,
    Info,
    Warn,
    Error,
    Stage,
    Success,
};

enum class AppEventSource {
    Engine,
    Training,
    Inference,
    Annotation,
};

struct AppLogEvent
{
    AppLogLevel level { AppLogLevel::Info };
    AppEventSource source { AppEventSource::Engine };
    QString message;
};

struct AppProgressEvent
{
    AppEventSource source { AppEventSource::Engine };
    QString stage;
    int percent { -1 };
    QJsonObject detail;
};

} // namespace trtforge
