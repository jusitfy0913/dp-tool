#pragma once

#include "common/AppEvents.h"

#include <QString>

namespace trtforge {

class AppFileLogger
{
public:
    static void append(const AppLogEvent& event);
    static QString currentLogFilePath();

private:
    static QString logRootDir();
    static QString sourceName(AppEventSource source);
    static QString levelName(AppLogLevel level);
};

} // namespace trtforge
