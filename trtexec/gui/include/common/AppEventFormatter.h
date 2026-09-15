#pragma once

#include "common/AppEvents.h"

#include <QString>

namespace trtforge {

class AppEventFormatter
{
public:
    static QString logColor(AppLogLevel level, const QString& defaultColor = QString());
    static QString logPrefix(AppLogLevel level);
    static AppLogLevel fromTrtLogLevel(int level);
    static int toTrtLogLevel(AppLogLevel level);
};

} // namespace trtforge
