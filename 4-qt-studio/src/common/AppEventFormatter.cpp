#include "common/AppEventFormatter.h"

#include "TRTLogger.h"

namespace trtforge {

QString AppEventFormatter::logColor(AppLogLevel level, const QString& defaultColor)
{
    switch (level) {
        case AppLogLevel::Debug: return QStringLiteral("#5A7896");
        case AppLogLevel::Info: return defaultColor;
        case AppLogLevel::Warn: return QStringLiteral("#FFB020");
        case AppLogLevel::Error: return QStringLiteral("#FF4D6D");
        case AppLogLevel::Stage: return QStringLiteral("#1FB6FF");
        case AppLogLevel::Success: return QStringLiteral("#22E27F");
    }
    return defaultColor;
}

QString AppEventFormatter::logPrefix(AppLogLevel level)
{
    switch (level) {
        case AppLogLevel::Debug: return QStringLiteral("[详细]");
        case AppLogLevel::Info: return QStringLiteral("[信息]");
        case AppLogLevel::Warn: return QStringLiteral("[警告]");
        case AppLogLevel::Error: return QStringLiteral("[错误]");
        case AppLogLevel::Stage: return QStringLiteral("[阶段]");
        case AppLogLevel::Success: return QStringLiteral("[成功]");
    }
    return QStringLiteral("[信息]");
}

AppLogLevel AppEventFormatter::fromTrtLogLevel(int level)
{
    switch (static_cast<LogLevel>(level)) {
        case LogLevel::Verbose: return AppLogLevel::Debug;
        case LogLevel::Info: return AppLogLevel::Info;
        case LogLevel::Warning: return AppLogLevel::Warn;
        case LogLevel::Error: return AppLogLevel::Error;
        case LogLevel::Stage: return AppLogLevel::Stage;
        case LogLevel::Success: return AppLogLevel::Success;
    }
    return AppLogLevel::Info;
}

int AppEventFormatter::toTrtLogLevel(AppLogLevel level)
{
    switch (level) {
        case AppLogLevel::Debug: return static_cast<int>(LogLevel::Verbose);
        case AppLogLevel::Info: return static_cast<int>(LogLevel::Info);
        case AppLogLevel::Warn: return static_cast<int>(LogLevel::Warning);
        case AppLogLevel::Error: return static_cast<int>(LogLevel::Error);
        case AppLogLevel::Stage: return static_cast<int>(LogLevel::Stage);
        case AppLogLevel::Success: return static_cast<int>(LogLevel::Success);
    }
    return static_cast<int>(LogLevel::Info);
}

} // namespace trtforge
