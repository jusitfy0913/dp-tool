#include "TRTLogger.h"

#include <QString>

namespace trtforge {

TRTLogger::TRTLogger(QObject* parent) : QObject(parent) {}

void TRTLogger::log(Severity severity, const char* msg) noexcept
{
    if (static_cast<int>(severity) > static_cast<int>(m_min)) {
        return;
    }
    int lvl = static_cast<int>(LogLevel::Info);
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:   lvl = static_cast<int>(LogLevel::Error);   break;
        case Severity::kWARNING: lvl = static_cast<int>(LogLevel::Warning); break;
        case Severity::kINFO:    lvl = static_cast<int>(LogLevel::Info);    break;
        case Severity::kVERBOSE: lvl = static_cast<int>(LogLevel::Verbose); break;
        default:                 lvl = static_cast<int>(LogLevel::Info);    break;
    }
    QString text = msg ? QString::fromUtf8(msg) : QString();
    emit logArrived(lvl, text);
}

void TRTLogger::emitInfo(const QString& msg)    { emit logArrived(static_cast<int>(LogLevel::Info),    msg); }
void TRTLogger::emitStage(const QString& msg)   { emit logArrived(static_cast<int>(LogLevel::Stage),   msg); }
void TRTLogger::emitWarning(const QString& msg) { emit logArrived(static_cast<int>(LogLevel::Warning), msg); }
void TRTLogger::emitError(const QString& msg)   { emit logArrived(static_cast<int>(LogLevel::Error),   msg); }
void TRTLogger::emitSuccess(const QString& msg) { emit logArrived(static_cast<int>(LogLevel::Success), msg); }

} // namespace trtforge
