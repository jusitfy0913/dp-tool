#include "common/AppFileLogger.h"

#include <QCoreApplication>
#include <QDate>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMutex>
#include <QMutexLocker>
#include <QTextStream>
#include <QStringConverter>

namespace trtforge {

namespace {

QMutex& logMutex()
{
    static QMutex mutex;
    return mutex;
}

QString guiRootDir()
{
    QDir dir(QCoreApplication::applicationDirPath());
    if (dir.dirName().compare(QStringLiteral("bin"), Qt::CaseInsensitive) == 0) {
        dir.cdUp();
    } else if (dir.dirName().compare(QStringLiteral("Release"), Qt::CaseInsensitive) == 0 ||
               dir.dirName().compare(QStringLiteral("Debug"), Qt::CaseInsensitive) == 0) {
        dir.cdUp();
        if (dir.dirName().compare(QStringLiteral("bin"), Qt::CaseInsensitive) == 0) {
            dir.cdUp();
        }
    }
    return dir.absolutePath();
}

} // namespace

void AppFileLogger::append(const AppLogEvent& event)
{
    if (event.message.trimmed().isEmpty()) {
        return;
    }

    QMutexLocker locker(&logMutex());

    const QString path = currentLogFilePath();
    QFile file(path);
    QDir().mkpath(QFileInfo(path).absolutePath());
    if (!file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
        return;
    }

    QTextStream out(&file);
    out.setEncoding(QStringConverter::Utf8);
    out << QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss.zzz"))
        << " [" << sourceName(event.source) << "]"
        << " [" << levelName(event.level) << "] "
        << event.message.trimmed()
        << '\n';
}

QString AppFileLogger::currentLogFilePath()
{
    const QString day = QDate::currentDate().toString(QStringLiteral("yyyy-MM-dd"));
    return QDir(logRootDir()).filePath(QStringLiteral("%1/trtforge_%1.log").arg(day));
}

QString AppFileLogger::logRootDir()
{
    return QDir(guiRootDir()).filePath(QStringLiteral("logs"));
}

QString AppFileLogger::sourceName(AppEventSource source)
{
    switch (source) {
        case AppEventSource::Engine: return QStringLiteral("ENGINE");
        case AppEventSource::Training: return QStringLiteral("TRAINING");
        case AppEventSource::Inference: return QStringLiteral("INFERENCE");
        case AppEventSource::Annotation: return QStringLiteral("ANNOTATION");
    }
    return QStringLiteral("APP");
}

QString AppFileLogger::levelName(AppLogLevel level)
{
    switch (level) {
        case AppLogLevel::Debug: return QStringLiteral("DEBUG");
        case AppLogLevel::Info: return QStringLiteral("INFO");
        case AppLogLevel::Warn: return QStringLiteral("WARN");
        case AppLogLevel::Error: return QStringLiteral("ERROR");
        case AppLogLevel::Stage: return QStringLiteral("STAGE");
        case AppLogLevel::Success: return QStringLiteral("SUCCESS");
    }
    return QStringLiteral("INFO");
}

} // namespace trtforge
