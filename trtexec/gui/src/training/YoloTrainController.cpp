#include "YoloTrainController.h"

#include "training/TrainEventParser.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QProcessEnvironment>
#include <QStandardPaths>

namespace trtforge {

namespace {

QString normalizePath(const QString& path)
{
    return QDir::toNativeSeparators(QFileInfo(path).absoluteFilePath());
}

QString firstExistingFile(const QStringList& paths)
{
    for (const QString& path : paths) {
        QFileInfo fi(path);
        if (fi.exists() && fi.isFile()) return normalizePath(path);
    }
    return {};
}

} // namespace

YoloTrainController::YoloTrainController(QObject* parent) : QObject(parent)
{
    m_process = new QProcess(this);
    m_process->setProcessChannelMode(QProcess::SeparateChannels);
    connect(m_process, &QProcess::readyReadStandardOutput, this, &YoloTrainController::onReadyReadStdout);
    connect(m_process, &QProcess::readyReadStandardError, this, &YoloTrainController::onReadyReadStderr);
    connect(m_process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
            this, &YoloTrainController::onFinished);
    connect(m_process, &QProcess::errorOccurred, this, &YoloTrainController::onErrorOccurred);
}

YoloTrainController::~YoloTrainController()
{
    stop();
}

void YoloTrainController::setPythonPath(const QString& path)
{
    m_pythonPath = QDir::toNativeSeparators(path.trimmed());
}

QString YoloTrainController::resolvePythonPath() const
{
    if (!m_pythonPath.trimmed().isEmpty() && QFileInfo::exists(m_pythonPath)) {
        return normalizePath(m_pythonPath);
    }

    const QString appDir = QCoreApplication::applicationDirPath();
    const QString cwd = QDir::currentPath();
    const QString condaPrefix = QString::fromLocal8Bit(qgetenv("CONDA_PREFIX"));
    const QString virtualEnv = QString::fromLocal8Bit(qgetenv("VIRTUAL_ENV"));

    QStringList candidates;
    candidates
        << appDir + "/../.venv/Scripts/python.exe"
        << appDir + "/../venv/Scripts/python.exe"
        << appDir + "/../runtime/python/python.exe"
        << cwd + "/.venv/Scripts/python.exe"
        << cwd + "/venv/Scripts/python.exe"
        << cwd + "/trtexec/gui/.venv/Scripts/python.exe"
        << cwd + "/trtexec/gui/venv/Scripts/python.exe"
        << cwd + "/trtexec/gui/runtime/python/python.exe";
    if (!virtualEnv.isEmpty()) candidates << virtualEnv + "/Scripts/python.exe";
    if (!condaPrefix.isEmpty()) candidates << condaPrefix + "/python.exe";

    QString found = firstExistingFile(candidates);
    if (!found.isEmpty()) return found;

    found = QStandardPaths::findExecutable("python");
    if (!found.isEmpty()) return normalizePath(found);
    found = QStandardPaths::findExecutable("python3");
    if (!found.isEmpty()) return normalizePath(found);
    found = QStandardPaths::findExecutable("py");
    if (!found.isEmpty()) return normalizePath(found);
    return {};
}

QString YoloTrainController::launcherPath() const
{
    const QString appDir = QCoreApplication::applicationDirPath();
    const QString cwd = QDir::currentPath();
    const QString launcherName = m_launcherKind == LauncherKind::Anomalib
        ? "tf_anomalib_launcher.py"
        : "tf_yolo_launcher.py";
    return firstExistingFile({
        appDir + "/training/scripts/" + launcherName,
        appDir + "/../training/scripts/" + launcherName,
        cwd + "/trtexec/gui/training/scripts/" + launcherName,
        cwd + "/training/scripts/" + launcherName,
    });
}

bool YoloTrainController::isRunning() const
{
    return m_process && m_process->state() != QProcess::NotRunning;
}

void YoloTrainController::start(const QStringList& arguments)
{
    if (isRunning()) {
        emit processError("已有训练/导出任务正在运行。");
        return;
    }

    const QString python = resolvePythonPath();
    if (python.isEmpty()) {
        emit processError("未找到可用 Python。请先选择 python.exe 或创建 .venv。");
        return;
    }
    const QString launcher = launcherPath();
    if (launcher.isEmpty()) {
        emit processError(m_launcherKind == LauncherKind::Anomalib
            ? "未找到 tf_anomalib_launcher.py。"
            : "未找到 tf_yolo_launcher.py。");
        return;
    }

    QStringList fullArgs;
    fullArgs << launcher;
    fullArgs << arguments;

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("PYTHONIOENCODING", "utf-8");
    env.insert("PYTHONUNBUFFERED", "1");
    m_process->setProcessEnvironment(env);
    m_process->setWorkingDirectory(QFileInfo(launcher).absolutePath());

    m_stdoutBuffer.clear();
    m_stderrBuffer.clear();
    emit processStarted(python, fullArgs);
    m_process->start(python, fullArgs);
}

void YoloTrainController::stop()
{
    if (!isRunning()) return;
    m_process->terminate();
    if (!m_process->waitForFinished(3000)) {
        m_process->kill();
    }
}

void YoloTrainController::onReadyReadStdout()
{
    consumeLines(m_stdoutBuffer, m_process->readAllStandardOutput(), false);
}

void YoloTrainController::onReadyReadStderr()
{
    consumeLines(m_stderrBuffer, m_process->readAllStandardError(), true);
}

void YoloTrainController::onFinished(int exitCode, QProcess::ExitStatus status)
{
    if (!m_stdoutBuffer.isEmpty()) {
        handleLine(QString::fromUtf8(m_stdoutBuffer).trimmed(), false);
        m_stdoutBuffer.clear();
    }
    if (!m_stderrBuffer.isEmpty()) {
        handleLine(QString::fromUtf8(m_stderrBuffer).trimmed(), true);
        m_stderrBuffer.clear();
    }
    emit processFinished(exitCode, status == QProcess::NormalExit ? "normal" : "crashed");
}

void YoloTrainController::onErrorOccurred(QProcess::ProcessError error)
{
    emit processError(QString("Python 子进程错误：%1").arg(static_cast<int>(error)));
}

void YoloTrainController::consumeLines(QByteArray& buffer, const QByteArray& bytes, bool isError)
{
    buffer.append(bytes);
    while (true) {
        const int idx = buffer.indexOf('\n');
        if (idx < 0) break;
        QByteArray line = buffer.left(idx);
        buffer.remove(0, idx + 1);
        if (line.endsWith('\r')) line.chop(1);
        handleLine(QString::fromUtf8(line), isError);
    }
}

void YoloTrainController::handleLine(const QString& line, bool isError)
{
    if (line.isEmpty()) return;
    const ParsedTrainEvent parsed = TrainEventParser::parseLine(line, isError);
    if (parsed.structured) {
        emit structuredEvent(parsed.event);
        return;
    }
    emit logLine(parsed.text, parsed.isError);
}

} // namespace trtforge
