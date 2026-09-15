#pragma once

#include <QObject>
#include <QJsonObject>
#include <QProcess>
#include <QStringList>

namespace trtforge {

class YoloTrainController : public QObject
{
    Q_OBJECT
public:
    enum class LauncherKind { Yolo, Anomalib };

    explicit YoloTrainController(QObject* parent = nullptr);
    ~YoloTrainController() override;

    QString pythonPath() const { return m_pythonPath; }
    void setPythonPath(const QString& path);
    void setLauncherKind(LauncherKind kind) { m_launcherKind = kind; }
    LauncherKind launcherKind() const { return m_launcherKind; }
    QString resolvePythonPath() const;
    QString launcherPath() const;
    bool isRunning() const;

    void start(const QStringList& arguments);
    void stop();

signals:
    void logLine(QString text, bool isError);
    void structuredEvent(QJsonObject event);
    void processStarted(QString program, QStringList arguments);
    void processFinished(int exitCode, QString reason);
    void processError(QString message);

private slots:
    void onReadyReadStdout();
    void onReadyReadStderr();
    void onFinished(int exitCode, QProcess::ExitStatus status);
    void onErrorOccurred(QProcess::ProcessError error);

private:
    void consumeLines(QByteArray& buffer, const QByteArray& bytes, bool isError);
    void handleLine(const QString& line, bool isError);

    QProcess* m_process { nullptr };
    QString m_pythonPath;
    LauncherKind m_launcherKind { LauncherKind::Yolo };
    QByteArray m_stdoutBuffer;
    QByteArray m_stderrBuffer;
};

} // namespace trtforge
