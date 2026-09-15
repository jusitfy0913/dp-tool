#pragma once

// ----------------------------------------------------------------------------
// 说明：TensorRT 8.6 公共头中尚未提供 nvinfer1::IProgressMonitor，因此该类
// 在 GUI 端作为"进度估算器"工作：在主线程持有 QTimer，根据 BuilderWorker
// 发出的关键日志切换阶段，并在阶段内做平滑插值，让进度条自然增长。
// ----------------------------------------------------------------------------

#include <QObject>
#include <QString>
#include <QTimer>
#include <QElapsedTimer>

namespace trtforge {

class TRTProgressMonitor : public QObject
{
    Q_OBJECT
public:
    enum class Stage { Idle = 0, Parse, Optimize, Tune, Serialize, Save, Done, Failed };

    explicit TRTProgressMonitor(QObject* parent = nullptr);

    void start();
    void stop();
    void reset();

    // 由 MainWindow 在收到日志时调用
    void onLog(int level, const QString& text);
    // 构建结束时强制收尾
    void finish(bool ok);

    Stage stage() const { return m_stage; }
    int progressPermille() const { return m_pm; }   // 0..1000
    qint64 elapsedMs() const { return m_timer.isValid() ? m_timer.elapsed() : 0; }

signals:
    void progressChanged(int permille);              // 0..1000
    void stageChanged(int stage);                    // Stage
    void etaChanged(qint64 etaMs);

private slots:
    void onTick();

private:
    void setStage(Stage s);
    void setPermille(int pm);

    QTimer  m_tick;
    QElapsedTimer m_timer;
    Stage   m_stage { Stage::Idle };
    int     m_pm    { 0 };
    int     m_stageStartPm { 0 };
    int     m_stageTargetPm { 0 };
    qint64  m_stageStartMs { 0 };
    qint64  m_stageEstMs   { 1000 };
};

} // namespace trtforge
