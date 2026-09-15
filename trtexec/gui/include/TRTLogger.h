#pragma once

#include <QObject>
#include <QString>
#include <NvInferRuntimeCommon.h>

namespace trtforge {

// 日志级别（与 UI 颜色映射保持一致）
enum class LogLevel : int {
    Verbose = 0,
    Info    = 1,
    Warning = 2,
    Error   = 3,
    Stage   = 4,   // GUI 自定义：阶段标记
    Success = 5,
};

// 实现 nvinfer1::ILogger，将 TRT 内部日志转发到 Qt 信号
class TRTLogger : public QObject, public nvinfer1::ILogger
{
    Q_OBJECT
public:
    explicit TRTLogger(QObject* parent = nullptr);
    ~TRTLogger() override = default;

    // 主线程允许通过此方法手动注入"阶段"或"信息"日志（用于 BuilderWorker 关键节点提示）
    void emitInfo(const QString& msg);
    void emitStage(const QString& msg);
    void emitWarning(const QString& msg);
    void emitError(const QString& msg);
    void emitSuccess(const QString& msg);

    // ILogger 接口
    void log(Severity severity, const char* msg) noexcept override;

    // 设置最低输出级别（默认 INFO）
    void setMinSeverity(Severity s) { m_min = s; }

signals:
    void logArrived(int level, QString text);

private:
    Severity m_min { Severity::kINFO };
};

} // namespace trtforge
