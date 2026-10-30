#pragma once

#include <QObject>
#include <QString>
#include <atomic>

#include "BuildOptions.h"

namespace trtforge {

class TRTLogger;

// 在子线程中执行 ONNX -> TensorRT 引擎的全流程
class BuilderWorker : public QObject
{
    Q_OBJECT
public:
    explicit BuilderWorker(QObject* parent = nullptr);
    ~BuilderWorker() override = default;

    void requestCancel() { m_cancel.store(true); }
    bool isCancelled() const { return m_cancel.load(); }

    TRTLogger* logger() const { return m_logger; }

public slots:
    void start(trtforge::BuildOptions opt);
    // 对已存在的 .trt 引擎做自检（不构建）
    void testEngineOnly(QString enginePath, int warmup, int iterations,
                        int device, QString shapeInputName, QString optShape);

signals:
    // 由 TRTLogger 转发的统一日志信号（直接连）
    void log(int level, QString text);
    // 引擎生成完毕
    void finished(bool ok, QString enginePath, qint64 elapsedMs);
    // 自检完成
    void testFinished(trtforge::TestReport report);

private:
    bool runImpl(const BuildOptions& opt, QString& outErr);
    void parseExtraArgsInto(BuildOptions& opt) const;

    TRTLogger* m_logger { nullptr };
    std::atomic<bool> m_cancel { false };
};

} // namespace trtforge
