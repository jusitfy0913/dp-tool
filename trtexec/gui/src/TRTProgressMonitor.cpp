#include "TRTProgressMonitor.h"

namespace trtforge {

TRTProgressMonitor::TRTProgressMonitor(QObject* parent) : QObject(parent)
{
    m_tick.setInterval(100);
    connect(&m_tick, &QTimer::timeout, this, &TRTProgressMonitor::onTick);
}

void TRTProgressMonitor::start()
{
    reset();
    m_timer.start();
    m_tick.start();
    setStage(Stage::Parse);
}

void TRTProgressMonitor::stop()
{
    m_tick.stop();
}

void TRTProgressMonitor::reset()
{
    m_tick.stop();
    m_pm = 0;
    m_stage = Stage::Idle;
    m_stageStartPm = 0;
    m_stageTargetPm = 0;
    m_stageStartMs = 0;
    m_stageEstMs = 1000;
    emit progressChanged(0);
    emit stageChanged(static_cast<int>(Stage::Idle));
    emit etaChanged(-1);
}

namespace {
struct StageBudget { int targetPm; qint64 estMs; };
StageBudget budgetOf(TRTProgressMonitor::Stage s)
{
    using S = TRTProgressMonitor::Stage;
    switch (s) {
        case S::Parse:     return { 100,  3000 };
        case S::Optimize:  return { 250,  8000 };
        case S::Tune:      return { 850, 60000 };  // 主要耗时阶段
        case S::Serialize: return { 970,  5000 };
        case S::Save:      return {1000,  1000 };
        default:           return {   0,     1 };
    }
}
} // namespace

void TRTProgressMonitor::setStage(Stage s)
{
    if (m_stage == s) return;
    m_stage = s;
    auto b = budgetOf(s);
    m_stageStartPm  = m_pm;
    m_stageTargetPm = b.targetPm;
    m_stageStartMs  = m_timer.isValid() ? m_timer.elapsed() : 0;
    m_stageEstMs    = b.estMs;
    emit stageChanged(static_cast<int>(s));
}

void TRTProgressMonitor::setPermille(int pm)
{
    pm = qBound(0, pm, 1000);
    if (pm != m_pm) {
        m_pm = pm;
        emit progressChanged(m_pm);
    }
}

void TRTProgressMonitor::onTick()
{
    if (m_stage == Stage::Idle || m_stage == Stage::Done || m_stage == Stage::Failed) {
        return;
    }
    qint64 now = m_timer.elapsed();
    qint64 dt  = now - m_stageStartMs;
    double r   = m_stageEstMs > 0 ? double(dt) / double(m_stageEstMs) : 1.0;
    if (r < 0) r = 0;
    if (r > 0.97) r = 0.97; // 阶段内不要冲到目标值，等关键日志触发跳转
    int target = m_stageStartPm + int((m_stageTargetPm - m_stageStartPm) * r);
    if (target > m_pm) setPermille(target);

    if (m_pm > 5 && m_pm < 1000) {
        qint64 totalEst = qint64(double(now) * 1000.0 / double(m_pm));
        emit etaChanged(qMax<qint64>(0, totalEst - now));
    }
}

void TRTProgressMonitor::onLog(int /*level*/, const QString& text)
{
    // 关键字识别 -> 阶段切换
    const QString s = text;
    if (s.contains("Parsing", Qt::CaseInsensitive) || s.contains("ONNX IR", Qt::CaseInsensitive)) {
        setStage(Stage::Parse);
    } else if (s.contains("Successfully parsed", Qt::CaseInsensitive) ||
               s.contains("parsing complete", Qt::CaseInsensitive) ||
               s.contains("Building config", Qt::CaseInsensitive)) {
        setPermille(100);
        setStage(Stage::Optimize);
    } else if (s.contains("Original: ", Qt::CaseInsensitive) ||
               s.contains("After Myelin", Qt::CaseInsensitive) ||
               s.contains("Graph optimization", Qt::CaseInsensitive)) {
        setStage(Stage::Optimize);
    } else if (s.contains("Tactic", Qt::CaseInsensitive) ||
               s.contains("Timing Runner", Qt::CaseInsensitive) ||
               s.contains("Autotuning", Qt::CaseInsensitive)) {
        if (m_stage == Stage::Parse || m_stage == Stage::Optimize) {
            setPermille(250);
        }
        setStage(Stage::Tune);
    } else if (s.contains("Total Activation Memory", Qt::CaseInsensitive) ||
               s.contains("Engine generation completed", Qt::CaseInsensitive) ||
               s.contains("Total time of building", Qt::CaseInsensitive) ||
               s.contains("Serializing", Qt::CaseInsensitive)) {
        setPermille(870);
        setStage(Stage::Serialize);
    } else if (s.contains("Saved engine", Qt::CaseInsensitive) ||
               s.contains("Writing engine", Qt::CaseInsensitive)) {
        setPermille(970);
        setStage(Stage::Save);
    }
}

void TRTProgressMonitor::finish(bool ok)
{
    if (ok) {
        setPermille(1000);
        setStage(Stage::Done);
        emit etaChanged(0);
    } else {
        setStage(Stage::Failed);
    }
    m_tick.stop();
}

} // namespace trtforge
