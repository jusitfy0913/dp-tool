#pragma once

#include <QString>
#include <QStringList>
#include <QMetaType>

namespace trtforge {

enum class Precision { FP32 = 0, FP16, INT8, BEST, TF32 };
enum class RuntimeMode { Full = 0, Lean, Dispatch };

struct BuildOptions
{
    QString onnxPath;
    QString enginePath;

    Precision precision    = Precision::FP16;
    RuntimeMode runtime    = RuntimeMode::Full;

    int  workspaceMiB      = 4096;
    int  maxBatch          = 1;     // 仅在隐式 batch 网络下生效，显式 batch 通过 shapes 控制
    int  dlaCore           = -1;    // -1 表示关闭
    int  device            = 0;

    bool sparsity          = false;
    bool refit             = false;
    bool versionCompatible = false;
    bool gpuFallback       = true;
    bool tf32              = false;

    // INT8 标定（可选）
    QString int8CalibCache;
    QString int8CalibDir;

    // 动态形状（输入名 -> 维度字符串，如 "1x3x640x640"）
    // 简化：单输入名 + 三组维度
    QString shapeInputName = "images";
    QString minShape       = "1x3x640x640";
    QString optShape       = "4x3x640x640";
    QString maxShape       = "8x3x640x640";

    // 额外参数 (trtexec 原生命令行风格 token，如 --useCudaGraph)
    QString extraArgs;

    // 其它（来自 extra args 解析，会被合入上面字段或单独保留）
    QString timingCacheFile;

    // 构建后自检
    bool testEnabled    = true;
    int  testIterations = 100;
    int  testWarmup     = 10;
};

// 自检报告（结构化结果，跨线程传递）
struct TestReport
{
    bool    ok            { false };
    QString enginePath;
    QString message;       // 失败原因或简短摘要

    // 性能
    double  avgMs         { 0.0 };
    double  minMs         { 0.0 };
    double  maxMs         { 0.0 };
    double  qps           { 0.0 };
    int     iterations    { 0 };
    int     warmup        { 0 };

    // 显存（构建后）
    double  vramUsedGB    { 0.0 };

    // 张量信息（每行： "[I] images  FP32  [1x3x640x640]"）
    QStringList tensorLines;

    // 第一个输出张量的前若干个数值（粗略一瞥）
    QString firstOutputName;
    QStringList firstOutputPreview;
};

} // namespace trtforge

Q_DECLARE_METATYPE(trtforge::BuildOptions)
Q_DECLARE_METATYPE(trtforge::TestReport)
