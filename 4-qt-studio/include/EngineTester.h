#pragma once

#include <QString>

#include "BuildOptions.h"

namespace trtforge {

class TRTLogger;

// 引擎自检：反序列化 .trt -> 随机数据填输入 -> 跑 N 次 -> 报告维度/吞吐
class EngineTester
{
public:
    // logger 仅用于发日志；调用线程必须能访问 CUDA Driver（建议在 BuilderWorker 同一线程使用）
    explicit EngineTester(TRTLogger* logger);

    // 输入：
    //   enginePath  - .trt 文件路径
    //   warmup      - 预热次数（不计时）
    //   iterations  - 计时次数
    //   device      - GPU index（-1 表示沿用当前线程的 cudaSetDevice）
    //   shapeInputName / optShape - 当遇到动态维度时使用，从 GUI BuildOptions 透传
    //
    // 输出：完整 TestReport（无论成功失败均填 ok / message）
    TestReport run(const QString& enginePath,
                   int            warmup,
                   int            iterations,
                   int            device,
                   const QString& shapeInputName,
                   const QString& optShape);

private:
    TRTLogger* m_logger { nullptr };
};

} // namespace trtforge
