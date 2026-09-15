#include "BuilderWorker.h"

#include "EngineTester.h"
#include "TRTLogger.h"

#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <cuda_runtime_api.h>

#include <memory>
#include <vector>

using namespace nvinfer1;

namespace trtforge {

namespace {

template <typename T>
struct TrtDeleter { void operator()(T* p) const { if (p) delete p; } };
template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;

// 解析 "1x3x640x640" 为 Dims
Dims parseDims(const QString& s, bool* ok = nullptr)
{
    Dims d{};
    d.nbDims = 0;
    if (ok) *ok = false;
    QStringList parts = s.split(QRegularExpression("[xX,]"), Qt::SkipEmptyParts);
    if (parts.isEmpty() || parts.size() > Dims::MAX_DIMS) return d;
    d.nbDims = parts.size();
    for (int i = 0; i < parts.size(); ++i) {
        bool oki = false;
        d.d[i] = parts[i].trimmed().toInt(&oki);
        if (!oki) { d.nbDims = 0; return d; }
    }
    if (ok) *ok = true;
    return d;
}

bool containsNonAsciiPath(const QString& text)
{
    for (const QChar ch : text) {
        if (ch.unicode() > 127) return true;
    }
    return false;
}

QString makeAsciiOnnxCopy(const QString& source)
{
    const QFileInfo info(source);
    const QString suffix = info.suffix().isEmpty() ? QStringLiteral("onnx") : info.suffix();
    const QString target = QDir::temp().absoluteFilePath(
        QStringLiteral("trtforge_onnx_%1.%2").arg(QDateTime::currentMSecsSinceEpoch()).arg(suffix));
    QFile::remove(target);
    return QFile::copy(source, target) ? target : QString();
}

} // namespace

BuilderWorker::BuilderWorker(QObject* parent) : QObject(parent)
{
    m_logger = new TRTLogger(this);
    connect(m_logger, &TRTLogger::logArrived,
            this,     &BuilderWorker::log,
            Qt::DirectConnection); // 同线程直连，由 MainWindow 用 QueuedConnection 跨线程接收
}

// ---------------------------------------------------------------------------
// 额外参数解析：把 "--key=value" / "--flag" 风格 token 合入 BuildOptions
// 仅识别常用项；不识别的项以 [WARN] 提示用户
// ---------------------------------------------------------------------------
void BuilderWorker::parseExtraArgsInto(BuildOptions& opt) const
{
    if (opt.extraArgs.trimmed().isEmpty()) return;

    QStringList tokens = opt.extraArgs.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
    for (const QString& raw : tokens) {
        QString t = raw;
        if (t.startsWith("--")) t.remove(0, 2);
        else if (t.startsWith("-")) t.remove(0, 1);

        QString key, val;
        int eq = t.indexOf('=');
        if (eq >= 0) { key = t.left(eq).toLower(); val = t.mid(eq + 1); }
        else         { key = t.toLower();          val = QString();    }

        if (key == "fp16")            opt.precision = Precision::FP16;
        else if (key == "int8" || key == "best") {
            m_logger->emitWarning(QString("已忽略不支持的精度参数: --%1（本工具未集成 INT8 标定，请改用 FP16）").arg(key));
        }
        else if (key == "tf32")       opt.tf32 = true;
        else if (key == "sparsity")   opt.sparsity = true;
        else if (key == "refit")      opt.refit = true;
        else if (key == "versioncompatible") opt.versionCompatible = true;
        else if (key == "allowgpufallback")  opt.gpuFallback = true;
        else if (key == "workspace")  { bool ok=false; int v = val.toInt(&ok); if (ok) opt.workspaceMiB = v; }
        else if (key == "device")     { bool ok=false; int v = val.toInt(&ok); if (ok) opt.device = v; }
        else if (key == "usedlacore") { bool ok=false; int v = val.toInt(&ok); if (ok) opt.dlaCore = v; }
        else if (key == "minshapes")  opt.minShape = val.section(':', -1);
        else if (key == "optshapes")  opt.optShape = val.section(':', -1);
        else if (key == "maxshapes")  opt.maxShape = val.section(':', -1);
        else if (key == "shapes")     { opt.minShape = opt.optShape = opt.maxShape = val.section(':', -1); }
        else if (key == "calib")      opt.int8CalibCache = val;
        else if (key == "timingcachefile") opt.timingCacheFile = val;
        else if (key == "savetimingcache") opt.timingCacheFile = val;
        else if (key == "loadtimingcache") opt.timingCacheFile = val;
        else if (key == "onnx")       opt.onnxPath   = val;
        else if (key == "saveengine") opt.enginePath = val;
        else {
            m_logger->emitWarning(QString("额外参数未识别（已忽略）: %1").arg(raw));
        }
    }
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------
void BuilderWorker::start(BuildOptions opt)
{
    m_cancel.store(false);
    QElapsedTimer timer;
    timer.start();

    parseExtraArgsInto(opt);

    QString err;
    bool ok = runImpl(opt, err);
    qint64 elapsed = timer.elapsed();

    if (!ok) {
        m_logger->emitError(QString("构建失败：%1").arg(err.isEmpty() ? "未知错误" : err));
    } else {
        m_logger->emitSuccess(QString("引擎已生成：%1（耗时 %2 ms）").arg(opt.enginePath).arg(elapsed));
    }
    emit finished(ok, opt.enginePath, elapsed);

    // 构建成功且勾选自检 -> 顺手跑一次
    if (ok && opt.testEnabled) {
        EngineTester tester(m_logger);
        TestReport report = tester.run(opt.enginePath, opt.testWarmup, opt.testIterations,
                                       opt.device, opt.shapeInputName, opt.optShape);
        emit testFinished(report);
    }
}

void BuilderWorker::testEngineOnly(QString enginePath, int warmup, int iterations,
                                   int device, QString shapeInputName, QString optShape)
{
    EngineTester tester(m_logger);
    TestReport report = tester.run(enginePath, warmup, iterations, device, shapeInputName, optShape);
    emit testFinished(report);
}

bool BuilderWorker::runImpl(const BuildOptions& opt, QString& outErr)
{
    if (opt.onnxPath.isEmpty() || !QFileInfo::exists(opt.onnxPath)) {
        outErr = QString("ONNX 文件不存在：%1").arg(opt.onnxPath);
        return false;
    }
    if (opt.enginePath.isEmpty()) {
        outErr = "未指定输出引擎路径";
        return false;
    }

    // 1) 设备
    cudaError_t cerr = cudaSetDevice(opt.device);
    if (cerr != cudaSuccess) {
        outErr = QString("cudaSetDevice(%1) 失败：%2").arg(opt.device).arg(cudaGetErrorString(cerr));
        return false;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, opt.device) == cudaSuccess) {
        m_logger->emitInfo(QString("使用设备：%1（计算能力 %2.%3）")
                               .arg(prop.name).arg(prop.major).arg(prop.minor));
    }

    // 2) 插件 + Builder
    initLibNvInferPlugins(m_logger, "");
    m_logger->emitStage("[阶段] 初始化 Builder");

    TrtPtr<IBuilder> builder(createInferBuilder(*m_logger));
    if (!builder) { outErr = "createInferBuilder 失败"; return false; }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtPtr<INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
    if (!network) { outErr = "createNetworkV2 失败"; return false; }

    // 3) 解析 ONNX
    m_logger->emitStage("[阶段] Parsing ONNX");
    TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, *m_logger));
    QString parserPath = opt.onnxPath;
    QString tempOnnxPath;
    if (containsNonAsciiPath(opt.onnxPath)) {
        tempOnnxPath = makeAsciiOnnxCopy(opt.onnxPath);
        if (tempOnnxPath.isEmpty()) {
            outErr = QString("ONNX 中文路径临时转存失败：%1").arg(opt.onnxPath);
            return false;
        }
        parserPath = tempOnnxPath;
        m_logger->emitInfo(QString("检测到 ONNX 中文路径，已使用临时 ASCII 路径解析：%1").arg(tempOnnxPath));
    }
    const QByteArray onnxBytes = parserPath.toLocal8Bit();
    if (!parser->parseFromFile(onnxBytes.constData(),
                               static_cast<int>(ILogger::Severity::kWARNING))) {
        if (!tempOnnxPath.isEmpty()) QFile::remove(tempOnnxPath);
        QStringList errs;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            errs << QString::fromUtf8(parser->getError(i)->desc());
        }
        outErr = "ONNX 解析失败：" + (errs.isEmpty() ? "未知" : errs.join("; "));
        return false;
    }
    if (!tempOnnxPath.isEmpty()) QFile::remove(tempOnnxPath);
    m_logger->emitInfo(QString("Successfully parsed ONNX，输入数=%1，输出数=%2")
                           .arg(network->getNbInputs()).arg(network->getNbOutputs()));

    if (m_cancel.load()) { outErr = "已取消"; return false; }

    // 4) 构建配置
    m_logger->emitStage("[阶段] Building config");
    TrtPtr<IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) { outErr = "createBuilderConfig 失败"; return false; }

    const size_t wsBytes = static_cast<size_t>(qMax(64, opt.workspaceMiB)) * 1024ULL * 1024ULL;
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, wsBytes);
    m_logger->emitInfo(QString("工作空间: %1 MiB").arg(opt.workspaceMiB));

    auto setFlag = [&](BuilderFlag f, bool on, const QString& name) {
        if (on) {
            config->setFlag(f);
            m_logger->emitInfo(QString("启用 %1").arg(name));
        }
    };
    setFlag(BuilderFlag::kFP16,               opt.precision == Precision::FP16, "FP16");
    setFlag(BuilderFlag::kTF32,               opt.tf32 || opt.precision == Precision::TF32, "TF32");
    setFlag(BuilderFlag::kSPARSE_WEIGHTS,     opt.sparsity, "稀疏化");
    setFlag(BuilderFlag::kREFIT,              opt.refit, "可重拟合");
#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 6))
    setFlag(BuilderFlag::kVERSION_COMPATIBLE, opt.versionCompatible, "版本兼容");
#else
    if (opt.versionCompatible) {
        m_logger->emitWarning(
            QStringLiteral("[TRT %1.%2] BuilderFlag::kVERSION_COMPATIBLE 需要 TensorRT 8.6+，当前版本已忽略该选项")
                .arg(NV_TENSORRT_MAJOR).arg(NV_TENSORRT_MINOR));
    }
#endif
    setFlag(BuilderFlag::kGPU_FALLBACK,       opt.gpuFallback, "允许GPU回退");

    if (opt.dlaCore >= 0) {
        config->setDefaultDeviceType(DeviceType::kDLA);
        config->setDLACore(opt.dlaCore);
        m_logger->emitInfo(QString("DLA 核心: %1").arg(opt.dlaCore));
    }

    if (opt.precision == Precision::INT8 && opt.int8CalibCache.isEmpty()) {
        m_logger->emitWarning("已选择 INT8 但未提供标定缓存（--calib=path），TRT 将使用默认/随机标定，精度不可保证。");
    }

    // 5) 动态形状 profile —— 仅当存在动态维(-1)时才添加
    if (network->getNbInputs() > 0) {
        // 判断网络是否有任何动态输入维
        bool hasDynamic = false;
        for (int i = 0; i < network->getNbInputs(); ++i) {
            const Dims d = network->getInput(i)->getDimensions();
            for (int k = 0; k < d.nbDims; ++k) {
                if (d.d[k] < 0) { hasDynamic = true; break; }
            }
            if (hasDynamic) break;
        }

        if (!hasDynamic) {
            // 静态 shape ONNX：profile 反而会让 TRT 报错，输出实际 shape 提示用户
            const Dims d = network->getInput(0)->getDimensions();
            QStringList parts;
            for (int k = 0; k < d.nbDims; ++k) parts << QString::number(d.d[k]);
            m_logger->emitInfo(QString("检测到静态输入 [%1]：%2，已跳过 optimization profile（无需动态 Shape）")
                                   .arg(QString::fromUtf8(network->getInput(0)->getName()),
                                        parts.join('x')));
        } else {
            IOptimizationProfile* profile = builder->createOptimizationProfile();
            const QString inputName = !opt.shapeInputName.isEmpty()
                                          ? opt.shapeInputName
                                          : QString::fromUtf8(network->getInput(0)->getName());
            bool okMin=false, okOpt=false, okMax=false;
            const Dims dMin = parseDims(opt.minShape, &okMin);
            const Dims dOpt = parseDims(opt.optShape, &okOpt);
            const Dims dMax = parseDims(opt.maxShape, &okMax);
            if (okMin && okOpt && okMax) {
                const QByteArray inp = inputName.toLocal8Bit();
                profile->setDimensions(inp.constData(), OptProfileSelector::kMIN, dMin);
                profile->setDimensions(inp.constData(), OptProfileSelector::kOPT, dOpt);
                profile->setDimensions(inp.constData(), OptProfileSelector::kMAX, dMax);
                config->addOptimizationProfile(profile);
                m_logger->emitInfo(QString("动态 Shape 配置 [%1]：%2 / %3 / %4")
                                       .arg(inputName, opt.minShape, opt.optShape, opt.maxShape));
            } else {
                m_logger->emitWarning("动态 Shape 字符串无法解析，已跳过 optimization profile");
            }
        }
    }

    if (m_cancel.load()) { outErr = "已取消"; return false; }

    // 6) 构建并序列化
    m_logger->emitStage("[阶段] Tactic 选择 / 构建中（耗时较长，请耐心等待）");
    TrtPtr<IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) { outErr = "buildSerializedNetwork 返回空"; return false; }
    m_logger->emitInfo(QString("Engine generation completed，序列化大小: %1 KB")
                           .arg(serialized->size() / 1024));

    // 7) 写盘
    m_logger->emitStage("[阶段] Writing engine 到磁盘");
    QFile f(opt.enginePath);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        outErr = QString("无法写入 %1：%2").arg(opt.enginePath, f.errorString());
        return false;
    }
    qint64 wrote = f.write(reinterpret_cast<const char*>(serialized->data()),
                           static_cast<qint64>(serialized->size()));
    f.close();
    if (wrote != static_cast<qint64>(serialized->size())) {
        outErr = "写入字节数不匹配";
        return false;
    }
    m_logger->emitInfo(QString("Saved engine to %1").arg(opt.enginePath));
    return true;
}

} // namespace trtforge
