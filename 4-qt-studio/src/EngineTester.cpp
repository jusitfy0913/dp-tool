#include "EngineTester.h"

#include "TRTLogger.h"

#include <QByteArray>
#include <QElapsedTimer>
#include <QFile>
#include <QRegularExpression>
#include <QStringList>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <vector>

using namespace nvinfer1;

namespace trtforge {

namespace {

template <typename T>
struct TrtDeleter { void operator()(T* p) const { if (p) delete p; } };
template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;

const char* dataTypeName(DataType t)
{
    switch (t) {
        case DataType::kFLOAT: return "FP32";
        case DataType::kHALF:  return "FP16";
        case DataType::kINT8:  return "INT8";
        case DataType::kINT32: return "INT32";
        case DataType::kBOOL:  return "BOOL";
        case DataType::kUINT8: return "UINT8";
#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 6)
        case DataType::kFP8:   return "FP8";
#endif
        default: return "UNKNOWN";
    }
}

int dataTypeSize(DataType t)
{
    switch (t) {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF:  return 2;
        case DataType::kINT8:  return 1;
        case DataType::kINT32: return 4;
        case DataType::kBOOL:  return 1;
        case DataType::kUINT8: return 1;
#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 6)
        case DataType::kFP8:   return 1;
#endif
        default: return 0;
    }
}

QString dimsToString(const Dims& d)
{
    QStringList p;
    for (int i = 0; i < d.nbDims; ++i) p << QString::number(d.d[i]);
    return p.join('x');
}

Dims parseDims(const QString& s)
{
    Dims d{};
    d.nbDims = 0;
    QStringList parts = s.split(QRegularExpression("[xX,]"), Qt::SkipEmptyParts);
    if (parts.isEmpty() || parts.size() > Dims::MAX_DIMS) return d;
    d.nbDims = parts.size();
    for (int i = 0; i < parts.size(); ++i) {
        bool ok = false;
        d.d[i] = parts[i].trimmed().toInt(&ok);
        if (!ok) { d.nbDims = 0; return d; }
    }
    return d;
}

int64_t volume(const Dims& d)
{
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= std::max<int64_t>(1, d.d[i]);
    return v;
}

bool hasDynamic(const Dims& d)
{
    for (int i = 0; i < d.nbDims; ++i) if (d.d[i] < 0) return true;
    return false;
}

// fp32 -> fp16 的简易转换（IEEE 754 round-to-nearest-even 简化）
uint16_t f32_to_f16(float x)
{
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t  exp  = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);            // 下溢/0
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);  // 上溢/Inf
    return static_cast<uint16_t>(sign | (uint32_t(exp) << 10) | (mant >> 13));
}

// 半精度 -> float（仅用于打印 preview）
float f16_to_f32(uint16_t h)
{
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t mant = (h & 0x03FFu);
    uint32_t f;
    if (exp == 0) {
        f = sign;
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
}

} // namespace

EngineTester::EngineTester(TRTLogger* logger) : m_logger(logger) {}

TestReport EngineTester::run(const QString& enginePath,
                             int warmup,
                             int iterations,
                             int device,
                             const QString& shapeInputName,
                             const QString& optShape)
{
    TestReport r;
    r.enginePath = enginePath;
    r.warmup     = std::max(0, warmup);
    r.iterations = std::max(1, iterations);

    auto failWith = [&](const QString& m) -> TestReport {
        r.ok = false; r.message = m;
        if (m_logger) m_logger->emitError(QString("[自检] %1").arg(m));
        return r;
    };

    if (m_logger) m_logger->emitStage(QString("[自检] 反序列化引擎: %1").arg(enginePath));

    if (device >= 0) cudaSetDevice(device);

    QFile f(enginePath);
    if (!f.open(QIODevice::ReadOnly)) return failWith(QString("无法打开引擎文件：%1").arg(f.errorString()));
    QByteArray bytes = f.readAll();
    f.close();
    if (bytes.isEmpty()) return failWith("引擎文件为空");

    TrtPtr<IRuntime> runtime(createInferRuntime(*m_logger));
    if (!runtime) return failWith("createInferRuntime 失败");

    TrtPtr<ICudaEngine> engine(runtime->deserializeCudaEngine(bytes.constData(), bytes.size()));
    if (!engine) return failWith("deserializeCudaEngine 失败（引擎可能与当前 TRT 版本不兼容）");

    TrtPtr<IExecutionContext> ctx(engine->createExecutionContext());
    if (!ctx) return failWith("createExecutionContext 失败");

    // ---- 收集所有 IO tensor ----
    struct Binding {
        std::string name;
        TensorIOMode mode;
        DataType dtype;
        Dims dims;
        size_t bytes = 0;
        void* dPtr = nullptr;
    };
    std::vector<Binding> bindings;
    int nbIO = engine->getNbIOTensors();
    bindings.reserve(nbIO);

    Dims dynOpt = parseDims(optShape);

    for (int i = 0; i < nbIO; ++i) {
        Binding b;
        b.name  = engine->getIOTensorName(i);
        b.mode  = engine->getTensorIOMode(b.name.c_str());
        b.dtype = engine->getTensorDataType(b.name.c_str());
        b.dims  = engine->getTensorShape(b.name.c_str());

        if (b.mode == TensorIOMode::kINPUT && hasDynamic(b.dims)) {
            // 优先使用与 GUI 一致的 optShape；否则用 profile 的 OPT
            Dims toSet = b.dims;
            bool used = false;
            if (dynOpt.nbDims == b.dims.nbDims) {
                toSet = dynOpt; used = true;
            } else {
                Dims optD = engine->getProfileShape(b.name.c_str(), 0, OptProfileSelector::kOPT);
                if (optD.nbDims == b.dims.nbDims) { toSet = optD; used = true; }
            }
            if (!used) return failWith(QString("输入 %1 形状动态但无法确定测试 shape").arg(QString::fromStdString(b.name)));
            if (!ctx->setInputShape(b.name.c_str(), toSet))
                return failWith(QString("setInputShape(%1) 失败").arg(QString::fromStdString(b.name)));
            b.dims = toSet;
        }

        // 输出形状在所有输入设定后再取
        bindings.push_back(b);
    }
    // 输出维度依赖输入 shape，重新查询
    for (auto& b : bindings) {
        if (b.mode == TensorIOMode::kOUTPUT) {
            b.dims = ctx->getTensorShape(b.name.c_str());
        }
        b.bytes = static_cast<size_t>(volume(b.dims)) * static_cast<size_t>(dataTypeSize(b.dtype));
        if (b.bytes == 0) return failWith(QString("张量 %1 计算大小为 0（dims=%2 dtype=%3）")
            .arg(QString::fromStdString(b.name), dimsToString(b.dims), dataTypeName(b.dtype)));
        cudaError_t cerr = cudaMalloc(&b.dPtr, b.bytes);
        if (cerr != cudaSuccess || !b.dPtr) {
            return failWith(QString("cudaMalloc(%1 bytes) 失败：%2")
                .arg(b.bytes).arg(cudaGetErrorString(cerr)));
        }
        ctx->setTensorAddress(b.name.c_str(), b.dPtr);

        QString line = QString("[%1] %2  %3  [%4]  %5KB")
                          .arg(b.mode == TensorIOMode::kINPUT ? "I" : "O",
                               QString::fromStdString(b.name),
                               dataTypeName(b.dtype),
                               dimsToString(b.dims))
                          .arg(b.bytes / 1024);
        r.tensorLines << line;
        if (m_logger) m_logger->emitInfo(QString("[自检] %1").arg(line));
    }

    // ---- 生成随机输入到 host，再拷到 device ----
    std::mt19937 rng(0xC0FFEEu);
    for (auto& b : bindings) {
        if (b.mode != TensorIOMode::kINPUT) continue;
        size_t n = static_cast<size_t>(volume(b.dims));
        std::vector<uint8_t> host(b.bytes);
        switch (b.dtype) {
            case DataType::kFLOAT: {
                std::uniform_real_distribution<float> u(0.0f, 1.0f);
                auto* p = reinterpret_cast<float*>(host.data());
                for (size_t i = 0; i < n; ++i) p[i] = u(rng);
                break;
            }
            case DataType::kHALF: {
                std::uniform_real_distribution<float> u(0.0f, 1.0f);
                auto* p = reinterpret_cast<uint16_t*>(host.data());
                for (size_t i = 0; i < n; ++i) p[i] = f32_to_f16(u(rng));
                break;
            }
            case DataType::kINT8:
            case DataType::kUINT8: {
                std::uniform_int_distribution<int> u(0, 255);
                for (size_t i = 0; i < n; ++i) host[i] = static_cast<uint8_t>(u(rng));
                break;
            }
            case DataType::kINT32: {
                std::uniform_int_distribution<int> u(0, 100);
                auto* p = reinterpret_cast<int32_t*>(host.data());
                for (size_t i = 0; i < n; ++i) p[i] = u(rng);
                break;
            }
            case DataType::kBOOL: {
                for (size_t i = 0; i < n; ++i) host[i] = static_cast<uint8_t>(rng() & 1);
                break;
            }
            default:
                std::fill(host.begin(), host.end(), 0);
                break;
        }
        cudaMemcpy(b.dPtr, host.data(), b.bytes, cudaMemcpyHostToDevice);
    }

    // ---- 计时 ----
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    if (m_logger) m_logger->emitStage(QString("[自检] Warmup x%1, Iterations x%2").arg(r.warmup).arg(r.iterations));
    for (int i = 0; i < r.warmup; ++i) {
        if (!ctx->enqueueV3(stream)) return failWith(QString("enqueueV3 失败（warmup #%1）").arg(i));
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t evStart{}, evStop{};
    cudaEventCreate(&evStart); cudaEventCreate(&evStop);
    std::vector<float> latMs;
    latMs.reserve(r.iterations);

    for (int i = 0; i < r.iterations; ++i) {
        cudaEventRecord(evStart, stream);
        if (!ctx->enqueueV3(stream)) {
            cudaEventDestroy(evStart); cudaEventDestroy(evStop); cudaStreamDestroy(stream);
            for (auto& b : bindings) if (b.dPtr) cudaFree(b.dPtr);
            return failWith(QString("enqueueV3 失败（iter #%1）").arg(i));
        }
        cudaEventRecord(evStop, stream);
        cudaEventSynchronize(evStop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, evStart, evStop);
        latMs.push_back(ms);
    }
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);

    // ---- 第一个输出 preview ----
    for (auto& b : bindings) {
        if (b.mode != TensorIOMode::kOUTPUT) continue;
        size_t n = static_cast<size_t>(volume(b.dims));
        size_t take = std::min<size_t>(n, 8);
        std::vector<uint8_t> host(b.bytes);
        cudaMemcpy(host.data(), b.dPtr, b.bytes, cudaMemcpyDeviceToHost);
        r.firstOutputName = QString::fromStdString(b.name);
        for (size_t i = 0; i < take; ++i) {
            float v = 0.0f;
            switch (b.dtype) {
                case DataType::kFLOAT: v = reinterpret_cast<float*>(host.data())[i]; break;
                case DataType::kHALF:  v = f16_to_f32(reinterpret_cast<uint16_t*>(host.data())[i]); break;
                case DataType::kINT32: v = static_cast<float>(reinterpret_cast<int32_t*>(host.data())[i]); break;
                case DataType::kINT8:  v = static_cast<float>(reinterpret_cast<int8_t*>(host.data())[i]); break;
                case DataType::kUINT8: v = static_cast<float>(host[i]); break;
                default: break;
            }
            r.firstOutputPreview << QString::asprintf("%.4f", v);
        }
        break;
    }

    cudaStreamDestroy(stream);
    for (auto& b : bindings) if (b.dPtr) cudaFree(b.dPtr);

    // ---- 统计 ----
    double sum = 0.0;
    r.minMs = latMs.front(); r.maxMs = latMs.front();
    for (float v : latMs) {
        sum += v;
        r.minMs = std::min<double>(r.minMs, v);
        r.maxMs = std::max<double>(r.maxMs, v);
    }
    r.avgMs = sum / latMs.size();
    r.qps   = r.avgMs > 0.0 ? 1000.0 / r.avgMs : 0.0;

    size_t freeB = 0, totalB = 0;
    if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess && totalB > 0) {
        r.vramUsedGB = double(totalB - freeB) / 1024.0 / 1024.0 / 1024.0;
    }

    r.ok      = true;
    r.message = QString("自检通过：avg %1 ms, %2 QPS").arg(r.avgMs, 0, 'f', 3).arg(r.qps, 0, 'f', 1);
    if (m_logger) {
        m_logger->emitSuccess(QString("[自检] %1").arg(r.message));
        m_logger->emitInfo(QString("[自检] 时延 min/avg/max = %1 / %2 / %3 ms")
                              .arg(r.minMs, 0, 'f', 3).arg(r.avgMs, 0, 'f', 3).arg(r.maxMs, 0, 'f', 3));
        if (!r.firstOutputPreview.isEmpty()) {
            m_logger->emitInfo(QString("[自检] 输出 %1 前 %2 个值: %3")
                                  .arg(r.firstOutputName).arg(r.firstOutputPreview.size())
                                  .arg(r.firstOutputPreview.join(", ")));
        }
    }
    return r;
}

} // namespace trtforge
