#include "engine/OnnxInputInspector.h"

#include <QFile>
#include <QFileInfo>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <memory>

namespace trtforge {

namespace {

template <typename T>
struct TrtDeleter
{
    void operator()(T* p) const
    {
        if (p) {
            delete p;
        }
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;

class InspectLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING && msg) {
            m_lastWarning = QString::fromUtf8(msg);
        }
    }

    QString lastWarning() const { return m_lastWarning; }

private:
    QString m_lastWarning;
};

int positiveOrFallback(const QVector<int>& dims, int index, int fallback)
{
    if (index >= 0 && index < dims.size() && dims[index] > 0) {
        return dims[index];
    }
    return fallback;
}

} // namespace

int OnnxInputInfo::channel(int fallback) const
{
    return isFourDim() ? positiveOrFallback(dims, 1, fallback) : fallback;
}

int OnnxInputInfo::height(int fallback) const
{
    return isFourDim() ? positiveOrFallback(dims, 2, fallback) : fallback;
}

int OnnxInputInfo::width(int fallback) const
{
    return isFourDim() ? positiveOrFallback(dims, 3, fallback) : fallback;
}

OnnxInputInfo OnnxInputInspector::inspect(const QString& onnxPath)
{
    OnnxInputInfo info;
    if (onnxPath.trimmed().isEmpty() || !QFileInfo::exists(onnxPath)) {
        info.error = QStringLiteral("ONNX 文件不存在：%1").arg(onnxPath);
        return info;
    }

    QFile file(onnxPath);
    if (!file.open(QIODevice::ReadOnly)) {
        info.error = QStringLiteral("无法读取 ONNX：%1").arg(file.errorString());
        return info;
    }
    const QByteArray model = file.readAll();
    if (model.isEmpty()) {
        info.error = QStringLiteral("ONNX 文件为空：%1").arg(onnxPath);
        return info;
    }

    InspectLogger logger;
    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        info.error = QStringLiteral("createInferBuilder 失败");
        return info;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
    if (!network) {
        info.error = QStringLiteral("createNetworkV2 失败");
        return info;
    }

    TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        info.error = QStringLiteral("createParser 失败");
        return info;
    }

    if (!parser->parse(model.constData(), static_cast<size_t>(model.size()))) {
        QStringList errors;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            errors << QString::fromUtf8(parser->getError(i)->desc());
        }
        info.error = QStringLiteral("ONNX 输入信息解析失败：%1")
                         .arg(errors.isEmpty() ? logger.lastWarning() : errors.join(QStringLiteral("; ")));
        return info;
    }

    if (network->getNbInputs() <= 0 || !network->getInput(0)) {
        info.error = QStringLiteral("ONNX 未发现输入 tensor");
        return info;
    }

    const nvinfer1::ITensor* input = network->getInput(0);
    info.name = QString::fromUtf8(input->getName());
    const nvinfer1::Dims dims = input->getDimensions();
    info.dims.reserve(dims.nbDims);
    for (int i = 0; i < dims.nbDims; ++i) {
        info.dims.append(static_cast<int>(dims.d[i]));
    }
    info.ok = true;
    return info;
}

} // namespace trtforge
