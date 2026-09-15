//
// Created by ubuntu on 4/7/23.
//
/**
 * @file    common.hpp
 * @brief   YOLOv8 OBB（旋转框检测）工程的通用工具集合。
 *
 * 提供：
 *  - CHECK(call)            : CUDA Runtime 错误检查宏
 *  - class Logger           : TensorRT 日志器（实现 nvinfer1::ILogger）
 *  - get_size_by_dims/type_to_size/clamp : 维度、数据类型、数值裁剪辅助
 *  - obb::Binding           : TensorRT 输入/输出绑定的元信息
 *  - obb::Object            : 单个旋转框检测结果（cv::RotatedRect + 类别 + 置信度）
 *  - obb::PreParam          : letterbox 预处理参数（缩放比/边距/原图尺寸），
 *                             供后处理把检测框还原回原图坐标
 */

#ifndef POSE_NORMAL_COMMON_HPP
#define POSE_NORMAL_COMMON_HPP
#include "NvInfer.h"
#include "filesystem.hpp"
#include "opencv2/opencv.hpp"

/**
 * @brief CUDA Runtime 调用错误检查宏；非 cudaSuccess 时打印位置并 exit(1)。
 */
#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        const cudaError_t error_code = call;                                                                           \
        if (error_code != cudaSuccess) {                                                                               \
            printf("CUDA Error:\n");                                                                                   \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error code: %d\n", error_code);                                                                \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));                                            \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

/**
 * @class Logger
 * @brief TensorRT 推理日志器；按 reportableSeverity 过滤后写到 std::cerr。
 */
class Logger: public nvinfer1::ILogger {
public:
    /// 最低输出级别（数值越大越啰嗦）
    nvinfer1::ILogger::Severity reportableSeverity;

    /**
     * @brief 构造函数。
     * @param severity 默认 kINFO；OBB 工程一般传入 kERROR 减少日志干扰。
     */
    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO):
        reportableSeverity(severity)
    {
    }

    /**
     * @brief TensorRT 内部回调：根据级别写带前缀的日志到 std::cerr。
     */
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        // 高于阈值的日志（数值更大表示更不重要）直接忽略
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "VERBOSE: ";
                break;
        }
        std::cerr << msg << std::endl;
    }
};

/**
 * @brief 计算 nvinfer1::Dims 描述张量的总元素个数（各维度连乘）。
 */
inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

/**
 * @brief 把 TensorRT DataType 映射成单元素字节数。
 *        FLOAT=4, HALF=2, INT32=4, INT8=1, BOOL=1, 其他默认 4。
 */
inline int type_to_size(const nvinfer1::DataType& dataType)
{
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 4;
    }
}

/**
 * @brief 数值范围裁剪：把 val 限制到 [min, max] 区间内。
 */
inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

namespace obb {
/**
 * @struct Binding
 * @brief 描述一个 TensorRT 输入/输出绑定（实际占用字节 = size * dsize）。
 */
struct Binding {
    size_t         size  = 1;  ///< 元素个数（dims 各维度乘积）
    size_t         dsize = 1;  ///< 单元素字节数
    nvinfer1::Dims dims;        ///< 张量维度
    std::string    name;        ///< 绑定名（与 engine 中导出的名一致）
};

/**
 * @struct Object
 * @brief 单个旋转框检测结果。
 */
struct Object {
    cv::RotatedRect rect;       ///< 旋转框（中心点、宽高、角度）
    int             label = 0;  ///< 类别 id
    float           prob  = 0.0;///< 置信度
};

/**
 * @struct PreParam
 * @brief letterbox 预处理参数，供后处理把模型坐标还原为原图坐标。
 *
 * 还原公式：原图坐标 = (模型输出坐标 - dw/dh) * ratio
 */
struct PreParam {
    float ratio  = 1.0f; ///< 1 / 缩放比（用于乘回原图尺度）
    float dw     = 0.0f; ///< 水平方向 letterbox 单边填充量（像素）
    float dh     = 0.0f; ///< 垂直方向 letterbox 单边填充量
    float height = 0;    ///< 原图高
    float width  = 0;    ///< 原图宽
};
}  // namespace obb
#endif  // POSE_NORMAL_COMMON_HPP
