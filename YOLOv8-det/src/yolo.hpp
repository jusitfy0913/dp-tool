/**
 * @file    yolo.hpp
 * @brief   YOLO 系列（YOLOv5/X/v3/v7/v8/v8-seg）TensorRT 推理的对外 C++ 接口。
 *
 * 本文件定义 yolo 命名空间下的：
 *  - Type                 : 枚举支持的 YOLO 模型类型；
 *  - Box / BoxArray       : 单张图的检测框结果；
 *  - Image                : 简化的 BGR 图像描述（裸指针+宽高）；
 *  - InstanceSegmentMap   : 实例分割掩码（仅 V8Seg 使用）；
 *  - Infer                : 抽象推理接口（forward / forwards）；
 *  - load / loadraw       : 工厂函数，根据 engine 文件创建 Infer 实现；
 *  - type_name            : Type → 字符串；
 *  - hsv2bgr / random_color : 可视化用的颜色工具。
 *
 * 实际实现见 yolo.cu，底层使用 trt::Infer（infer.hpp/infer.cu）。
 */
#ifndef __YOLO_HPP__
#define __YOLO_HPP__

#include <future>
#include <memory>
#include <string>
#include <vector>

namespace yolo {

/**
 * @enum  Type
 * @brief 支持的 YOLO 模型类型，决定输出张量布局与归一化方式。
 */
enum class Type : int {
  V5 = 0,
  X = 1,
  V3 = 2,
  V7 = 3,
  V8 = 5,    ///< yolov8 object detection
  V8Seg = 6  ///< yolov8 instance segmentation
};

/**
 * @struct InstanceSegmentMap
 * @brief  单个实例分割掩码（CPU 锁页内存）。
 *
 * 注意 width 需保证 8 字节对齐（width % 8 == 0），data 指向 width*height 字节。
 */
struct InstanceSegmentMap {
  int width = 0, height = 0;      ///< width 必须是 8 的倍数
  unsigned char *data = nullptr;  ///< 指向 width*height 字节的锁页内存

  /// 构造时分配 width*height 的 cudaMallocHost 缓冲
  InstanceSegmentMap(int width, int height);
  /// 析构时 cudaFreeHost
  virtual ~InstanceSegmentMap();
};

/**
 * @struct Box
 * @brief  单个检测框结果（坐标已还原到原图）。
 *
 * 对于 V8Seg，seg 指向对应实例的二值掩码；其它任务下 seg 为空。
 */
struct Box {
  float left, top, right, bottom, confidence;
  int class_label;
  std::shared_ptr<InstanceSegmentMap> seg;  ///< 仅 segment 任务有效

  Box() = default;
  Box(float left, float top, float right, float bottom, float confidence, int class_label)
      : left(left),
        top(top),
        right(right),
        bottom(bottom),
        confidence(confidence),
        class_label(class_label) {}
};

/**
 * @struct Image
 * @brief  推理输入：BGR 图像的裸指针 + 尺寸（避免引入 OpenCV 依赖）。
 */
struct Image {
  const void *bgrptr = nullptr;
  int width = 0, height = 0;

  Image() = default;
  Image(const void *bgrptr, int width, int height) : bgrptr(bgrptr), width(width), height(height) {}
};

/// 单张图的检测框列表
typedef std::vector<Box> BoxArray;

/**
 * @class Infer
 * @brief YOLO 推理抽象接口，使用方应通过 load() 获取实现。
 */
class Infer {
 public:
  /**
   * @brief 单张图推理。
   * @param image   输入图像
   * @param stream  可选 CUDA 流（cudaStream_t），nullptr 表示默认流
   * @return        检测框列表（已还原到原图坐标）
   */
  virtual BoxArray forward(const Image &image, void *stream = nullptr) = 0;

  /**
   * @brief 批量图像推理（输入为 vector<Image>）。
   */
  virtual std::vector<BoxArray> forwards(const std::vector<Image> &images,
                                         void *stream = nullptr) = 0;
};

/**
 * @brief 加载 engine 并返回 Infer 智能指针；失败返回 nullptr。
 * @param engine_file          序列化引擎文件路径
 * @param type                 模型类型（决定输出布局）
 * @param confidence_threshold 置信度阈值
 * @param nms_threshold        NMS IOU 阈值
 */
std::shared_ptr<Infer> load(const std::string &engine_file, Type type,
                            float confidence_threshold = 0.25f, float nms_threshold = 0.5f);

/**
 * @brief 同 load()，但返回裸指针，由调用方负责 delete。
 *        通常用于通过 C ABI 暴露给 Python ctypes 的场景（见 detect.cpp 中 Init/Detect）。
 */
Infer *loadraw(const std::string &engine_file, Type type, 
							float confidence_threshold = 0.25f, float nms_threshold = 0.5f);

/// Type → 可读字符串（"YoloV5" / "YoloV8" 等）。
const char *type_name(Type type);

/// HSV (h,s,v ∈ [0,1]) 转 BGR uint8 三元组。
std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);

/// 根据类别 id 生成稳定的可视化颜色。
std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);
};  // namespace yolo

#endif  // __YOLO_HPP__