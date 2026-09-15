/**
 * @file    yolo.hpp
 * @brief   C++ API for YOLO family (YOLOv5/X/v3/v7/v8/v8-seg) TensorRT inference.
 *
 * This repo's YOLOv8-seg example typically uses Type::V8Seg (boxes + per-instance mask).
 *
 * Declares in namespace yolo:
 *  - Type, Box, BoxArray, Image, InstanceSegmentMap
 *  - Infer, load, loadraw, type_name, hsv2bgr, random_color
 *
 * Implementation: yolo.cu, built on trt::Infer (infer.hpp / infer.cu).
 */
#ifndef __YOLO_HPP__
#define __YOLO_HPP__

#include <future>
#include <memory>
#include <string>
#include <vector>

namespace yolo {

enum class Type : int {
  V5 = 0,
  X = 1,
  V3 = 2,
  V7 = 3,
  V8 = 5,
  V8Seg = 6,
  V8Pose = 7,
  V8OBB = 8
};

/**
 * @struct InstanceSegmentMap
 * @brief Single instance mask in pinned host memory. width should be 8-byte aligned.
 */
struct InstanceSegmentMap {
  int width = 0, height = 0;
  unsigned char *data = nullptr;

  InstanceSegmentMap(int width, int height);
  virtual ~InstanceSegmentMap();
};

/**
 * @struct Box
 * @brief One detection; coords are in original image space. V8Seg may set seg.
 */
struct Box {
  float left, top, right, bottom, confidence;
  int class_label;
  std::shared_ptr<InstanceSegmentMap> seg;
  std::vector<float> kps;  ///< Pose keypoints: 17*(x,y,visibility) when V8Pose
  // Oriented bounding box (V8OBB only). Default 0 for other tasks.
  // left/top/right/bottom also receive the enclosing AABB so existing
  // list/render paths keep working.
  float cx = 0.f, cy = 0.f;     ///< OBB center in original image space
  float ow = 0.f, oh = 0.f;     ///< OBB oriented width/height in original image space
  float angle = 0.f;            ///< OBB rotation angle in radians

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
 * @brief BGR 输入图像视图（uint8 x 3 通道）。
 *
 * - bgrptr   : 指向 BGR 像素首地址，按行存储；
 * - width    : 像素宽度；
 * - height   : 像素高度；
 * - line_size: 每行字节数（与 OpenCV Mat::step、Qt QImage::bytesPerLine() 等价的概念）。
 *              紧凑缓冲：传 width * 3 或 0；
 *              带 padding：例如 QImage::Format_RGB888 通常按 4 字节对齐，
 *              宽 810 时 width*3=2430，但 bytesPerLine()=2432，必须传真实的 2432。
 *              line_size == 0 时由构造函数自动取 width * 3，与历史「紧凑 BGR」调用完全兼容。
 */
struct Image {
  const void *bgrptr = nullptr;
  int width = 0, height = 0;
  int line_size = 0;  ///< 每行字节数；0 表示紧凑（width*3）

  Image() = default;
  Image(const void *bgrptr, int width, int height, int line_size = 0)
      : bgrptr(bgrptr),
        width(width),
        height(height),
        line_size(line_size > 0 ? line_size : width * 3) {}
};

typedef std::vector<Box> BoxArray;

class Infer {
 public:
  virtual BoxArray forward(const Image &image, void *stream = nullptr) = 0;
  virtual std::vector<BoxArray> forwards(const std::vector<Image> &images,
                                         void *stream = nullptr) = 0;
};

std::shared_ptr<Infer> load(const std::string &engine_file, Type type,
                            float confidence_threshold = 0.25f, float nms_threshold = 0.5f);

Infer *loadraw(const std::string &engine_file, Type type, float confidence_threshold = 0.25f,
               float nms_threshold = 0.5f);

const char *type_name(Type type);

std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);
std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);
};  // namespace yolo

#endif  // __YOLO_HPP__
