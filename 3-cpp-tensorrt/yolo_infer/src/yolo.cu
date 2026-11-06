/**
 * @file    yolo.cu
 * @brief   YOLO family TensorRT inference and GPU post-processing.
 *
 * 5-reference/YOLOv8-seg samples typically use Type::V8Seg (proto mask + detection head).
 *
 * Overview:
 *  - Norm / NormType / ChannelType: pixel norm and channel order
 *  - decode_kernel_common / _v8: version-specific decode kernels
 *  - fast_nms_kernel: GPU NMS (IOU suppression)
 *  - warp_affine_bilinear_and_normalize_plane_kernel: letterbox + norm fused preprocess
 *  - decode_single_mask_kernel: V8Seg per-instance mask decode
 *  - AffineMatrix: letterbox 2x3 maps between image and network input
 *  - InstanceSegmentMap: pinned host mask storage
 *  - InferImpl: preprocess -> TRT -> decode -> NMS -> optional masks
 *  - load / loadraw: factories
 */
#include "infer.hpp"
#include "yolo.hpp"
#include "raw_dump.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <stdexcept>
#include <string>

namespace yolo {

using namespace std;

#define GPU_BLOCK_THREADS 512
#define checkRuntime(call)                                                                 \
  do {                                                                                     \
    auto ___call__ret_code__ = (call);                                                     \
    if (___call__ret_code__ != cudaSuccess) {                                              \
      INFO("CUDA Runtime error %s # %s, code = %s [ %d ]", #call,                              \
           cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__), \
           ___call__ret_code__);                                                           \
      abort();                                                                             \
    }                                                                                      \
  } while (0)

#define checkKernel(...)                 \
  do {                                   \
    { (__VA_ARGS__); }                   \
    checkRuntime(cudaPeekAtLastError()); \
  } while (0)

enum class NormType : int { None = 0, MeanStd = 1, AlphaBeta = 2 };

enum class ChannelType : int { None = 0, SwapRB = 1 };

struct Norm {
  float mean[3];
  float std[3];
  float alpha, beta;
  NormType type = NormType::None;
  ChannelType channel_type = ChannelType::None;

  // out = (x * alpha - mean) / std
  static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f,
                       ChannelType channel_type = ChannelType::None);

  // out = x * alpha + beta
  static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type = ChannelType::None);

  // None
  static Norm None();
};

Norm Norm::mean_std(const float mean[3], const float std[3], float alpha,
                    ChannelType channel_type) {
  Norm out;
  out.type = NormType::MeanStd;
  out.alpha = alpha;
  out.channel_type = channel_type;
  memcpy(out.mean, mean, sizeof(out.mean));
  memcpy(out.std, std, sizeof(out.std));
  return out;
}

Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type) {
  Norm out;
  out.type = NormType::AlphaBeta;
  out.alpha = alpha;
  out.beta = beta;
  out.channel_type = channel_type;
  return out;
}

Norm Norm::None() { return Norm(); }

const int NUM_BOX_ELEMENT = 8;
const int NUM_KPT = 17;
const int NUM_BOX_ELEMENT_POSE = NUM_BOX_ELEMENT + NUM_KPT * 3;  // 8 + 51 = 59
// OBB: append (cx, cy, ow, oh, angle) after the common 8 fields. Total = 13 floats.
// keepflag is set to 1 by the decode kernel; final NMS runs on CPU via
// cv::dnn::NMSBoxes(RotatedRect).
const int NUM_BOX_ELEMENT_OBB = NUM_BOX_ELEMENT + 5;  // 8 + 5 = 13
const int MAX_IMAGE_BOXES = 1024;
inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align; }

static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox,
                                               float *oy) {
  *ox = matrix[0] * x + matrix[1] * y + matrix[2];
  *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel_common(float *predict, int num_bboxes, int num_classes,
                                            int output_cdim, float confidence_threshold,
                                            float *invert_affine_matrix, float *parray,
                                            int MAX_IMAGE_BOXES) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= num_bboxes) return;

  float *pitem = predict + output_cdim * position;
  float objectness = pitem[4];
  if (objectness < confidence_threshold) return;

  float *class_confidence = pitem + 5;
  float confidence = *class_confidence++;
  int label = 0;
  for (int i = 1; i < num_classes; ++i, ++class_confidence) {
    if (*class_confidence > confidence) {
      confidence = *class_confidence;
      label = i;
    }
  }

  confidence *= objectness;
  if (confidence < confidence_threshold) return;

  int index = atomicAdd(parray, 1);
  if (index >= MAX_IMAGE_BOXES) return;

  float cx = *pitem++;
  float cy = *pitem++;
  float width = *pitem++;
  float height = *pitem++;
  float left = cx - width * 0.5f;
  float top = cy - height * 0.5f;
  float right = cx + width * 0.5f;
  float bottom = cy + height * 0.5f;
  affine_project(invert_affine_matrix, left, top, &left, &top);
  affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

  float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
  *pout_item++ = left;
  *pout_item++ = top;
  *pout_item++ = right;
  *pout_item++ = bottom;
  *pout_item++ = confidence;
  *pout_item++ = label;
  *pout_item++ = 1;
}

static __global__ void decode_kernel_v8(float *predict, int num_bboxes, int num_classes,
                                        int output_cdim, float confidence_threshold,
                                        float *invert_affine_matrix, float *parray,
                                        int MAX_IMAGE_BOXES) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= num_bboxes) return;

  float *pitem = predict + output_cdim * position;
  float *class_confidence = pitem + 4;
  float confidence = *class_confidence++;
  int label = 0;
  for (int i = 1; i < num_classes; ++i, ++class_confidence) {
    if (*class_confidence > confidence) {
      confidence = *class_confidence;
      label = i;
    }
  }
  if (confidence < confidence_threshold) return;

  int index = atomicAdd(parray, 1);
  if (index >= MAX_IMAGE_BOXES) return;

  float cx = *pitem++;
  float cy = *pitem++;
  float width = *pitem++;
  float height = *pitem++;
  float left = cx - width * 0.5f;
  float top = cy - height * 0.5f;
  float right = cx + width * 0.5f;
  float bottom = cy + height * 0.5f;
  affine_project(invert_affine_matrix, left, top, &left, &top);
  affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

  float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
  *pout_item++ = left;
  *pout_item++ = top;
  *pout_item++ = right;
  *pout_item++ = bottom;
  *pout_item++ = confidence;
  *pout_item++ = label;
  *pout_item++ = 1;
  *pout_item++ = position;
}

// Pose decode: layout per anchor = [cx,cy,w,h, score, k1x,k1y,k1v,..., k17x,k17y,k17v]
// Single class (person), num_classes_=1.
static __global__ void decode_kernel_v8_pose(float *predict, int num_bboxes, int output_cdim,
                                             float confidence_threshold,
                                             float *invert_affine_matrix, float *parray,
                                             int MAX_IMAGE_BOXES) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= num_bboxes) return;

  float *pitem = predict + output_cdim * position;
  float confidence = pitem[4];
  if (confidence < confidence_threshold) return;

  int index = atomicAdd(parray, 1);
  if (index >= MAX_IMAGE_BOXES) return;

  float cx = pitem[0];
  float cy = pitem[1];
  float width = pitem[2];
  float height = pitem[3];
  float left = cx - width * 0.5f;
  float top = cy - height * 0.5f;
  float right = cx + width * 0.5f;
  float bottom = cy + height * 0.5f;
  affine_project(invert_affine_matrix, left, top, &left, &top);
  affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

  float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT_POSE;
  pout_item[0] = left;
  pout_item[1] = top;
  pout_item[2] = right;
  pout_item[3] = bottom;
  pout_item[4] = confidence;
  pout_item[5] = 0;        // label (single class)
  pout_item[6] = 1;        // keepflag
  pout_item[7] = position; // row index (unused for pose)

  // 17 keypoints, each (x, y, visibility); map (x,y) back to original image space
  float *pkpt_src = pitem + 5;
  float *pkpt_dst = pout_item + 8;
  for (int k = 0; k < NUM_KPT; ++k) {
    float kx = pkpt_src[k * 3 + 0];
    float ky = pkpt_src[k * 3 + 1];
    float kv = pkpt_src[k * 3 + 2];
    float ox, oy;
    affine_project(invert_affine_matrix, kx, ky, &ox, &oy);
    pkpt_dst[k * 3 + 0] = ox;
    pkpt_dst[k * 3 + 1] = oy;
    pkpt_dst[k * 3 + 2] = kv;
  }
}

// OBB decode (transposed layout): per anchor = [cx,cy,w,h, c0..c{nc-1}, angle]
// output_cdim = 4 + num_classes + 1. (cx,cy) is mapped back via d2i,
// w/h is scaled by d2i[0] (letterbox affine is uniform-scale + translate,
// d2i[0] == d2i[4] == 1/scale). The AABB enclosing the rotated rect is written
// to [0..3] as a fallback for the existing list/render path.
static __global__ void decode_kernel_v8_obb(float *predict, int num_bboxes, int num_classes,
                                            int output_cdim, float confidence_threshold,
                                            float *invert_affine_matrix, float *parray,
                                            int MAX_IMAGE_BOXES) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= num_bboxes) return;

  float *pitem = predict + output_cdim * position;
  float *class_confidence = pitem + 4;
  float confidence = class_confidence[0];
  int label = 0;
  for (int i = 1; i < num_classes; ++i) {
    if (class_confidence[i] > confidence) {
      confidence = class_confidence[i];
      label = i;
    }
  }
  if (confidence < confidence_threshold) return;

  int index = atomicAdd(parray, 1);
  if (index >= MAX_IMAGE_BOXES) return;

  float cx_in = pitem[0];
  float cy_in = pitem[1];
  float w_in = pitem[2];
  float h_in = pitem[3];
  float angle = pitem[4 + num_classes];  // radians

  float cx_out, cy_out;
  affine_project(invert_affine_matrix, cx_in, cy_in, &cx_out, &cy_out);
  float scale = invert_affine_matrix[0];  // letterbox: uniform scale
  float ow = w_in * scale;
  float oh = h_in * scale;

  // Enclosing AABB for the rotated rect (fallback for visualization)
  float cosA = fabsf(cosf(angle));
  float sinA = fabsf(sinf(angle));
  float bw = ow * cosA + oh * sinA;
  float bh = ow * sinA + oh * cosA;
  float left = cx_out - bw * 0.5f;
  float top = cy_out - bh * 0.5f;
  float right = cx_out + bw * 0.5f;
  float bottom = cy_out + bh * 0.5f;

  float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT_OBB;
  pout_item[0] = left;
  pout_item[1] = top;
  pout_item[2] = right;
  pout_item[3] = bottom;
  pout_item[4] = confidence;
  pout_item[5] = (float)label;
  pout_item[6] = 1;  // keepflag=1, rotated NMS runs on CPU later
  pout_item[7] = (float)position;
  pout_item[8] = cx_out;
  pout_item[9] = cy_out;
  pout_item[10] = ow;
  pout_item[11] = oh;
  pout_item[12] = angle;
}

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft,
                                float btop, float bright, float bbottom) {
  float cleft = max(aleft, bleft);
  float ctop = max(atop, btop);
  float cright = min(aright, bright);
  float cbottom = min(abottom, bbottom);

  float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
  if (c_area == 0.0f) return 0.0f;

  float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
  float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
  return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float *bboxes, int MAX_IMAGE_BOXES, float threshold,
                                       int box_stride) {
  int position = (blockDim.x * blockIdx.x + threadIdx.x);
  int count = min((int)*bboxes, MAX_IMAGE_BOXES);
  if (position >= count) return;

  // left, top, right, bottom, confidence, class, keepflag
  float *pcurrent = bboxes + 1 + position * box_stride;
  for (int i = 0; i < count; ++i) {
    float *pitem = bboxes + 1 + i * box_stride;
    if (i == position || pcurrent[5] != pitem[5]) continue;

    if (pitem[4] >= pcurrent[4]) {
      if (pitem[4] == pcurrent[4] && i < position) continue;

      float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1],
                          pitem[2], pitem[3]);

      if (iou > threshold) {
        pcurrent[6] = 0;
        return;
      }
    }
  }
}

static dim3 grid_dims(int numJobs) {
  int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
  return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

static dim3 block_dims(int numJobs) {
  return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

static void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                  float confidence_threshold, float nms_threshold,
                                  float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                                  Type type, cudaStream_t stream) {
  auto grid = grid_dims(num_bboxes);
  auto block = block_dims(num_bboxes);

  int box_stride = NUM_BOX_ELEMENT;
  if (type == Type::V8Pose) {
    box_stride = NUM_BOX_ELEMENT_POSE;
    checkKernel(decode_kernel_v8_pose<<<grid, block, 0, stream>>>(
        predict, num_bboxes, output_cdim, confidence_threshold, invert_affine_matrix, parray,
        MAX_IMAGE_BOXES));
  } else if (type == Type::V8OBB) {
    box_stride = NUM_BOX_ELEMENT_OBB;
    checkKernel(decode_kernel_v8_obb<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
        parray, MAX_IMAGE_BOXES));
  } else if (type == Type::V8 || type == Type::V8Seg) {
    checkKernel(decode_kernel_v8<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
        parray, MAX_IMAGE_BOXES));
  } else {
    checkKernel(decode_kernel_common<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
        parray, MAX_IMAGE_BOXES));
  }

  // OBB performs rotated IoU NMS on CPU; skip GPU fast_nms_kernel here.
  if (type != Type::V8OBB) {
    grid = grid_dims(MAX_IMAGE_BOXES);
    block = block_dims(MAX_IMAGE_BOXES);
    checkKernel(
        fast_nms_kernel<<<grid, block, 0, stream>>>(parray, MAX_IMAGE_BOXES, nms_threshold, box_stride));
  }
}

static __global__ void warp_affine_bilinear_and_normalize_plane_kernel(
    uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width,
    int dst_height, uint8_t const_value_st, float *warp_affine_matrix_2_3, Norm norm) {
  int dx = blockDim.x * blockIdx.x + threadIdx.x;
  int dy = blockDim.y * blockIdx.y + threadIdx.y;
  if (dx >= dst_width || dy >= dst_height) return;

  float m_x1 = warp_affine_matrix_2_3[0];
  float m_y1 = warp_affine_matrix_2_3[1];
  float m_z1 = warp_affine_matrix_2_3[2];
  float m_x2 = warp_affine_matrix_2_3[3];
  float m_y2 = warp_affine_matrix_2_3[4];
  float m_z2 = warp_affine_matrix_2_3[5];

  float src_x = m_x1 * dx + m_y1 * dy + m_z1;
  float src_y = m_x2 * dx + m_y2 * dy + m_z2;
  float c0, c1, c2;

  if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
    // out of range
    c0 = const_value_st;
    c1 = const_value_st;
    c2 = const_value_st;
  } else {
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
    float ly = src_y - y_low;
    float lx = src_x - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    uint8_t *v1 = const_value;
    uint8_t *v2 = const_value;
    uint8_t *v3 = const_value;
    uint8_t *v4 = const_value;
    if (y_low >= 0) {
      if (x_low >= 0) v1 = src + y_low * src_line_size + x_low * 3;

      if (x_high < src_width) v2 = src + y_low * src_line_size + x_high * 3;
    }

    if (y_high < src_height) {
      if (x_low >= 0) v3 = src + y_high * src_line_size + x_low * 3;

      if (x_high < src_width) v4 = src + y_high * src_line_size + x_high * 3;
    }

    // same to opencv
    c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
    c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
    c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
  }

  if (norm.channel_type == ChannelType::SwapRB) {
    float t = c2;
    c2 = c0;
    c0 = t;
  }

  if (norm.type == NormType::MeanStd) {
    c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
    c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
    c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
  } else if (norm.type == NormType::AlphaBeta) {
    c0 = c0 * norm.alpha + norm.beta;
    c1 = c1 * norm.alpha + norm.beta;
    c2 = c2 * norm.alpha + norm.beta;
  }

  int area = dst_width * dst_height;
  float *pdst_c0 = dst + dy * dst_width + dx;
  float *pdst_c1 = pdst_c0 + area;
  float *pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;
}

static void warp_affine_bilinear_and_normalize_plane(uint8_t *src, int src_line_size, int src_width,
                                                     int src_height, float *dst, int dst_width,
                                                     int dst_height, float *matrix_2_3,
                                                     uint8_t const_value, const Norm &norm,
                                                     cudaStream_t stream) {
  dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
  dim3 block(32, 32);

  checkKernel(warp_affine_bilinear_and_normalize_plane_kernel<<<grid, block, 0, stream>>>(
      src, src_line_size, src_width, src_height, dst, dst_width, dst_height, const_value,
      matrix_2_3, norm));
}

static __global__ void decode_single_mask_kernel(int left, int top, float *mask_weights,
                                                 float *mask_predict, int mask_width,
                                                 int mask_height, unsigned char *mask_out,
                                                 int mask_dim, int out_width, int out_height) {
  // mask_predict to mask_out
  // mask_weights @ mask_predict
  int dx = blockDim.x * blockIdx.x + threadIdx.x;
  int dy = blockDim.y * blockIdx.y + threadIdx.y;
  if (dx >= out_width || dy >= out_height) return;

  int sx = left + dx;
  int sy = top + dy;
  if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height) {
    mask_out[dy * out_width + dx] = 0;
    return;
  }

  float cumprod = 0;
  for (int ic = 0; ic < mask_dim; ++ic) {
    float cval = mask_predict[(ic * mask_height + sy) * mask_width + sx];
    float wval = mask_weights[ic];
    cumprod += cval * wval;
  }

  float alpha = 1.0f / (1.0f + exp(-cumprod));
  mask_out[dy * out_width + dx] = alpha * 255;
}

static void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                               int mask_width, int mask_height, unsigned char *mask_out,
                               int mask_dim, int out_width, int out_height, cudaStream_t stream) {
  // mask_weights is mask_dim(32 element) gpu pointer
  dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
  dim3 block(32, 32);

  checkKernel(decode_single_mask_kernel<<<grid, block, 0, stream>>>(
      left, top, mask_weights, mask_predict, mask_width, mask_height, mask_out, mask_dim, out_width,
      out_height));
}

const char *type_name(Type type) {
  switch (type) {
    case Type::V5:
      return "YoloV5";
    case Type::V3:
      return "YoloV3";
    case Type::V7:
      return "YoloV7";
    case Type::X:
      return "YoloX";
    case Type::V8:
      return "YoloV8";
    case Type::V8Seg:
      return "YoloV8Seg";
    case Type::V8Pose:
      return "YoloV8Pose";
    case Type::V8OBB:
      return "YoloV8OBB";
    default:
      return "Unknow";
  }
}

struct AffineMatrix {
  float i2d[6];
  float d2i[6];

  void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to) {
    float scale_x = get<0>(to) / (float)get<0>(from);
    float scale_y = get<1>(to) / (float)get<1>(from);
    float scale = std::min(scale_x, scale_y);
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

    double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
    D = D != 0. ? double(1.) / D : double(0.);
    double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
    double b1 = -A11 * i2d[2] - A12 * i2d[5];
    double b2 = -A21 * i2d[2] - A22 * i2d[5];

    d2i[0] = A11;
    d2i[1] = A12;
    d2i[2] = b1;
    d2i[3] = A21;
    d2i[4] = A22;
    d2i[5] = b2;
  }
};

InstanceSegmentMap::InstanceSegmentMap(int width, int height) {
  this->width = width;
  this->height = height;
  checkRuntime(cudaMallocHost(&this->data, width * height));
}

InstanceSegmentMap::~InstanceSegmentMap() {
  if (this->data) {
    checkRuntime(cudaFreeHost(this->data));
    this->data = nullptr;
  }
  this->width = 0;
  this->height = 0;
}

class InferImpl : public Infer {
 public:
  shared_ptr<trt::Infer> trt_;
  string engine_file_;
  Type type_;
  float confidence_threshold_;
  float nms_threshold_;
  vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
  trt::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
  trt::Memory<float> segment_predict_;
  int network_input_width_, network_input_height_;
  Norm normalize_;
  vector<int> bbox_head_dims_;
  vector<int> segment_head_dims_;
  int num_classes_ = 0;
  bool has_segment_ = false;
  bool is_pose_ = false;
  bool is_obb_ = false;
  int box_stride_ = NUM_BOX_ELEMENT;
  bool isdynamic_model_ = false;
  vector<shared_ptr<trt::Memory<unsigned char>>> box_segment_cache_;

  virtual ~InferImpl() = default;

  void adjust_memory(int batch_size) {
    // the inference batch_size
    size_t input_numel = network_input_width_ * network_input_height_ * 3;
    input_buffer_.gpu(batch_size * input_numel);
    bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
    output_boxarray_.gpu(batch_size * (32 + MAX_IMAGE_BOXES * box_stride_));
    output_boxarray_.cpu(batch_size * (32 + MAX_IMAGE_BOXES * box_stride_));

    if (has_segment_)
      segment_predict_.gpu(batch_size * segment_head_dims_[1] * segment_head_dims_[2] *
                           segment_head_dims_[3]);

    if ((int)preprocess_buffers_.size() < batch_size) {
      for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
        preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char>>());
    }
  }

  void preprocess(int ibatch, const Image &image,
                  shared_ptr<trt::Memory<unsigned char>> preprocess_buffer, AffineMatrix &affine,
                  void *stream = nullptr) {
    affine.compute(make_tuple(image.width, image.height),
                   make_tuple(network_input_width_, network_input_height_));

    size_t input_numel = network_input_width_ * network_input_height_ * 3;
    float *input_device = input_buffer_.gpu() + ibatch * input_numel;

    // Use Image::line_size as the source row stride in bytes. Buffers from Qt
    // (QImage::Format_RGB888) are commonly aligned to 4 bytes per row, so for
    // width=810 the real stride is 2432 instead of 810*3=2430. Using width*3
    // here would shift each row by the padding amount and tear the picture
    // along the row direction, breaking the GPU letterbox/normalize kernel.
    const int src_line_size_bytes =
        (image.line_size > 0) ? image.line_size : (image.width * 3);
    size_t size_image = static_cast<size_t>(src_line_size_bytes) * image.height;
    size_t size_matrix = upbound(sizeof(affine.d2i), 32);
    uint8_t *gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
    float *affine_matrix_device = (float *)gpu_workspace;
    uint8_t *image_device = gpu_workspace + size_matrix;

    uint8_t *cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
    float *affine_matrix_host = (float *)cpu_workspace;
    uint8_t *image_host = cpu_workspace + size_matrix;

    // speed up
    cudaStream_t stream_ = (cudaStream_t)stream;
    memcpy(image_host, image.bgrptr, size_image);
    memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
    checkRuntime(
        cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
    checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                                 cudaMemcpyHostToDevice, stream_));

    warp_affine_bilinear_and_normalize_plane(image_device, src_line_size_bytes, image.width,
                                             image.height, input_device, network_input_width_,
                                             network_input_height_, affine_matrix_device, 114,
                                             normalize_, stream_);
  }

  bool load(const string &engine_file, Type type, float confidence_threshold, float nms_threshold) {
    trt_ = trt::load(engine_file);
    if (trt_ == nullptr) return false;

    trt_->print();

    this->type_ = type;
    this->confidence_threshold_ = confidence_threshold;
    this->nms_threshold_ = nms_threshold;

    auto input_dim = trt_->static_dims(0);
    const int nb = trt_->num_bindings();
    // 1 input + N outputs. SEG needs 2 outputs (proto + det), others need 1.
    const int nb_outputs = nb - 1;
    has_segment_ = type == Type::V8Seg;
    if (has_segment_) {
      if (nb_outputs < 2) {
        throw std::runtime_error(
            "Task/model mismatch: V8Seg requires an engine with 2 outputs "
            "(proto + det), but the loaded engine has " +
            std::to_string(nb_outputs) +
            " output(s). Please pick a segmentation engine, or switch the task to DET/POSE/OBB.");
      }
      bbox_head_dims_ = trt_->static_dims(2);
      segment_head_dims_ = trt_->static_dims(1);
    } else {
      if (nb_outputs < 1) {
        throw std::runtime_error("Engine has no output binding.");
      }
      if (nb_outputs > 1) {
        // Most likely a SEG engine selected with a non-SEG task.
        throw std::runtime_error(
            "Task/model mismatch: the loaded engine has " +
            std::to_string(nb_outputs) +
            " outputs (looks like a SEG engine), but the selected task is not SEG. "
            "Please switch the task to SEG, or load a single-output engine.");
      }
      bbox_head_dims_ = trt_->static_dims(1);
    }
    network_input_width_ = input_dim[3];
    network_input_height_ = input_dim[2];
    isdynamic_model_ = trt_->has_dynamic_dim();

    if (type == Type::V5 || type == Type::V3 || type == Type::V7) {
      normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
      num_classes_ = bbox_head_dims_[2] - 5;
    } else if (type == Type::V8) {
      normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
      num_classes_ = bbox_head_dims_[2] - 4;
      if (num_classes_ <= 0) {
        throw std::runtime_error(
            "Task/model mismatch: V8 (DET) expects channels = 4 + nc, "
            "got " + std::to_string(bbox_head_dims_[2]) +
            ". The engine layout must be transposed [1, anchors, 4+nc]. "
            "If this is a POSE/OBB engine, please switch the task accordingly.");
      }
      // Heuristic: pose=56, obb=4+nc+1 (e.g. dota=20). Warn user via exception.
      if (bbox_head_dims_[2] == 56) {
        throw std::runtime_error(
            "Task/model mismatch: channels=56 looks like a POSE engine. "
            "Please switch the task to POSE.");
      }
    } else if (type == Type::V8Seg) {
      normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
      num_classes_ = bbox_head_dims_[2] - 4 - segment_head_dims_[1];
      if (num_classes_ <= 0) {
        throw std::runtime_error(
            "Task/model mismatch: V8Seg channel layout invalid (det=" +
            std::to_string(bbox_head_dims_[2]) + ", proto_dim=" +
            std::to_string(segment_head_dims_[1]) + ").");
      }
    } else if (type == Type::V8Pose) {
      normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
      num_classes_ = 1;  // pose: single class (person)
      is_pose_ = true;
      box_stride_ = NUM_BOX_ELEMENT_POSE;
      // Sanity check: expected channels = 4 + 1 + 17*3 = 56 (transposed layout: [N, anchors, 56])
      if (bbox_head_dims_[2] != 56) {
        throw std::runtime_error(
            "Task/model mismatch: V8Pose expects channels=56 (4+1+17*3), got " +
            std::to_string(bbox_head_dims_[2]) +
            ". Make sure the engine is built from a transposed pose ONNX "
            "(yolo_pose_export_transd.py), and the selected task matches the model.");
      }
    } else if (type == Type::V8OBB) {
      normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
      // OBB channels = 4 + nc + 1 (last channel is angle in radians)
      num_classes_ = bbox_head_dims_[2] - 4 - 1;
      is_obb_ = true;
      box_stride_ = NUM_BOX_ELEMENT_OBB;
      if (num_classes_ <= 0) {
        throw std::runtime_error(
            "Task/model mismatch: V8OBB expects channels = 4 + nc + 1, got " +
            std::to_string(bbox_head_dims_[2]) +
            ". The engine layout must be transposed [1, anchors, 4+nc+1]. "
            "If this is a DET/POSE engine, please switch the task accordingly.");
      }
      if (bbox_head_dims_[2] == 56) {
        throw std::runtime_error(
            "Task/model mismatch: channels=56 looks like a POSE engine, not OBB. "
            "Please switch the task to POSE.");
      }
    } else if (type == Type::X) {
      // float mean[] = {0.485, 0.456, 0.406};
      // float std[]  = {0.229, 0.224, 0.225};
      // normalize_ = Norm::mean_std(mean, std, 1/255.0f, ChannelType::SwapRB);
      normalize_ = Norm::None();
      num_classes_ = bbox_head_dims_[2] - 5;
    } else {
      INFO("Unsupport type %d", type);
    }
    return true;
  }

  virtual BoxArray forward(const Image &image, void *stream = nullptr) override {
    auto output = forwards({image}, stream);
    if (output.empty()) return {};
    return output[0];
  }

  virtual vector<BoxArray> forwards(const vector<Image> &images, void *stream = nullptr) override {
    int num_image = images.size();
    if (num_image == 0) return {};

    auto input_dims = trt_->static_dims(0);
    int infer_batch_size = input_dims[0];
    if (infer_batch_size != num_image) {
      if (isdynamic_model_) {
        infer_batch_size = num_image;
        input_dims[0] = num_image;
        if (!trt_->set_run_dims(0, input_dims)) return {};
      } else {
        if (infer_batch_size < num_image) {
          INFO(
              "When using static shape model, number of images[%d] must be "
              "less than or equal to the maximum batch[%d].",
              num_image, infer_batch_size);
          return {};
        }
      }
    }
    adjust_memory(infer_batch_size);

    vector<AffineMatrix> affine_matrixs(num_image);
    cudaStream_t stream_ = (cudaStream_t)stream;
    for (int i = 0; i < num_image; ++i)
      preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);

    float *bbox_output_device = bbox_predict_.gpu();
    vector<void *> bindings{input_buffer_.gpu(), bbox_output_device};

    if (has_segment_) {
      bindings = {input_buffer_.gpu(), segment_predict_.gpu(), bbox_output_device};
    }

    if (!trt_->forward(bindings, stream)) {
      INFO("Failed to tensorRT forward.");
      return {};
    }

    // ---------------------------------------------------------------------
    // 诊断导出（TRTFORGE_RAW_DUMP）：把喂给引擎的输入张量与后处理前的
    // 输出张量拷回主机落盘，供 tools/compare_raw_tensors.py 与 ONNX 侧对齐。
    // 未设置环境变量时整段被跳过，正常推理路径不变。
    // ---------------------------------------------------------------------
    unsigned dump_run = 0;
    if (oi::dump::enabled()) {
      dump_run = oi::dump::begin_run();

      // 只为导出等一次 GPU；正常路径不在这里引入同步点
      checkRuntime(cudaStreamSynchronize(stream_));

      const size_t input_numel = static_cast<size_t>(network_input_width_) *
                                 network_input_height_ * 3;
      std::vector<float> input_host(input_numel * num_image);
      checkRuntime(cudaMemcpy(input_host.data(), input_buffer_.gpu(),
                              input_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
      std::vector<int64_t> input_shape{num_image, 3, network_input_height_,
                                       network_input_width_};
      oi::dump::write(dump_run, "yolo_infer", "input", 0, "images", input_shape,
                      input_host.data(), input_host.size(), "float32", "TensorRT");

      const size_t bbox_numel = static_cast<size_t>(bbox_head_dims_[1]) *
                                bbox_head_dims_[2] * num_image;
      std::vector<float> bbox_host(bbox_numel);
      checkRuntime(cudaMemcpy(bbox_host.data(), bbox_output_device,
                              bbox_numel * sizeof(float), cudaMemcpyDeviceToHost));
      std::vector<int64_t> bbox_shape{num_image, bbox_head_dims_[1], bbox_head_dims_[2]};
      oi::dump::write(dump_run, "yolo_infer", "output", 0, "output0", bbox_shape,
                      bbox_host.data(), bbox_host.size(), "float32", "TensorRT");

      if (has_segment_) {
        const size_t seg_numel = static_cast<size_t>(segment_head_dims_[1]) *
                                 segment_head_dims_[2] * segment_head_dims_[3] * num_image;
        std::vector<float> seg_host(seg_numel);
        checkRuntime(cudaMemcpy(seg_host.data(), segment_predict_.gpu(),
                                seg_numel * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<int64_t> seg_shape{num_image, segment_head_dims_[1],
                                       segment_head_dims_[2], segment_head_dims_[3]};
        oi::dump::write(dump_run, "yolo_infer", "output", 1, "output1", seg_shape,
                        seg_host.data(), seg_host.size(), "float32", "TensorRT");
      }
    }

    for (int ib = 0; ib < num_image; ++ib) {
      float *boxarray_device =
          output_boxarray_.gpu() + ib * (32 + MAX_IMAGE_BOXES * box_stride_);
      float *affine_matrix_device = (float *)preprocess_buffers_[ib]->gpu();
      float *image_based_bbox_output =
          bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
      checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
      decode_kernel_invoker(image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                            bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                            affine_matrix_device, boxarray_device, MAX_IMAGE_BOXES, type_, stream_);
    }
    checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(), output_boxarray_.gpu(),
                                 output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
    checkRuntime(cudaStreamSynchronize(stream_));

    // OBB: per-class rotated IoU NMS on CPU (aligned with ultralytics / triple-Mu).
    // The decode kernel sets keepflag=1 for every candidate; here we use
    // cv::dnn::NMSBoxes(RotatedRect) per label and clear keepflag for
    // suppressed entries.
    if (is_obb_) {
      for (int ib = 0; ib < num_image; ++ib) {
        float *parray =
            output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * box_stride_);
        int count = std::min(MAX_IMAGE_BOXES, (int)*parray);
        if (count <= 0) continue;

        // group by label, then run rotated NMS per class
        std::map<int, std::vector<int>> by_label;
        for (int i = 0; i < count; ++i) {
          float *pbox = parray + 1 + i * box_stride_;
          if ((int)pbox[6] != 1) continue;
          by_label[(int)pbox[5]].push_back(i);
        }
        // clear all first, then restore keepflag for indices that pass NMS
        for (int i = 0; i < count; ++i) {
          parray[1 + i * box_stride_ + 6] = 0;
        }

        for (auto &kv : by_label) {
          std::vector<cv::RotatedRect> rects;
          std::vector<float> scores;
          rects.reserve(kv.second.size());
          scores.reserve(kv.second.size());
          for (int idx : kv.second) {
            float *pbox = parray + 1 + idx * box_stride_;
            float cx = pbox[8], cy = pbox[9], ow = pbox[10], oh = pbox[11];
            float ang_deg = pbox[12] * 180.0f / (float)CV_PI;
            rects.emplace_back(cv::Point2f(cx, cy), cv::Size2f(ow, oh), ang_deg);
            scores.push_back(pbox[4]);
          }
          std::vector<int> keep;
          cv::dnn::NMSBoxes(rects, scores, confidence_threshold_, nms_threshold_, keep);
          for (int k : keep) {
            int idx = kv.second[k];
            parray[1 + idx * box_stride_ + 6] = 1;
          }
        }
      }
    }

    vector<BoxArray> arrout(num_image);
    int imemory = 0;
    INFO("num_image: %d", num_image);

    for (int ib = 0; ib < num_image; ++ib) {
      float *parray = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * box_stride_);
      int count = min(MAX_IMAGE_BOXES, (int)*parray);
      BoxArray &output = arrout[ib];
      output.reserve(count);

      //INFO("count: %d", count);
      for (int i = 0; i < count; ++i) {
        float *pbox = parray + 1 + i * box_stride_;
        int label = pbox[5];
        int keepflag = pbox[6];
        if (keepflag == 1) {
          Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
          if (is_pose_) {
            result_object_box.kps.resize(NUM_KPT * 3);
            const float *pkpt = pbox + 8;
            for (int k = 0; k < NUM_KPT * 3; ++k) result_object_box.kps[k] = pkpt[k];
          }
          if (is_obb_) {
            result_object_box.cx = pbox[8];
            result_object_box.cy = pbox[9];
            result_object_box.ow = pbox[10];
            result_object_box.oh = pbox[11];
            result_object_box.angle = pbox[12];
          }
          if (has_segment_) {
            int row_index = pbox[7];
            int mask_dim = segment_head_dims_[1];
            float *mask_weights = bbox_output_device +
                                  (ib * bbox_head_dims_[1] + row_index) * bbox_head_dims_[2] +
                                  num_classes_ + 4;

            float *mask_head_predict = segment_predict_.gpu();
            float left, top, right, bottom;
            float *i2d = affine_matrixs[ib].i2d;
            affine_project(i2d, pbox[0], pbox[1], &left, &top);
            affine_project(i2d, pbox[2], pbox[3], &right, &bottom);

            float box_width = right - left;
            float box_height = bottom - top;

            float scale_to_predict_x = segment_head_dims_[3] / (float)network_input_width_;
            float scale_to_predict_y = segment_head_dims_[2] / (float)network_input_height_;
            int mask_out_width = box_width * scale_to_predict_x + 0.5f;
            int mask_out_height = box_height * scale_to_predict_y + 0.5f;

            if (mask_out_width > 0 && mask_out_height > 0) {
              //INFO("Current size of box_segment_cache_: %d", (int)box_segment_cache_.size());
              if (imemory >= (int)box_segment_cache_.size()) {
                box_segment_cache_.push_back(std::make_shared<trt::Memory<unsigned char>>());
              }

              int bytes_of_mask_out = mask_out_width * mask_out_height;
              auto box_segment_output_memory = box_segment_cache_[imemory];
              result_object_box.seg =
                  make_shared<InstanceSegmentMap>(mask_out_width, mask_out_height);

              unsigned char *mask_out_device = box_segment_output_memory->gpu(bytes_of_mask_out);
              unsigned char *mask_out_host = result_object_box.seg->data;
              decode_single_mask(left * scale_to_predict_x, top * scale_to_predict_y, mask_weights,
                                 mask_head_predict + ib * segment_head_dims_[1] *
                                                         segment_head_dims_[2] *
                                                         segment_head_dims_[3],
                                 segment_head_dims_[3], segment_head_dims_[2], mask_out_device,
                                 mask_dim, mask_out_width, mask_out_height, stream_);

              checkRuntime(cudaMemcpyAsync(mask_out_host, mask_out_device,
                                           box_segment_output_memory->gpu_bytes(),
                                           cudaMemcpyDeviceToHost, stream_));
              // increase imemory
              imemory++;
            }
          }
          output.emplace_back(result_object_box);
        }
      }
    }

    if (has_segment_) checkRuntime(cudaStreamSynchronize(stream_));

    return arrout;
  }
};

Infer *loadraw(const std::string &engine_file, Type type, float confidence_threshold,
               float nms_threshold) {
  InferImpl *impl = new InferImpl();
  try {
    if (!impl->load(engine_file, type, confidence_threshold, nms_threshold)) {
      delete impl;
      return nullptr;
    }
  } catch (...) {
    delete impl;
    throw;  // 让上层 (yi_create) 捕获并写入 g_last_error
  }
  return impl;
}

shared_ptr<Infer> load(const string &engine_file, Type type, float confidence_threshold,
                       float nms_threshold) {
  return std::shared_ptr<InferImpl>(
      (InferImpl *)loadraw(engine_file, type, confidence_threshold, nms_threshold));
}

std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
  const int h_i = static_cast<int>(h * 6);
  const float f = h * 6 - h_i;
  const float p = v * (1 - s);
  const float q = v * (1 - f * s);
  const float t = v * (1 - (1 - f) * s);
  float r, g, b;
  switch (h_i) {
    case 0:
      r = v, g = t, b = p;
      break;
    case 1:
      r = q, g = v, b = p;
      break;
    case 2:
      r = p, g = v, b = t;
      break;
    case 3:
      r = p, g = q, b = v;
      break;
    case 4:
      r = t, g = p, b = v;
      break;
    case 5:
      r = v, g = p, b = q;
      break;
    default:
      r = 1, g = 1, b = 1;
      break;
  }
  return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255),
                    static_cast<uint8_t>(r * 255));
}

std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
  float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
  float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
  return hsv2bgr(h_plane, s_plane, 1);
}

};  // namespace yolo