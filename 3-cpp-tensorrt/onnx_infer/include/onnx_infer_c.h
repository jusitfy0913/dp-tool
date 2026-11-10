#ifndef ONNX_INFER_C_H
#define ONNX_INFER_C_H

/*
 * onnx_infer 的 C ABI。
 *
 * 设计约束（与 yolo_infer_c.h / anomalib_infer_c.h 保持一致）：
 *   1. 只暴露 C 语法与固定宽度整数，外部 C++、C#、Python ctypes 都能直接声明；
 *   2. 结果结构体内存由句柄持有，只在下一次推理或销毁前有效，调用方需立即深拷贝；
 *   3. 一个句柄不允许多线程同时调用推理，需要并发请创建多个句柄；
 *   4. 路径参数一律 UTF-8，Windows 内部转宽字符后访问文件。
 *
 * 结构体布局刻意与既有结构体逐字段对齐，Python 侧可以复用同一份 ctypes 定义：
 *   oi_box            == yi_box
 *   oi_frame_stat     == yi_frame_stat / ai_frame_stat
 *   oi_anomaly_result == ai_result
 * 一旦发布即视为 ABI，只允许在末尾追加字段。
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ONNX_INFER_EXPORTS
#define OI_API __declspec(dllexport)
#else
#define OI_API __declspec(dllimport)
#endif

typedef void* oi_handle;

/* 任务类型。0~3 与 yi_task 取值一致，便于适配层按同一张表分派。
 * OI_CLASSIFY 是 v1.1.0 追加的第五项：只在本 DLL 存在，yolo_infer.dll 没有对应任务。 */
typedef enum {
    OI_DETECT   = 0,
    OI_SEGMENT  = 1,
    OI_POSE     = 2,
    OI_OBB      = 3,
    OI_CLASSIFY = 4
} oi_task;

/* 返回码。0 表示成功，其余可用 oi_last_error() 取可读原因。 */
typedef enum {
    OI_OK = 0,
    OI_ERR_INVALID_ARGUMENT = 1,
    OI_ERR_MODEL = 2,
    OI_ERR_RUNTIME = 3,
    OI_ERR_INFERENCE = 4
} oi_status;

typedef struct {
    float left, top, right, bottom;
    float confidence;
    int   class_label;
    /* 分割掩码：has_mask=1 时 mask_data 有效，指向 mask_w * mask_h 的单通道数据。
     * 数据由句柄持有，仅在下一次推理前有效。 */
    int   has_mask;
    int   mask_x, mask_y;
    int   mask_w, mask_h;
    const unsigned char* mask_data;
    /* 姿态关键点：num_kpts 个三元组 (x, y, visibility)，总长度 num_kpts * 3。 */
    int   num_kpts;
    const float* kpts;
    /* 旋转框：is_obb=1 时以下字段有效，角度单位为弧度。 */
    int   is_obb;
    float cx, cy;
    float ow, oh;
    float angle;
} oi_box;

typedef struct {
    int    src_w, src_h;
    int    count;
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double total_ms;
} oi_frame_stat;

typedef struct {
    /* 原图 RGB888，长度为 source_h * source_stride。 */
    const unsigned char* source_rgb;
    int source_w;
    int source_h;
    int source_stride;

    /* 热力图叠加图 RGB888。 */
    const unsigned char* overlay_rgb;
    int overlay_w;
    int overlay_h;
    int overlay_stride;

    /* 红色异常轮廓图 RGB888。 */
    const unsigned char* mask_rgb;
    int mask_w;
    int mask_h;
    int mask_stride;

    /* 归一化异常图，float32，尺寸 map_h * map_w，值域已裁剪到 [0, 1]。 */
    const float* normalized_map;
    int map_w;
    int map_h;

    float raw_score;
    float normalized_score;
    int is_anomaly;
} oi_anomaly_result;

/* 句柄诊断信息。provider 用于让页面显示这次到底跑在 CUDA 还是 CPU 上。 */
typedef struct {
    char provider[32];
    char ort_version[32];
    int  task;
    int  input_w;
    int  input_h;
    int  num_classes;
    int  num_keypoints;
} oi_info;

/*
 * 分类结果（v1.1.0 新增）。
 *
 * scores 是模型输出的**原始分数**，不是概率：是否做 softmax、阈值多少、
 * 怎么取 top-k 全部交给调用方。这样同一份"分数 → 类别"的判定逻辑只存在于
 * 一处（Python 的 classifyinfer.interpret），DLL 路径与 onnxruntime 降级路径
 * 不会因为两套归一化实现而在边界样本上给出不同结论。
 *
 * 指针由句柄持有，只在下一次 oi_infer_classify 或 oi_destroy 之前有效。
 */
typedef struct {
    int   num_classes;   /* 类别数，等于 scores 的长度 */
    int   top1;          /* 原始分数最大的下标 */
    float top1_score;    /* 对应原始分数 */
    const float* scores; /* num_classes 个 float32 原始分数 */
    int   input_w;       /* 实际喂给网络的分辨率 */
    int   input_h;
    /*
     * 模型 metadata 里的 `names` 原文（Ultralytics 写的是 Python dict 字面量，
     * 例如 "{0: '良品', 1: '不良'}"）；老模型可能为空。
     *
     * 为什么把它传出来而不是只在 Python 侧用 onnxruntime 读：主路径不应该为了
     * 读一行 metadata 而依赖 onnxruntime。调用方拿到后按自己的方言解析，
     * 判决"以模型自带顺序为准"的逻辑仍然只有一份。
     */
    const char* names_utf8;
} oi_classify_result;

/*
 * 创建 YOLO ONNX 句柄。
 *
 * 参数：
 * - onnx_path_utf8：.onnx 路径，UTF-8。
 * - task：OI_DETECT / OI_SEGMENT / OI_POSE / OI_OBB。
 * - conf_thr / nms_thr：置信度与 NMS 阈值，进入句柄，后续推理沿用。
 *
 * 返回：
 * - 成功返回非空句柄；失败返回 nullptr，用 oi_last_error() 取原因。
 *
 * 说明：
 * - 任务与模型通道数不匹配时在创建阶段就失败，不会等到推理时才越界。
 * - 输入尺寸必须是静态导出（Ultralytics 默认就是固定 640/960 等）；
 *   动态输入会明确报错，而不是猜一个尺寸。
 */
OI_API oi_handle oi_create_yolo(const char* onnx_path_utf8, int task,
                                float conf_thr, float nms_thr);

/*
 * 创建 Anomalib ONNX 句柄。
 *
 * 参数：
 * - onnx_path_utf8：Anomalib 导出的 .onnx，UTF-8。
 * - metadata_path_utf8：TRTForge metadata.json，UTF-8。
 * - model_type_utf8：提示用模型类型（"PatchCore"/"PaDiM"/"EfficientAd"/"Auto"）。
 *   目前只用于诊断与错误信息，输出解析按张量形状进行。
 * - image_threshold_override / pixel_threshold_override：
 *   >= 0 覆盖 metadata，< 0 沿用 metadata，与 ai_create 语义一致。
 */
OI_API oi_handle oi_create_anomalib(const char* onnx_path_utf8,
                                    const char* metadata_path_utf8,
                                    const char* model_type_utf8,
                                    float image_threshold_override,
                                    float pixel_threshold_override);

/*
 * YOLO 推理。输入 BGR888（与 yi_infer_bgr 相同通道顺序）。
 *
 * 返回：
 * - OI_OK 时 out_boxes 指向句柄内部数组，out_count 为框数量。
 * - 其余返回值表示失败，用 oi_last_error() 取原因。
 * - out_stat 可为 nullptr。
 */
OI_API int oi_infer_yolo(oi_handle h,
                         const unsigned char* bgr, int width, int height, int stride,
                         oi_box** out_boxes, int* out_count,
                         oi_frame_stat* out_stat);

/*
 * Anomalib 推理。输入 RGB888（与 ai_infer_rgb 相同通道顺序）。
 *
 * 返回：
 * - OI_OK 时 out_result 指向句柄内部结构，其中所有指针同样由句柄持有。
 * - out_stat 可为 nullptr。
 */
OI_API int oi_infer_anomalib(oi_handle h,
                             const unsigned char* rgb, int width, int height, int stride,
                             const oi_anomaly_result** out_result,
                             oi_frame_stat* out_stat);

/*
 * 创建分类 ONNX 句柄（v1.1.0 新增）。
 *
 * 参数：
 * - onnx_path_utf8：Ultralytics 导出的分类 ONNX（`yolov8n-cls.onnx` 之类），UTF-8。
 *
 * 返回：
 * - 成功返回非空句柄；失败返回 nullptr，用 oi_last_error() 取原因。
 *
 * 说明：
 * - 输入必须是 4 维 NCHW、通道 3。batch 维是动态时自动固定成 1；
 *   高宽是动态时按 224×224 处理（与 Python 侧 `_input_hw` 的默认值一致）。
 * - 不接收置信度阈值：阈值只影响"是否标记为不确定"，属于调用方的展示逻辑。
 */
OI_API oi_handle oi_create_classify(const char* onnx_path_utf8);

/*
 * 分类推理。输入 BGR888（与 oi_infer_yolo 相同通道顺序）。
 *
 * 预处理与 Ultralytics 的 classify_transforms 对齐：
 *   短边等比缩放到目标边长 → 中心裁剪 → BGR2RGB → ÷255 → NCHW。
 * 非正方形输入时直接 resize 到目标尺寸，这一点与 Python 侧完全一致。
 *
 * 返回：
 * - OI_OK 时 out_result 指向句柄内部结构，scores 指针同样由句柄持有。
 * - 其余返回值表示失败，用 oi_last_error() 取原因。
 * - out_stat 可为 nullptr。
 */
OI_API int oi_infer_classify(oi_handle h,
                             const unsigned char* bgr, int width, int height, int stride,
                             const oi_classify_result** out_result,
                             oi_frame_stat* out_stat);

/* 读取句柄信息。out_info 必须非空。 */
OI_API int oi_get_info(oi_handle h, oi_info* out_info);

/* 销毁句柄。传 nullptr 是安全的空操作。 */
OI_API void oi_destroy(oi_handle h);

/* 返回最近一次错误信息。指针由 DLL 持有，调用方不要释放。 */
OI_API const char* oi_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
