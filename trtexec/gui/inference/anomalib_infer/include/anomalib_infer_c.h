#ifndef ANOMALIB_INFER_C_H
#define ANOMALIB_INFER_C_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ANOMALIB_INFER_EXPORTS
#define AI_API __declspec(dllexport)
#else
#define AI_API __declspec(dllimport)
#endif

typedef void* ai_handle;

typedef enum {
    AI_OK = 0,
    AI_ERR_INVALID_ARGUMENT = 1,
    AI_ERR_METADATA = 2,
    AI_ERR_ENGINE = 3,
    AI_ERR_CUDA = 4,
    AI_ERR_INFERENCE = 5
} ai_status;

typedef struct {
    char model_type[64];
    char normalization[32];
    char output_contract[64];
    float image_threshold;
    float pixel_threshold;
    float score_min;
    float score_max;
    float map_min;
    float map_max;
    int input_h;
    int input_w;
} ai_metadata;

typedef struct {
    int src_w;
    int src_h;
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double total_ms;
} ai_frame_stat;

typedef struct {
    /* 原图 RGB888，长度为 source_h * source_stride。 */
    const unsigned char* source_rgb;
    int source_w;
    int source_h;
    int source_stride;

    /* 热力图叠加图 RGB888，长度为 overlay_h * overlay_stride。 */
    const unsigned char* overlay_rgb;
    int overlay_w;
    int overlay_h;
    int overlay_stride;

    /* 红色异常轮廓图 RGB888，长度为 mask_h * mask_stride。 */
    const unsigned char* mask_rgb;
    int mask_w;
    int mask_h;
    int mask_stride;

    /* 归一化 anomaly map，float32，尺寸为 map_h * map_w，值域裁剪到 [0, 1]。 */
    const float* normalized_map;
    int map_w;
    int map_h;

    float raw_score;
    float normalized_score;
    int is_anomaly;
} ai_result;

/*
 * 创建 Anomalib TensorRT 推理句柄。
 *
 * 参数：
 * - engine_path_utf8：TensorRT .engine/.trt 路径，UTF-8 编码。Windows 内部会转宽字符读取。
 * - metadata_path_utf8：TRTForge metadata.json 路径，UTF-8 编码。
 * - model_type_utf8：外部提示的模型类型，例如 "EfficientAd"、"PaDiM"、"PatchCore"、"Auto"。
 * - image_threshold_override：>=0 时覆盖 metadata 中的 image_threshold；<0 使用 metadata。
 * - pixel_threshold_override：>=0 时覆盖 metadata 中的 pixel_threshold；<0 使用 metadata。
 *
 * 返回：
 * - 成功返回非空 ai_handle。
 * - 失败返回 nullptr，可调用 ai_last_error() 获取错误信息。
 *
 * 线程说明：
 * - 一个 ai_handle 不应被多个线程同时调用 ai_infer_rgb。
 * - 多路并发请创建多个 ai_handle。
 */
AI_API ai_handle ai_create(const char* engine_path_utf8,
                           const char* metadata_path_utf8,
                           const char* model_type_utf8,
                           float image_threshold_override,
                           float pixel_threshold_override);

/*
 * 读取句柄内的 metadata。out_metadata 必须非空。
 * 成功返回 AI_OK，失败返回非 0，并可调用 ai_last_error()。
 */
AI_API int ai_get_metadata(ai_handle h, ai_metadata* out_metadata);

/*
 * 输入 RGB888 图像并执行一次推理。
 *
 * 参数：
 * - rgb：输入图像首地址，必须为 RGB888，每像素 3 字节。
 * - width/height：输入图像宽高。
 * - stride：每行字节数，允许大于 width * 3，用于兼容带行对齐的图像缓冲。
 * - out_result：成功后指向 DLL 内部结果结构。
 * - out_stat：可选，返回耗时统计；不需要时可传 nullptr。
 *
 * 内存生命周期：
 * - out_result 以及其中所有指针均由 DLL 句柄持有。
 * - 这些指针只保证在下一次 ai_infer_rgb、ai_destroy 前有效。
 * - 调用方（TRTForge GUI、外部 C++、Python ctypes）如需长期保存，必须立即深拷贝。
 */
AI_API int ai_infer_rgb(ai_handle h,
                        const unsigned char* rgb,
                        int width,
                        int height,
                        int stride,
                        const ai_result** out_result,
                        ai_frame_stat* out_stat);

/* 销毁句柄并释放 TensorRT/CUDA/结果缓存资源。 */
AI_API void ai_destroy(ai_handle h);

/* 返回最近一次错误信息。返回指针由 DLL 持有，调用方不要释放。 */
AI_API const char* ai_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
