#ifndef OCR_INFER_C_H
#define OCR_INFER_C_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OCR_INFER_EXPORTS
#define OI_API __declspec(dllexport)
#else
#define OI_API __declspec(dllimport)
#endif

typedef void* oi_handle;

typedef enum {
    OI_OK = 0,
    OI_ERR_INVALID_ARGUMENT = 1,
    OI_ERR_CONFIG = 2,
    OI_ERR_MODEL = 3,
    OI_ERR_INFERENCE = 4
} oi_status;

typedef struct {
    const char* det_model_dir_utf8;
    const char* rec_model_dir_utf8;
    const char* textline_orientation_model_dir_utf8;
    const char* doc_orientation_model_dir_utf8;
    const char* doc_unwarping_model_dir_utf8;
    const char* device_utf8;       /* "cpu" or "gpu:0" */
    const char* precision_utf8;    /* "fp32" by default */
    int enable_mkldnn;
    int use_textline_orientation;
    int use_doc_orientation_classify;
    int use_doc_unwarping;
    int cpu_threads;
    int rec_batch_size;
    float det_thresh;
    float det_box_thresh;
    float det_unclip_ratio;
    float rec_score_thresh;
} oi_config;

typedef struct {
    int index;
    int point_count;
    const float* xs;
    const float* ys;
    float rec_score;
    int textline_angle;
    const char* text_utf8;
} oi_text_line;

typedef struct {
    int src_w;
    int src_h;
    int line_count;
    const oi_text_line* lines;

    const unsigned char* overlay_rgb;
    int overlay_w;
    int overlay_h;
    int overlay_stride;

    const char* raw_json_utf8;
} oi_result;

typedef struct {
    int src_w;
    int src_h;
    int count;
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double total_ms;
    double io_ms;
    double pipeline_ms;
    double result_ms;
    char device_utf8[16];
    char precision_utf8[16];
    char run_mode_utf8[32];
    int mkldnn_enabled;
    int cpu_threads;
    int rec_batch_size;
} oi_frame_stat;

/*
 * Creates a PaddleOCR inference handle. Paths are UTF-8.
 * Required: det_model_dir_utf8 and rec_model_dir_utf8.
 * Optional modules are enabled only when their use_* flag is non-zero and
 * the matching model directory is provided.
 */
OI_API oi_handle oi_create(const oi_config* config);

/*
 * Runs OCR on an image file path. The returned result memory is owned by the
 * handle and remains valid until the next oi_infer_* call or oi_destroy.
 */
OI_API int oi_infer_file(oi_handle h,
                         const char* image_path_utf8,
                         const oi_result** out_result,
                         oi_frame_stat* out_stat);

/*
 * Runs OCR on a BGR888 image buffer. stride is bytes per row and may be larger
 * than width * 3. The returned result memory follows the same lifecycle as
 * oi_infer_file.
 */
OI_API int oi_infer_bgr(oi_handle h,
                        const unsigned char* bgr,
                        int width,
                        int height,
                        int stride,
                        const oi_result** out_result,
                        oi_frame_stat* out_stat);

OI_API void oi_destroy(oi_handle h);

/* Returns the latest thread-local error message. Do not free the pointer. */
OI_API const char* oi_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
