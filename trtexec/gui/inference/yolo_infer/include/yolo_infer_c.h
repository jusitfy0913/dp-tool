#ifndef YOLO_INFER_C_H
#define YOLO_INFER_C_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef YOLO_INFER_EXPORTS
#define YI_API __declspec(dllexport)
#else
#define YI_API __declspec(dllimport)
#endif

typedef void* yi_handle;

typedef enum {
    YI_DETECT  = 0,
    YI_SEGMENT = 1,
    YI_POSE    = 2,
    YI_OBB     = 3
} yi_task;

typedef struct {
    float left, top, right, bottom;
    float confidence;
    int   class_label;
    int   has_mask;
    int   mask_x, mask_y;
    int   mask_w, mask_h;
    const unsigned char* mask_data;
    /* Pose keypoints (YI_POSE only). num_kpts = 0 when not pose.
     * kpts layout: num_kpts triples of (x, y, visibility), float32, length = num_kpts * 3.
     * Memory owned by the engine handle until the next yi_infer_bgr call. */
    int                  num_kpts;
    const float*         kpts;
    /* Oriented bounding box (YI_OBB only). is_obb=0 when not OBB.
     * (cx, cy, ow, oh) in original image space, angle in radians. */
    int                  is_obb;
    float                cx, cy;
    float                ow, oh;
    float                angle;
} yi_box;

typedef struct {
    int    src_w, src_h;
    int    count;
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double total_ms;
} yi_frame_stat;

YI_API yi_handle    yi_create(const char* engine_path, yi_task task, float conf_thr, float nms_thr);
YI_API int          yi_infer_bgr(yi_handle h, const unsigned char* bgr, int w, int height, int stride, yi_box** out_boxes, int* out_count, yi_frame_stat* out_stat);
YI_API void         yi_destroy(yi_handle h);
YI_API const char*  yi_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
