// 用途：打印 C 侧结构体的 sizeof 与关键字段 offsetof。
// 目的：与 Python ctypes 的定义逐项比对，确认 ABI 一致。
// 这两个结构体跨语言传值，一旦对齐不一致会静默读错内存，必须实测而不是靠推断。
//
// 编译（在 VS 开发者命令提示符下）：
//   cl /EHsc /std:c++17 /I <repo>\3-cpp-tensorrt\yolo_infer\include ^
//      /I <repo>\3-cpp-tensorrt\anomalib_infer\include abi_probe.cpp

#include <cstdio>
#include <cstddef>

// 头文件里带 __declspec(dllimport)，仅为探测类型时必须绕开
#define YOLO_INFER_EXPORTS
#include "yolo_infer_c.h"
#undef YOLO_INFER_EXPORTS

#include "anomalib_infer_c.h"

int main() {
    std::printf("yi_box size=%zu\n", sizeof(yi_box));
    std::printf("yi_box.left=%zu top=%zu right=%zu bottom=%zu conf=%zu cls=%zu\n",
                offsetof(yi_box, left), offsetof(yi_box, top),
                offsetof(yi_box, right), offsetof(yi_box, bottom),
                offsetof(yi_box, confidence), offsetof(yi_box, class_label));
    std::printf("yi_box.has_mask=%zu mask_data=%zu num_kpts=%zu kpts=%zu is_obb=%zu angle=%zu\n",
                offsetof(yi_box, has_mask), offsetof(yi_box, mask_data),
                offsetof(yi_box, num_kpts), offsetof(yi_box, kpts),
                offsetof(yi_box, is_obb), offsetof(yi_box, angle));

    std::printf("yi_frame_stat size=%zu\n", sizeof(yi_frame_stat));

    std::printf("ai_metadata size=%zu image_threshold=%zu pixel_threshold=%zu input_h=%zu\n",
                sizeof(ai_metadata), offsetof(ai_metadata, image_threshold),
                offsetof(ai_metadata, pixel_threshold), offsetof(ai_metadata, input_h));

    std::printf("ai_result size=%zu\n", sizeof(ai_result));
    std::printf("ai_result.source_rgb=%zu source_w=%zu source_stride=%zu\n",
                offsetof(ai_result, source_rgb), offsetof(ai_result, source_w),
                offsetof(ai_result, source_stride));
    std::printf("ai_result.overlay_rgb=%zu mask_rgb=%zu normalized_map=%zu\n",
                offsetof(ai_result, overlay_rgb), offsetof(ai_result, mask_rgb),
                offsetof(ai_result, normalized_map));
    std::printf("ai_result.raw_score=%zu normalized_score=%zu is_anomaly=%zu\n",
                offsetof(ai_result, raw_score), offsetof(ai_result, normalized_score),
                offsetof(ai_result, is_anomaly));

    std::printf("ai_frame_stat size=%zu\n", sizeof(ai_frame_stat));
    return 0;
}
