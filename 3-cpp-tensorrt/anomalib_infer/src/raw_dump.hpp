#pragma once

/*
 * 原始张量诊断导出（三个推理 DLL 共用一份实现）。
 *
 * 为什么放在共享头里：
 *   一致性校验需要把「预处理后的输入」与「后处理前的输出」在两条后端之间对齐比对。
 *   如果 yolo_infer / anomalib_infer / onnx_infer 各写一份落盘逻辑，文件名、清单字段
 *   迟早漂移，比对工具就得为每个后端写特例。这里统一定义命名与清单格式：
 *
 *     <backend>_run0001_input0.f32
 *     <backend>_run0001_output0.f32
 *     manifest.jsonl          # 每次张量一行，含形状、元素数、dtype、provider
 *
 * 约定：
 *   - 环境变量 TRTFORGE_RAW_DUMP 指向目录才生效，未设置时只有一次函数内静态初始化，
 *     推理热路径上只剩一个 bool 判断，不产生任何文件系统调用。
 *   - 数据统一按 float32 小端落盘；后端内部若是 fp16，由调用方先转换。
 *   - 目录不存在时只创建一级，多级目录由调用方负责（脚本里已经 mkdir -p）。
 */

#include <windows.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace oi {
namespace dump {

/* 导出目录；未设置环境变量时返回空串。只初始化一次，之后是普通读。 */
inline const std::wstring& directory() {
    static const std::wstring dir = []() -> std::wstring {
        wchar_t buffer[MAX_PATH * 2] = {0};
        const DWORD length = ::GetEnvironmentVariableW(
            L"TRTFORGE_RAW_DUMP", buffer, static_cast<DWORD>(std::size(buffer)));
        if (length == 0 || length >= std::size(buffer)) {
            return std::wstring();
        }
        std::wstring value(buffer, length);
        ::CreateDirectoryW(value.c_str(), nullptr);
        return value;
    }();
    return dir;
}

inline bool enabled() { return !directory().empty(); }

/* 一次推理占一个序号，多个张量共用，便于与另一侧后端按序号配对。 */
inline unsigned begin_run() {
    static std::atomic<unsigned> counter{0};
    return ++counter;
}

/*
 * 把一块 float32 数据落盘并追加清单行。
 *
 * data 必须指向 elements 个 float；调用方负责在 GPU 场景下先拷回主机内存。
 * 写文件失败不抛异常、不返回错误：诊断导出不能影响正常推理结果，
 * 缺文件时由比对工具报「某侧缺张量」，而不是让一次推理失败。
 */
inline void write(unsigned run, const char* backend, const char* phase, std::size_t index,
                  const char* name, const std::vector<int64_t>& shape, const void* data,
                  std::size_t elements, const char* dtype, const char* provider) {
    const std::wstring& dir = directory();
    if (dir.empty()) {
        return;
    }

    std::ostringstream stem;
    stem << backend << "_run" << std::setw(4) << std::setfill('0') << run
         << "_" << phase << index;
    const std::string narrow_stem = stem.str();
    const std::wstring wide_stem(narrow_stem.begin(), narrow_stem.end());
    const std::wstring path = dir + L"\\" + wide_stem + L".f32";

    const std::size_t bytes = elements * sizeof(float);
    HANDLE file = ::CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                                CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file != INVALID_HANDLE_VALUE) {
        if (bytes > 0 && data) {
            DWORD written = 0;
            ::WriteFile(file, data, static_cast<DWORD>(bytes), &written, nullptr);
        }
        ::CloseHandle(file);
    }

    std::ostringstream line;
    line << "{\"backend\":\"" << backend << "\",\"run\":" << run
         << ",\"phase\":\"" << phase << "\",\"index\":" << index
         << ",\"name\":\"" << (name ? name : "") << "\""
         << ",\"shape\":[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        line << (i ? "," : "") << shape[i];
    }
    line << "],\"elements\":" << elements
         << ",\"dtype\":\"" << dtype << "\",\"provider\":\"" << provider << "\"}\n";
    const std::string text = line.str();

    static std::mutex manifest_mutex;
    std::lock_guard<std::mutex> guard(manifest_mutex);
    const std::wstring manifest = dir + L"\\manifest.jsonl";
    HANDLE out = ::CreateFileW(manifest.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, nullptr,
                               OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (out != INVALID_HANDLE_VALUE) {
        DWORD written = 0;
        ::WriteFile(out, text.data(), static_cast<DWORD>(text.size()), &written, nullptr);
        ::CloseHandle(out);
    }
}

}  // namespace dump
}  // namespace oi
