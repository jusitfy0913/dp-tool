#pragma once

/*
 * ONNX Runtime 的显式加载封装。
 *
 * 为什么不用导入库直接链接 onnxruntime.lib：
 *   Windows 的加载器按模块名去重。后端进程里已经加载了 Python 的 onnxruntime.dll
 *   （_probe_onnx 与降级路径都在用），如果本 DLL 的导入表里写死 onnxruntime.dll，
 *   链接时会直接复用进程里那份，版本不一致就会出现缺符号或行为漂移。
 *   这里改为运行时 LoadLibraryExW + OrtGetApiBase，拿到的函数表只属于本 DLL。
 *
 * 代价是所有调用都要经函数表指针，因此统一收敛在 Session 里。
 */

#include <cstdint>
#include <string>
#include <vector>

#include "onnxruntime_c_api.h"

namespace oi {

/* 一个输入/输出张量的静态描述。shape 里的 -1 表示动态维度。 */
struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    int element_type = 0;

    /* 把动态维度替换成 1，得到实际可喂的形状。 */
    std::vector<int64_t> concrete_shape() const;
    int64_t element_count() const;
};

/*
 * 全局运行时：进程内只加载一次 ORT，句柄与函数表常驻。
 * 加载失败不会抛异常，而是把原因记录在 error() 里，由上层转成可读错误。
 */
class Runtime {
public:
    static Runtime& get();

    bool available() const { return api_ != nullptr; }
    const OrtApi* api() const { return api_; }
    const std::string& error() const { return error_; }
    const std::string& ort_version() const { return version_; }

    /* 探测某个依赖模块能否加载（用于 CUDA provider 的可用性判断）。 */
    bool probe_module(const wchar_t* name) const;

private:
    Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    void load();
    /* 尝试在 DLL 同目录与默认搜索路径下加载指定名字的模块，不释放句柄。 */
    void* load_module(const wchar_t* name) const;

    const OrtApi* api_ = nullptr;
    std::string version_;
    std::string error_;
};

/*
 * 一个推理会话。生命周期内持有 OrtEnv（进程级共享）、SessionOptions 与 Session。
 * 不再需要时按成员析构自动释放；析构顺序按成员声明逆序，因此 session_ 早于 options_。
 */
class Session {
public:
    Session() = default;
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    /*
     * 打开模型。CUDA provider 仅在 DLL 可加载且会话创建成功时启用，
     * 否则自动回退 CPU，并在 provider() 中如实反映。
     */
    bool open(const std::wstring& model_path, std::string& err);

    /* 按输入顺序喂入 float32 数据，输出统一转成 float32。 */
    bool run(const std::vector<std::vector<float>>& inputs,
             std::vector<std::vector<float>>& outputs,
             std::string& err);

    /*
     * 把某个输入的动态维度固定成具体值。
     * Anomalib 导出的 ONNX 常把宽高设成动态，运行前必须按 metadata 的
     * input_size 定下来，否则 CreateTensorWithDataAsOrtValue 会用一个元素
     * 却对不上实际数据的形状。
     */
    bool set_input_dimensions(size_t index, const std::vector<int64_t>& dims, std::string& err);

    const std::vector<TensorInfo>& inputs() const { return inputs_; }
    const std::vector<TensorInfo>& outputs() const { return outputs_; }
    const std::string& provider() const { return provider_; }

    /*
     * 读取模型的 custom metadata（例如 Ultralytics 写入的 `names`）。
     * 取不到时返回空串，不视为错误：老模型或非常规导出本来就没有这项。
     */
    std::string custom_metadata(const char* key) const;

private:
    bool create_session(const std::wstring& model_path, bool with_cuda, std::string& err);
    void close();
    bool describe_io(std::string& err);

    const OrtApi* api_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSessionOptions* options_ = nullptr;
    OrtSession* session_ = nullptr;
    OrtMemoryInfo* memory_info_ = nullptr;

    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;
    std::string provider_ = "CPU";
};

/* 句柄作用域内的错误串，供 C API 的 oi_last_error 读取。 */
void set_last_error(const std::string& message);
const char* last_error();

/* UTF-8 -> 宽字符，失败时返回空串。 */
std::wstring utf8_to_wide(const char* text);

/* 半精度浮点转换，输出张量为 float16 时使用。 */
float half_to_float(uint16_t value);

}  // namespace oi
