/* ONNX Runtime 显式加载与推理会话实现，详见 ort_session.hpp 的说明。 */

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iterator>
#include <mutex>
#include <sstream>

#include "ort_session.hpp"
#include "raw_dump.hpp"

namespace oi {
namespace {

/* 取本 DLL 所在目录。用函数地址反查模块，避免依赖任何固定安装路径。 */
std::wstring this_module_dir() {
    HMODULE module = nullptr;
    if (!::GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&this_module_dir), &module)) {
        return std::wstring();
    }
    wchar_t buffer[MAX_PATH * 2] = {0};
    const DWORD length = ::GetModuleFileNameW(module, buffer, static_cast<DWORD>(std::size(buffer)));
    if (length == 0) {
        return std::wstring();
    }
    std::wstring full(buffer, length);
    const size_t slash = full.find_last_of(L"\\/");
    return slash == std::wstring::npos ? std::wstring() : full.substr(0, slash);
}

std::string status_message(const OrtApi* api, OrtStatus* status) {
    if (!status) {
        return std::string();
    }
    const char* raw = api ? api->GetErrorMessage(status) : nullptr;
    std::string message = raw ? raw : "ORT 调用失败";
    if (api) {
        api->ReleaseStatus(status);
    }
    return message;
}

/* 进程级共享的环境对象。ORT 允许创建多个 Env，但共享一个更省内存也更干净。 */
OrtEnv* shared_env(const OrtApi* api, std::string& err) {
    static std::mutex mutex;
    static OrtEnv* env = nullptr;
    std::lock_guard<std::mutex> guard(mutex);
    if (env) {
        return env;
    }
    OrtStatus* status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnx_infer", &env);
    if (status) {
        err = "创建 ONNX Runtime 环境失败：" + status_message(api, status);
        env = nullptr;
        return nullptr;
    }
    return env;
}

int64_t shape_product(const std::vector<int64_t>& shape) {
    int64_t total = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            continue;
        }
        total *= dim;
    }
    return total;
}

}  // namespace

/* ------------------------------------------------------------------------- */
/* TensorInfo                                                                */
/* ------------------------------------------------------------------------- */

std::vector<int64_t> TensorInfo::concrete_shape() const {
    std::vector<int64_t> resolved = shape;
    for (int64_t& dim : resolved) {
        if (dim <= 0) {
            dim = 1;  // 动态维度统一按 1 喂，Anomalib 导出只把 batch 设成动态
        }
    }
    return resolved;
}

int64_t TensorInfo::element_count() const { return shape_product(shape); }

/* ------------------------------------------------------------------------- */
/* Runtime                                                                   */
/* ------------------------------------------------------------------------- */

Runtime& Runtime::get() {
    static Runtime instance;
    return instance;
}

Runtime::Runtime() { load(); }

void* Runtime::load_module(const wchar_t* name) const {
    if (!name) {
        return nullptr;
    }

    // 1) 显式指定目录（部署时可以指向独立的 ORT 目录，避免与 Python 版抢同名 DLL）
    wchar_t env_dir[MAX_PATH * 2] = {0};
    const DWORD env_len = ::GetEnvironmentVariableW(L"TRTFORGE_ORT_DIR", env_dir,
                                                    static_cast<DWORD>(std::size(env_dir)));
    if (env_len > 0 && env_len < std::size(env_dir)) {
        const std::wstring full = std::wstring(env_dir) + L"\\" + name;
        HMODULE module = ::LoadLibraryExW(full.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (module) {
            return module;
        }
    }

    // 2) 本 DLL 同目录（正常部署形态：onnx_infer.dll 与 onnxruntime.dll 并排）
    const std::wstring dir = this_module_dir();
    if (!dir.empty()) {
        const std::wstring full = dir + L"\\" + name;
        HMODULE module = ::LoadLibraryExW(full.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (module) {
            return module;
        }
    }

    // 3) 默认搜索路径（开发机上可能已经由 PATH 或已加载模块提供）
    return ::LoadLibraryW(name);
}

void Runtime::load() {
    void* module = load_module(L"onnxruntime.dll");
    if (!module) {
        std::ostringstream oss;
        oss << "无法加载 onnxruntime.dll（错误码 " << ::GetLastError() << "）。"
            << "请把 onnxruntime.dll 与 onnx_infer.dll 放在同一目录，"
            << "或设置环境变量 TRTFORGE_ORT_DIR 指向 ORT 目录。";
        error_ = oss.str();
        return;
    }

    // 不直接引用 OrtGetApiBase 符号：那会在导入表里写死 onnxruntime.dll，
    // 从而复用进程里已加载的另一份（可能是 Python 的 1.20.x）。
    using OrtGetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)(void);
    OrtGetApiBaseFn get_api_base = reinterpret_cast<OrtGetApiBaseFn>(
        ::GetProcAddress(reinterpret_cast<HMODULE>(module), "OrtGetApiBase"));
    if (!get_api_base) {
        error_ = "onnxruntime.dll 中找不到 OrtGetApiBase，文件可能不是官方 ONNX Runtime。";
        return;
    }

    const OrtApiBase* base = get_api_base();
    if (!base) {
        error_ = "OrtGetApiBase 返回空指针，onnxruntime.dll 可能已损坏。";
        return;
    }

    api_ = base->GetApi(ORT_API_VERSION);
    if (!api_) {
        std::ostringstream oss;
        oss << "onnxruntime.dll 的 API 版本低于本 DLL 编译期要求的 v" << ORT_API_VERSION
            << "。请升级 onnxruntime.dll，或使用配套版本。";
        error_ = oss.str();
        return;
    }

    const char* version = base->GetVersionString();
    version_ = version ? version : "unknown";
    error_.clear();
}

bool Runtime::probe_module(const wchar_t* name) const { return load_module(name) != nullptr; }

/* ------------------------------------------------------------------------- */
/* Session                                                                   */
/* ------------------------------------------------------------------------- */

Session::~Session() { close(); }

void Session::close() {
    if (!api_) {
        return;
    }
    if (session_) {
        api_->ReleaseSession(session_);
        session_ = nullptr;
    }
    if (options_) {
        api_->ReleaseSessionOptions(options_);
        options_ = nullptr;
    }
    if (memory_info_) {
        api_->ReleaseMemoryInfo(memory_info_);
        memory_info_ = nullptr;
    }
    // env_ 是进程级共享对象，这里不释放。
    env_ = nullptr;
}

bool Session::open(const std::wstring& model_path, std::string& err) {
    Runtime& runtime = Runtime::get();
    if (!runtime.available()) {
        err = "ONNX Runtime 不可用：" + runtime.error();
        return false;
    }
    api_ = runtime.api();

    env_ = shared_env(api_, err);
    if (!env_) {
        return false;
    }

    if (!memory_info_) {
        OrtStatus* status = api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_);
        if (status) {
            err = "创建 CPU 内存描述失败：" + status_message(api_, status);
            return false;
        }
    }

    // CUDA provider DLL 先探测再启用：编译期支持不等于运行时可用，
    // 缺 cuDNN 的机器上直接 append 会让 CreateSession 抛错。
    const bool cuda_ready = runtime.probe_module(L"onnxruntime_providers_cuda.dll");
    if (cuda_ready) {
        std::string cuda_err;
        if (create_session(model_path, /*with_cuda=*/true, cuda_err) && describe_io(cuda_err)) {
            provider_ = "CUDA";
            return true;
        }
        // CUDA 起不来时清干净再走 CPU，不把 CUDA 的错误抛给页面。
        close();
        api_ = runtime.api();
        env_ = shared_env(api_, err);
        if (!env_) {
            return false;
        }
        OrtStatus* status = api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_);
        if (status) {
            err = "创建 CPU 内存描述失败：" + status_message(api_, status);
            return false;
        }
        if (!create_session(model_path, /*with_cuda=*/false, err)) {
            return false;
        }
        provider_ = "CPU";
        return describe_io(err);
    }

    if (!create_session(model_path, /*with_cuda=*/false, err)) {
        return false;
    }
    provider_ = "CPU";
    return describe_io(err);
}

bool Session::create_session(const std::wstring& model_path, bool with_cuda, std::string& err) {
    OrtStatus* status = api_->CreateSessionOptions(&options_);
    if (status) {
        err = "创建会话选项失败：" + status_message(api_, status);
        return false;
    }

    // 线程数允许外部覆盖：批量推理时限制线程能避免和训练/其它服务抢核。
    wchar_t threads_text[32] = {0};
    const DWORD threads_len = ::GetEnvironmentVariableW(L"TRTFORGE_ORT_THREADS", threads_text,
                                                        static_cast<DWORD>(std::size(threads_text)));
    if (threads_len > 0 && threads_len < std::size(threads_text)) {
        const int threads = _wtoi(threads_text);
        if (threads > 0) {
            status = api_->SetIntraOpNumThreads(options_, threads);
            if (status) {
                api_->ReleaseStatus(status);
            }
        }
    }

    status = api_->SetSessionGraphOptimizationLevel(options_, ORT_ENABLE_ALL);
    if (status) {
        api_->ReleaseStatus(status);  // 优化级别设置失败不致命，继续用默认值
    }

    if (with_cuda) {
        status = api_->SessionOptionsAppendExecutionProvider_CUDA(options_, 0);
        if (status) {
            err = "启用 CUDA 执行提供程序失败：" + status_message(api_, status);
            return false;
        }
    }

    status = api_->CreateSession(env_, model_path.c_str(), options_, &session_);
    if (status) {
        err = "加载 ONNX 模型失败：" + status_message(api_, status);
        return false;
    }
    return true;
}

bool Session::describe_io(std::string& err) {
    inputs_.clear();
    outputs_.clear();

    OrtAllocator* allocator = nullptr;
    OrtStatus* status = api_->GetAllocatorWithDefaultOptions(&allocator);
    if (status) {
        err = "获取默认分配器失败：" + status_message(api_, status);
        return false;
    }

    auto read_one = [&](bool is_input, size_t index, TensorInfo& info) -> bool {
        char* name = nullptr;
        OrtStatus* local = is_input
            ? api_->SessionGetInputName(session_, index, allocator, &name)
            : api_->SessionGetOutputName(session_, index, allocator, &name);
        if (local) {
            err = "读取张量名称失败：" + status_message(api_, local);
            return false;
        }
        info.name = name ? name : "";
        if (name) {
            allocator->Free(allocator, name);
        }

        OrtTypeInfo* type_info = nullptr;
        local = is_input ? api_->SessionGetInputTypeInfo(session_, index, &type_info)
                         : api_->SessionGetOutputTypeInfo(session_, index, &type_info);
        if (local) {
            err = "读取张量类型失败：" + status_message(api_, local);
            return false;
        }

        const OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        local = api_->CastTypeInfoToTensorInfo(type_info, &shape_info);
        if (local || !shape_info) {
            if (local) {
                api_->ReleaseStatus(local);
            }
            api_->ReleaseTypeInfo(type_info);
            err = "只支持张量类型的输入输出，" + std::string(is_input ? "输入" : "输出") +
                  " " + info.name + " 不是张量。";
            return false;
        }

        size_t dim_count = 0;
        local = api_->GetDimensionsCount(shape_info, &dim_count);
        if (local) {
            api_->ReleaseTypeInfo(type_info);
            err = "读取张量维度失败：" + status_message(api_, local);
            return false;
        }

        info.shape.assign(dim_count, 0);
        if (dim_count > 0) {
            local = api_->GetDimensions(shape_info, info.shape.data(), dim_count);
            if (local) {
                api_->ReleaseTypeInfo(type_info);
                err = "读取张量形状失败：" + status_message(api_, local);
                return false;
            }
        }

        ONNXType onnx_type = ONNX_TYPE_UNKNOWN;
        api_->GetOnnxTypeFromTypeInfo(type_info, &onnx_type);
        if (onnx_type != ONNX_TYPE_TENSOR) {
            api_->ReleaseTypeInfo(type_info);
            err = std::string("只支持张量类型，当前") + (is_input ? "输入" : "输出") + "不是张量。";
            return false;
        }

        ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        local = api_->GetTensorElementType(shape_info, &element_type);
        if (local) {
            api_->ReleaseTypeInfo(type_info);
            err = "读取张量数据类型失败：" + status_message(api_, local);
            return false;
        }
        info.element_type = static_cast<int>(element_type);
        api_->ReleaseTypeInfo(type_info);
        return true;
    };

    size_t input_count = 0;
    status = api_->SessionGetInputCount(session_, &input_count);
    if (status) {
        err = "读取输入数量失败：" + status_message(api_, status);
        return false;
    }
    if (input_count == 0) {
        err = "模型没有输入张量。";
        return false;
    }

    for (size_t i = 0; i < input_count; ++i) {
        TensorInfo info;
        if (!read_one(true, i, info)) {
            return false;
        }
        inputs_.push_back(std::move(info));
    }

    size_t output_count = 0;
    status = api_->SessionGetOutputCount(session_, &output_count);
    if (status) {
        err = "读取输出数量失败：" + status_message(api_, status);
        return false;
    }
    if (output_count == 0) {
        err = "模型没有输出张量。";
        return false;
    }

    for (size_t i = 0; i < output_count; ++i) {
        TensorInfo info;
        if (!read_one(false, i, info)) {
            return false;
        }
        outputs_.push_back(std::move(info));
    }
    return true;
}

bool Session::set_input_dimensions(size_t index, const std::vector<int64_t>& dims,
                                   std::string& err) {
    if (index >= inputs_.size()) {
        err = "输入索引越界。";
        return false;
    }
    if (dims.size() != inputs_[index].shape.size()) {
        err = "输入维度数量与模型不一致。";
        return false;
    }
    // 静态维不允许被改写，只把动态维（<=0）填成具体值，避免喂错形状导致越界读。
    for (size_t i = 0; i < dims.size(); ++i) {
        const int64_t declared = inputs_[index].shape[i];
        if (declared > 0 && declared != dims[i]) {
            std::ostringstream oss;
            oss << "输入维度冲突：模型声明第 " << i << " 维为 " << declared
                << "，试图设置为 " << dims[i] << "。";
            err = oss.str();
            return false;
        }
    }
    inputs_[index].shape = dims;
    return true;
}

bool Session::run(const std::vector<std::vector<float>>& inputs,
                  std::vector<std::vector<float>>& outputs,
                  std::string& err) {
    if (!session_ || !api_) {
        err = "推理会话尚未创建。";
        return false;
    }
    if (inputs.size() != inputs_.size()) {
        std::ostringstream oss;
        oss << "输入数量不匹配：模型需要 " << inputs_.size() << " 个，实际 " << inputs.size() << " 个。";
        err = oss.str();
        return false;
    }

    std::vector<const char*> input_names;
    std::vector<OrtValue*> input_values(inputs.size(), nullptr);
    std::vector<const OrtValue*> input_const;

    // 诊断导出：只在环境变量开启时分配序号，否则 dump_run 恒为 0，后续判断一次即跳过
    const unsigned dump_run = dump::enabled() ? dump::begin_run() : 0;

    auto cleanup_inputs = [&]() {
        for (OrtValue* value : input_values) {
            if (value) {
                api_->ReleaseValue(value);
            }
        }
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        const std::vector<int64_t> shape = inputs_[i].concrete_shape();
        const int64_t expected = shape_product(shape);
        if (expected != static_cast<int64_t>(inputs[i].size())) {
            std::ostringstream oss;
            oss << "输入 " << inputs_[i].name << " 尺寸不匹配：模型期望 " << expected
                << " 个元素，实际 " << inputs[i].size() << " 个。";
            err = oss.str();
            cleanup_inputs();
            return false;
        }

        OrtStatus* status = api_->CreateTensorWithDataAsOrtValue(
            memory_info_, const_cast<float*>(inputs[i].data()), inputs[i].size() * sizeof(float),
            shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_values[i]);
        if (status) {
            err = "创建输入张量失败：" + status_message(api_, status);
            cleanup_inputs();
            return false;
        }
        input_names.push_back(inputs_[i].name.c_str());
        input_const.push_back(input_values[i]);

        if (dump_run) {
            dump::write(dump_run, "onnx_infer", "input", i, inputs_[i].name.c_str(), shape,
                        inputs[i].data(), inputs[i].size(), "float32", provider_.c_str());
        }
    }

    std::vector<const char*> output_names;
    for (const TensorInfo& info : outputs_) {
        output_names.push_back(info.name.c_str());
    }
    std::vector<OrtValue*> output_values(outputs_.size(), nullptr);

    OrtStatus* status = api_->Run(session_, nullptr, input_names.data(), input_const.data(),
                                  input_const.size(), output_names.data(), output_names.size(),
                                  output_values.data());
    if (status) {
        err = "推理执行失败：" + status_message(api_, status);
        cleanup_inputs();
        return false;
    }

    outputs.clear();
    outputs.reserve(output_values.size());
    bool ok = true;

    for (size_t i = 0; i < output_values.size() && ok; ++i) {
        OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        OrtStatus* local = api_->GetTensorTypeAndShape(output_values[i], &shape_info);
        if (local) {
            err = "读取输出形状失败：" + status_message(api_, local);
            ok = false;
            break;
        }

        size_t dim_count = 0;
        api_->GetDimensionsCount(shape_info, &dim_count);
        std::vector<int64_t> dims(dim_count, 0);
        if (dim_count > 0) {
            api_->GetDimensions(shape_info, dims.data(), dim_count);
        }

        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        api_->GetTensorElementType(shape_info, &type);
        api_->ReleaseTensorTypeAndShapeInfo(shape_info);

        const int64_t count = shape_product(dims);
        void* data = nullptr;
        local = api_->GetTensorMutableData(output_values[i], &data);
        if (local || !data) {
            if (local) {
                api_->ReleaseStatus(local);
            }
            err = "读取输出数据失败：" + outputs_[i].name;
            ok = false;
            break;
        }

        std::vector<float> converted(static_cast<size_t>(count > 0 ? count : 0), 0.0f);
        switch (type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                const float* src = static_cast<const float*>(data);
                std::copy(src, src + converted.size(), converted.begin());
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
                const uint16_t* src = static_cast<const uint16_t*>(data);
                for (size_t k = 0; k < converted.size(); ++k) {
                    converted[k] = half_to_float(src[k]);
                }
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
                const double* src = static_cast<const double*>(data);
                for (size_t k = 0; k < converted.size(); ++k) {
                    converted[k] = static_cast<float>(src[k]);
                }
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                const int64_t* src = static_cast<const int64_t*>(data);
                for (size_t k = 0; k < converted.size(); ++k) {
                    converted[k] = static_cast<float>(src[k]);
                }
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                const int32_t* src = static_cast<const int32_t*>(data);
                for (size_t k = 0; k < converted.size(); ++k) {
                    converted[k] = static_cast<float>(src[k]);
                }
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
                const uint8_t* src = static_cast<const uint8_t*>(data);
                for (size_t k = 0; k < converted.size(); ++k) {
                    converted[k] = static_cast<float>(src[k]);
                }
                break;
            }
            default: {
                err = "暂不支持的输出数据类型（" + std::to_string(static_cast<int>(type)) + "）：" +
                      outputs_[i].name;
                ok = false;
                break;
            }
        }

        outputs.push_back(std::move(converted));

        if (dump_run) {
            dump::write(dump_run, "onnx_infer", "output", i, outputs_[i].name.c_str(), dims,
                        outputs.back().data(), outputs.back().size(), "float32",
                        provider_.c_str());
        }
    }

    for (OrtValue* value : output_values) {
        if (value) {
            api_->ReleaseValue(value);
        }
    }
    cleanup_inputs();
    return ok;
}

/* ------------------------------------------------------------------------- */
/* 公共工具                                                                  */
/* ------------------------------------------------------------------------- */

namespace {

/* 每个线程一份错误串，oi_last_error 直接返回它的 c_str()。 */
std::string& error_buffer() {
    static thread_local std::string message;
    return message;
}

}  // namespace

void set_last_error(const std::string& message) { error_buffer() = message; }

const char* last_error() { return error_buffer().c_str(); }

std::wstring utf8_to_wide(const char* text) {
    if (!text || !*text) {
        return std::wstring();
    }
    const int length = ::MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    if (length <= 1) {
        return std::wstring();
    }
    std::wstring wide(static_cast<size_t>(length - 1), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, text, -1, wide.data(), length);
    return wide;
}

float half_to_float(uint16_t value) {
    const uint32_t sign = (value >> 15) & 0x1u;
    const uint32_t exponent = (value >> 10) & 0x1Fu;
    const uint32_t mantissa = value & 0x3FFu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign << 31;
        } else {
            uint32_t e = 1;
            uint32_t m = mantissa;
            while (!(m & 0x400u)) {
                m <<= 1;
                --e;
            }
            m &= 0x3FFu;
            bits = (sign << 31) | ((e + 127u - 15u) << 23) | (m << 13);
        }
    } else if (exponent == 31) {
        bits = (sign << 31) | 0x7F800000u | (mantissa << 13);
    } else {
        bits = (sign << 31) | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

}  // namespace oi
