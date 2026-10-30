/**
 * @file    infer.cu
 * @brief   trt::Infer 的具体实现：封装 nvinfer1 的 Runtime / Engine / Context，
 *          并提供 Timer、BaseMemory 的实现。
 *
 * 主要内容：
 *  - 错误检查宏 checkRuntime / checkKernel / Assert / Assertf；
 *  - file_name / __log_func / format_shape : 日志与工具；
 *  - Timer / BaseMemory : 显存/计时器实现；
 *  - __native_nvinfer_logger : nvinfer1::ILogger 实现；
 *  - __native_engine_context : Runtime+Engine+Context 的 RAII 三件套；
 *  - InferImpl              : trt::Infer 的具体实现；
 *  - load / loadraw         : 工厂函数。
 */
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <stdarg.h>

#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "infer.hpp"
//#include "Windows.h"

namespace trt {

using namespace std;
using namespace nvinfer1;

#define checkRuntime(call)                                                                 \
  do {                                                                                     \
    auto ___call__ret_code__ = (call);                                                     \
    if (___call__ret_code__ != cudaSuccess) {                                              \
      INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,                         \
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

#define Assert(op)                 \
  do {                             \
    bool cond = !(!(op));          \
    if (!cond) {                   \
      INFO("Assert failed, " #op); \
      abort();                     \
    }                              \
  } while (0)

#define Assertf(op, ...)                             \
  do {                                               \
    bool cond = !(!(op));                            \
    if (!cond) {                                     \
      INFO("Assert failed, " #op " : " __VA_ARGS__); \
      abort();                                       \
    }                                                \
  } while (0)

/**
 * @brief 从完整路径中提取文件名，可选是否带后缀。
 * @param path           原始路径（支持 / 或 \\ 分隔）
 * @param include_suffix true 保留后缀；false 去掉最后一个 '.' 之后的内容
 */
static string file_name(const string &path, bool include_suffix) {
  if (path.empty()) return "";

  int p = path.rfind('/');
  int e = path.rfind('\\');
  p = max(p, e);
  p += 1;

  // include suffix
  if (include_suffix) return path.substr(p);

  int u = path.rfind('.');
  if (u == -1) return path.substr(p);

  if (u <= p) u = path.size();
  return path.substr(p, u - p);
}

/**
 * @brief INFO 宏的真正实现：拼接 "[文件:行号]: " 前缀后按 printf 风格输出到 stdout。
 */
void __log_func(const char *file, int line, const char *fmt, ...) {
  va_list vl;
  va_start(vl, fmt);
  char buffer[2048];
  string filename = file_name(file, true);
  int n = snprintf(buffer, sizeof(buffer), "[%s:%d]: ", filename.c_str(), line);
  vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
  fprintf(stdout, "%s\n", buffer);
}

/// 把 nvinfer1::Dims 格式化为 "AxBxC..." 字符串（内部重载，仅本编译单元可见）。
static std::string format_shape(const Dims &shape) {
  stringstream output;
  char buf[64];
  const char *fmts[] = {"%d", "x%d"};
  for (int i = 0; i < shape.nbDims; ++i) {
    snprintf(buf, sizeof(buf), fmts[i != 0], shape.d[i]);
    output << buf;
  }
  return output.str();
}

/// 构造时创建 start/stop 两个 cudaEvent。
Timer::Timer() {
  checkRuntime(cudaEventCreate((cudaEvent_t *)&start_));
  checkRuntime(cudaEventCreate((cudaEvent_t *)&stop_));
}

/// 析构时销毁 cudaEvent，避免泄漏。
Timer::~Timer() {
  checkRuntime(cudaEventDestroy((cudaEvent_t)start_));
  checkRuntime(cudaEventDestroy((cudaEvent_t)stop_));
}

/// 在指定 stream 上记录起点时间，stream==nullptr 则使用默认流。
void Timer::start(void *stream) {
  stream_ = stream;
  checkRuntime(cudaEventRecord((cudaEvent_t)start_, (cudaStream_t)stream_));
}

/**
 * @brief 记录终点并同步，返回 start→stop 的毫秒数；可选打印。
 */
float Timer::stop(const char *prefix, bool print) {
  checkRuntime(cudaEventRecord((cudaEvent_t)stop_, (cudaStream_t)stream_));
  checkRuntime(cudaEventSynchronize((cudaEvent_t)stop_));

  float latency = 0;
  checkRuntime(cudaEventElapsedTime(&latency, (cudaEvent_t)start_, (cudaEvent_t)stop_));

  if (print) {
    printf("[%s]: %.5f ms\n", prefix, latency);
  }
  return latency;
}

/// 引用外部内存的构造函数（不接管所有权，析构不释放）。
BaseMemory::BaseMemory(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes) {
  reference(cpu, cpu_bytes, gpu, gpu_bytes);
}

/**
 * @brief 引用外部内存：记录指针与字节数，并把 owner 标记设为 false（外部所有）。
 *        若任一指针/字节为 0，则该侧视为未挂载，owner 置 true。
 */
void BaseMemory::reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes) {
  release();

  if (cpu == nullptr || cpu_bytes == 0) {
    cpu = nullptr;
    cpu_bytes = 0;
  }

  if (gpu == nullptr || gpu_bytes == 0) {
    gpu = nullptr;
    gpu_bytes = 0;
  }

  this->cpu_ = cpu;
  this->cpu_capacity_ = cpu_bytes;
  this->cpu_bytes_ = cpu_bytes;
  this->gpu_ = gpu;
  this->gpu_capacity_ = gpu_bytes;
  this->gpu_bytes_ = gpu_bytes;

  this->owner_cpu_ = !(cpu && cpu_bytes > 0);
  this->owner_gpu_ = !(gpu && gpu_bytes > 0);
}

/// 析构：自有的内存通过 cudaFree / cudaFreeHost 释放。
BaseMemory::~BaseMemory() { release(); }

/**
 * @brief 按需扩容设备显存（容量足够则不重新分配，仅更新 gpu_bytes_）。
 * @return 设备指针 gpu_。
 */
void *BaseMemory::gpu_realloc(size_t bytes) {
  if (gpu_capacity_ < bytes) {
    release_gpu();

    gpu_capacity_ = bytes;
    checkRuntime(cudaMalloc(&gpu_, bytes));
    // checkRuntime(cudaMemset(gpu_, 0, size));
  }
  gpu_bytes_ = bytes;
  return gpu_;
}

/**
 * @brief 按需扩容主机锁页内存（cudaMallocHost），便于异步 H2D/D2H 拷贝。
 */
void *BaseMemory::cpu_realloc(size_t bytes) {
  if (cpu_capacity_ < bytes) {
    release_cpu();

    cpu_capacity_ = bytes;
    checkRuntime(cudaMallocHost(&cpu_, bytes));
    Assert(cpu_ != nullptr);
    // memset(cpu_, 0, size);
  }
  cpu_bytes_ = bytes;
  return cpu_;
}

/// 释放 CPU 锁页内存（仅当 owner 为真时调用 cudaFreeHost）。
void BaseMemory::release_cpu() {
  if (cpu_) {
    if (owner_cpu_) {
      checkRuntime(cudaFreeHost(cpu_));
    }
    cpu_ = nullptr;
  }
  cpu_capacity_ = 0;
  cpu_bytes_ = 0;
}

/// 释放设备显存（仅当 owner 为真时调用 cudaFree）。
void BaseMemory::release_gpu() {
  if (gpu_) {
    if (owner_gpu_) {
      checkRuntime(cudaFree(gpu_));
    }
    gpu_ = nullptr;
  }
  gpu_capacity_ = 0;
  gpu_bytes_ = 0;
}

/// 同时释放 CPU 与 GPU 侧资源。
void BaseMemory::release() {
  release_cpu();
  release_gpu();
}

/**
 * @class __native_nvinfer_logger
 * @brief nvinfer1::ILogger 实现，仅打印 ERROR 及以上级别，
 *        遇到 INTERNAL_ERROR 时直接 abort。WARNING/INFO 默认静默。
 */
class __native_nvinfer_logger : public ILogger {
 public:
  virtual void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kINTERNAL_ERROR) {
      INFO("NVInfer INTERNAL_ERROR: %s", msg);
      abort();
    } else if (severity == Severity::kERROR) {
      INFO("NVInfer: %s", msg);
    }
    // else  if (severity == Severity::kWARNING) {
    //     INFO("NVInfer: %s", msg);
    // }
    // else  if (severity == Severity::kINFO) {
    //     INFO("NVInfer: %s", msg);
    // }
    // else {
    //     INFO("%s", msg);
    // }
  }
};
static __native_nvinfer_logger gLogger;

/// 用于 shared_ptr 的自定义 deleter：调用 nvinfer1 对象的 destroy()。
template <typename _T>
static void destroy_nvidia_pointer(_T *ptr) {
  if (ptr) ptr->destroy();
}

/**
 * @brief 把整个 engine 文件读入字节 vector；文件不存在则返回空 vector。
 */
static std::vector<uint8_t> load_file(const string &file) {
  ifstream in(file, ios::in | ios::binary);
  if (!in.is_open()) return {};

  in.seekg(0, ios::end);
  size_t length = in.tellg();

  std::vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, ios::beg);
    data.resize(length);

    in.read((char *)&data[0], length);
  }
  in.close();
  return data;
}

/**
 * @class __native_engine_context
 * @brief Runtime + ICudaEngine + IExecutionContext 三件套的 RAII 持有者。
 *        construct() 反序列化字节流并创建执行上下文；析构自动按相反顺序释放。
 */
class __native_engine_context {
 public:
  virtual ~__native_engine_context() { destroy(); }

  bool construct(const void *pdata, size_t size) {
    destroy();

    if (pdata == nullptr || size == 0) return false;

    runtime_ = shared_ptr<IRuntime>(createInferRuntime(gLogger), destroy_nvidia_pointer<IRuntime>);
    if (runtime_ == nullptr) return false;

    engine_ = shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size, nullptr),
                                      destroy_nvidia_pointer<ICudaEngine>);
    if (engine_ == nullptr) return false;

    context_ = shared_ptr<IExecutionContext>(engine_->createExecutionContext(),
                                             destroy_nvidia_pointer<IExecutionContext>);
    return context_ != nullptr;
  }

 private:
  void destroy() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
  }

 public:
  shared_ptr<IExecutionContext> context_;
  shared_ptr<ICudaEngine> engine_;
  shared_ptr<IRuntime> runtime_ = nullptr;
};

/**
 * @class InferImpl
 * @brief trt::Infer 的实现：组合 __native_engine_context 并维护 binding 名→index 映射。
 */
class InferImpl : public Infer {
 public:
  shared_ptr<__native_engine_context> context_;
  unordered_map<string, int> binding_name_to_index_;

  virtual ~InferImpl() = default;

  /// 由内存中的 engine 字节构造（反序列化 + setup）。
  bool construct(const void *data, size_t size) {
    context_ = make_shared<__native_engine_context>();
    if (!context_->construct(data, size)) {
      return false;
    }

    setup();
    return true;
  }

  /// 从文件加载 engine 并 construct，文件为空时打印 INFO 并返回 false。
  bool load(const string &file) {
    auto data = load_file(file);
    if (data.empty()) {
      INFO("An empty file has been loaded. Please confirm your file path: %s", file.c_str());
      return false;
    }
    return this->construct(data.data(), data.size());
  }

  /// 遍历 engine 所有 binding，构建 名称→index 哈希表，便于按名查询。
  void setup() {
    auto engine = this->context_->engine_;
    int nbBindings = engine->getNbBindings();

    binding_name_to_index_.clear();
    for (int i = 0; i < nbBindings; ++i) {
      const char *bindingName = engine->getBindingName(i);
      binding_name_to_index_[bindingName] = i;
    }
  }

  /// 名称查 index；若不存在直接 abort。
  virtual int index(const std::string &name) override {
    auto iter = binding_name_to_index_.find(name);
    Assertf(iter != binding_name_to_index_.end(), "Can not found the binding name: %s",
            name.c_str());
    return iter->second;
  }

  /// 调用 IExecutionContext::enqueueV2，异步推理。
  virtual bool forward(const std::vector<void *> &bindings, void *stream,
                       void *input_consum_event) override {
    return this->context_->context_->enqueueV2((void**)bindings.data(), (cudaStream_t)stream,
                                               (cudaEvent_t *)input_consum_event);
  }

  /// 名称版本：转 index 后调用同名重载。
  virtual std::vector<int> run_dims(const std::string &name) override {
    return run_dims(index(name));
  }

  /// 取当前 context 的运行时维度（动态 shape 下随 set_run_dims 变化）。
  virtual std::vector<int> run_dims(int ibinding) override {
    auto dim = this->context_->context_->getBindingDimensions(ibinding);
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
  }

  /// 名称版本：转 index 后调用同名重载。
  virtual std::vector<int> static_dims(const std::string &name) override {
    return static_dims(index(name));
  }

  /// 取 engine 中静态维度（动态轴为 -1）。
  virtual std::vector<int> static_dims(int ibinding) override {
    auto dim = this->context_->engine_->getBindingDimensions(ibinding);
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
  }

  /// engine 中的 binding 数量（含输入与输出）。
  virtual int num_bindings() override { return this->context_->engine_->getNbBindings(); }

  /// 该 binding 是否为输入。
  virtual bool is_input(int ibinding) override {
    return this->context_->engine_->bindingIsInput(ibinding);
  }

  /// 名称版本：动态 shape 时设置实际维度。
  virtual bool set_run_dims(const std::string &name, const std::vector<int> &dims) override {
    return this->set_run_dims(index(name), dims);
  }

  /**
   * @brief 设置某个 binding 在本次推理时使用的形状（动态 shape 必备）。
   * @return setBindingDimensions 的返回值。
   */
  virtual bool set_run_dims(int ibinding, const std::vector<int> &dims) override {
    Dims d;
    memcpy(d.d, dims.data(), sizeof(int) * dims.size());
    d.nbDims = dims.size();
    return this->context_->context_->setBindingDimensions(ibinding, d);
  }

  /// 名称版本元素总数。
  virtual int numel(const std::string &name) override { return numel(index(name)); }

  /// 元素总数 = run_dims 各维度乘积。
  virtual int numel(int ibinding) override {
    auto dim = this->context_->context_->getBindingDimensions(ibinding);
    return std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<int>());
  }

  /// 名称版本 dtype 查询。
  virtual DType dtype(const std::string &name) override { return dtype(index(name)); }

  /// 转换 nvinfer1::DataType → trt::DType。
  virtual DType dtype(int ibinding) override {
    return (DType)this->context_->engine_->getBindingDataType(ibinding);
  }

  /// 任一 binding 的任一维度为 -1 即视为动态 shape。
  virtual bool has_dynamic_dim() override {
    // check if any input or output bindings have dynamic shapes
    int numBindings = this->context_->engine_->getNbBindings();
    for (int i = 0; i < numBindings; ++i) {
      nvinfer1::Dims dims = this->context_->engine_->getBindingDimensions(i);
      for (int j = 0; j < dims.nbDims; ++j) {
        if (dims.d[j] == -1) return true;
      }
    }
    return false;
  }

  /// 打印 engine 概要：动/静态、输入与输出 binding 名称及形状。
  virtual void print() override {
    INFO("Infer %p [%s]", this, has_dynamic_dim() ? "DynamicShape" : "StaticShape");

    int num_input = 0;
    int num_output = 0;
    auto engine = this->context_->engine_;
    for (int i = 0; i < engine->getNbBindings(); ++i) {
      if (engine->bindingIsInput(i))
        num_input++;
      else
        num_output++;
    }

    INFO("Inputs: %d", num_input);
    for (int i = 0; i < num_input; ++i) {
      auto name = engine->getBindingName(i);
      auto dim = engine->getBindingDimensions(i);
      INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
    }

    INFO("Outputs: %d", num_output);
    for (int i = 0; i < num_output; ++i) {
      auto name = engine->getBindingName(i + num_input);
      auto dim = engine->getBindingDimensions(i + num_input);
      INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
    }
  }
};

/// 工厂：new InferImpl + load；失败返回 nullptr，由调用方 delete。
Infer *loadraw(const std::string &file) {
  InferImpl *impl = new InferImpl();
  if (!impl->load(file)) {
    delete impl;
    impl = nullptr;
  }
  return impl;
}

/// 智能指针版本工厂，包装 loadraw 结果。
std::shared_ptr<Infer> load(const std::string &file) {
  return std::shared_ptr<InferImpl>((InferImpl *)loadraw(file));
}

/// 公有 API：把 vector<int> 形状格式化为 "AxBxC..." 字符串。
std::string format_shape(const std::vector<int> &shape) {
  stringstream output;
  char buf[64];
  const char *fmts[] = {"%d", "x%d"};
  for (int i = 0; i < (int)shape.size(); ++i) {
    snprintf(buf, sizeof(buf), fmts[i != 0], shape[i]);
    output << buf;
  }
  return output.str();
}
};  // namespace trt
