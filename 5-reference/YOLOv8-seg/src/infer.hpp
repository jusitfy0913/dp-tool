/**
 * @file    infer.hpp
 * @brief   对 TensorRT (nvinfer1) 的轻量包装，屏蔽 NvInfer.h 头文件依赖。
 *
 * 提供：
 *  - INFO / __log_func     : 简易日志宏；
 *  - DType                 : 与 nvinfer1::DataType 同语义的枚举；
 *  - Timer                 : 基于 cudaEvent 的 GPU 计时器；
 *  - BaseMemory / Memory   : 主机/设备内存的 RAII 容器（按需 realloc）；
 *  - Infer                 : engine 推理抽象（forward / 维度 / dtype 查询等）；
 *  - load / format_shape   : 工厂与工具函数。
 *
 * 实现见 infer.cu，被 yolo.cu 使用。
 */
#ifndef __INFER_HPP__
#define __INFER_HPP__

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace trt {

/// 日志宏，自动带上文件名与行号，调用 __log_func。
#define INFO(...) trt::__log_func(__FILE__, __LINE__, __VA_ARGS__)
/// 内部日志实现，等价于 printf 风格输出。
void __log_func(const char *file, int line, const char *fmt, ...);

/// 与 nvinfer1::DataType 对应的数据类型枚举（避免泄露 NvInfer.h）。
enum class DType : int { FLOAT = 0, HALF = 1, INT8 = 2, INT32 = 3, BOOL = 4, UINT8 = 5 };

/**
 * @class Timer
 * @brief 基于 cudaEvent 的 GPU 段计时器。用法：start(stream) ... stop("tag")。
 */
class Timer {
 public:
  Timer();           ///< 构造时创建两个 cudaEvent_t
  virtual ~Timer();  ///< 析构时销毁 event
  /// 在指定 stream 上记录起点；stream=nullptr 表示默认流。
  void start(void *stream = nullptr);
  /// 在同一 stream 上记录终点并同步，返回毫秒；print=true 时打印 "[prefix]: x ms"。
  float stop(const char *prefix = "Timer", bool print = true);

 private:
  void *start_, *stop_;
  void *stream_;
};

/**
 * @class BaseMemory
 * @brief 同时管理主机锁页内存与设备显存的 RAII 容器，按需自动扩容。
 *
 * - cpu_realloc / gpu_realloc 仅在容量不足时才真正重新分配；
 * - reference() 可挂载外部内存（不接管所有权）。
 */
class BaseMemory {
 public:
  BaseMemory() = default;
  BaseMemory(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);
  virtual ~BaseMemory();
  virtual void *gpu_realloc(size_t bytes);
  virtual void *cpu_realloc(size_t bytes);
  void release_gpu();
  void release_cpu();
  void release();
  inline bool owner_gpu() const { return owner_gpu_; }
  inline bool owner_cpu() const { return owner_cpu_; }
  inline size_t cpu_bytes() const { return cpu_bytes_; }
  inline size_t gpu_bytes() const { return gpu_bytes_; }
  virtual inline void *get_gpu() const { return gpu_; }
  virtual inline void *get_cpu() const { return cpu_; }
  void reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);

 protected:
  void *cpu_ = nullptr;
  size_t cpu_bytes_ = 0, cpu_capacity_ = 0;
  bool owner_cpu_ = true;

  void *gpu_ = nullptr;
  size_t gpu_bytes_ = 0, gpu_capacity_ = 0;
  bool owner_gpu_ = true;
};

/**
 * @class Memory
 * @brief BaseMemory 的类型化封装：以元素个数（而非字节数）为单位管理内存。
 * @tparam _DT 元素类型
 */
template <typename _DT>
class Memory : public BaseMemory {
 public:
  Memory() = default;
  Memory(const Memory &other) = delete;
  Memory &operator=(const Memory &other) = delete;
  virtual _DT *gpu(size_t size) { return (_DT *)BaseMemory::gpu_realloc(size * sizeof(_DT)); }
  virtual _DT *cpu(size_t size) { return (_DT *)BaseMemory::cpu_realloc(size * sizeof(_DT)); }

  inline size_t cpu_size() const { return cpu_bytes_ / sizeof(_DT); }
  inline size_t gpu_size() const { return gpu_bytes_ / sizeof(_DT); }

  virtual inline _DT *gpu() const { return (_DT *)gpu_; }
  virtual inline _DT *cpu() const { return (_DT *)cpu_; }
};

/**
 * @class Infer
 * @brief TensorRT 引擎推理抽象，封装 IExecutionContext->enqueueV2 等常见操作。
 *
 * 典型用法：
 *  1) load(engine_file) 拿到 shared_ptr<Infer>；
 *  2) 用 static_dims/run_dims/has_dynamic_dim 查询输入输出形状；
 *  3) 动态 shape 时调用 set_run_dims 设定本次形状；
 *  4) 准备 bindings(GPU 指针数组)；
 *  5) forward(bindings, stream) 执行推理。
 */
class Infer {
 public:
  /**
   * @brief 异步执行一次 enqueueV2。
   * @param bindings           按 binding 顺序排列的 GPU 指针
   * @param stream             cudaStream_t；nullptr 表示默认流
   * @param input_consum_event 可选 cudaEvent_t，TRT 会在输入消费完后 record
   */
  virtual bool forward(const std::vector<void *> &bindings, void *stream = nullptr,
                       void *input_consum_event = nullptr) = 0;
  /// 名称 → binding index
  virtual int index(const std::string &name) = 0;
  /// 当前 context 的运行时维度（动态 shape 下随 set_run_dims 变化）
  virtual std::vector<int> run_dims(const std::string &name) = 0;
  virtual std::vector<int> run_dims(int ibinding) = 0;
  /// engine 中的静态维度（动态轴为 -1）
  virtual std::vector<int> static_dims(const std::string &name) = 0;
  virtual std::vector<int> static_dims(int ibinding) = 0;
  /// 元素总数（dims 各维度乘积）
  virtual int numel(const std::string &name) = 0;
  virtual int numel(int ibinding) = 0;
  virtual int num_bindings() = 0;
  virtual bool is_input(int ibinding) = 0;
  /// 动态 shape 模型设置本次推理的实际维度
  virtual bool set_run_dims(const std::string &name, const std::vector<int> &dims) = 0;
  virtual bool set_run_dims(int ibinding, const std::vector<int> &dims) = 0;
  virtual DType dtype(const std::string &name) = 0;
  virtual DType dtype(int ibinding) = 0;
  /// 是否包含至少一个动态维度（dims.d[*]==-1）
  virtual bool has_dynamic_dim() = 0;
  /// 打印输入输出 binding 信息（名称 + 形状）。
  virtual void print() = 0;
};

/// 工厂函数：读取 engine 文件并构造 Infer 实现，失败返回 nullptr。
std::shared_ptr<Infer> load(const std::string &file);
/// 把 vector<int> 形状格式化为类似 "1x3x640x640" 的字符串。
std::string format_shape(const std::vector<int> &shape);

}  // namespace trt

#endif  // __INFER_HPP__
