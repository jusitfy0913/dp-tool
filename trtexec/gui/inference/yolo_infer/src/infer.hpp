/**
 * @file    infer.hpp
 * @brief   Lightweight wrap over TensorRT (nvinfer1); hides direct NvInfer.h includes.
 *
 * Provides:
 *  - INFO / __log_func     : simple logging macros
 *  - DType                 : enum aligned with nvinfer1::DataType
 *  - Timer                 : GPU timer via cudaEvent
 *  - BaseMemory / Memory   : host/pinned+device memory RAII (realloc on demand)
 *  - Infer                 : engine inference (forward, dims, dtype)
 *  - load / format_shape   : factory and shape formatting
 *
 * Implementation is in infer.cu, used by yolo.cu.
 */
#ifndef __INFER_HPP__
#define __INFER_HPP__

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace trt {

#define INFO(...) trt::__log_func(__FILE__, __LINE__, __VA_ARGS__)
void __log_func(const char *file, int line, const char *fmt, ...);

enum class DType : int { FLOAT = 0, HALF = 1, INT8 = 2, INT32 = 3, BOOL = 4, UINT8 = 5 };

/**
 * @class Timer
 * @brief GPU segment timer using cudaEvent. Call start(stream) ... stop("tag").
 */
class Timer {
 public:
  Timer();
  virtual ~Timer();
  void start(void *stream = nullptr);
  float stop(const char *prefix = "Timer", bool print = true);

 private:
  void *start_, *stop_;
  void *stream_;
};

/**
 * @class BaseMemory
 * @brief RAII for pinned host + device memory; auto-grow on demand.
 *
 * - cpu_realloc / gpu_realloc only reallocate when capacity is insufficient
 * - reference() can attach external buffers (no ownership)
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
 * @brief Typed BaseMemory: sizes are element counts, not bytes.
 * @tparam _DT element type
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
 * @brief TensorRT engine wrapper around IExecutionContext->enqueueV2, etc.
 *
 * Typical use:
 *  1) load(engine_file) -> shared_ptr<Infer>
 *  2) static_dims / run_dims / has_dynamic_dim for I/O shapes
 *  3) For dynamic models, set_run_dims for this run
 *  4) Prepare bindings (GPU pointer array)
 *  5) forward(bindings, stream)
 */
class Infer {
 public:
  virtual bool forward(const std::vector<void *> &bindings, void *stream = nullptr,
                       void *input_consum_event = nullptr) = 0;
  virtual int index(const std::string &name) = 0;
  virtual std::vector<int> run_dims(const std::string &name) = 0;
  virtual std::vector<int> run_dims(int ibinding) = 0;
  virtual std::vector<int> static_dims(const std::string &name) = 0;
  virtual std::vector<int> static_dims(int ibinding) = 0;
  virtual int numel(const std::string &name) = 0;
  virtual int numel(int ibinding) = 0;
  virtual int num_bindings() = 0;
  virtual bool is_input(int ibinding) = 0;
  virtual bool set_run_dims(const std::string &name, const std::vector<int> &dims) = 0;
  virtual bool set_run_dims(int ibinding, const std::vector<int> &dims) = 0;
  virtual DType dtype(const std::string &name) = 0;
  virtual DType dtype(int ibinding) = 0;
  virtual bool has_dynamic_dim() = 0;
  virtual void print() = 0;
};

std::shared_ptr<Infer> load(const std::string &file);
std::string format_shape(const std::vector<int> &shape);

}  // namespace trt

#endif  // __INFER_HPP__
