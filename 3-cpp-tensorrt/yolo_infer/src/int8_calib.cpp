// INT8 校准并编译 YOLO 引擎。
//
// 预处理和 yolo.cu 里的推理对齐：letterbox、灰边 114、BGR 转 RGB、除以 255。
// 成功后写出引擎、校准缓存，并把 .key.pending 改名为 .key，供下次跳过读图。
//
// 日志里的几行英文标记是给网页进度条认的，不要改字：
//   &&&& RUNNING / Start parsing network model / Finish parsing network model
//   Local timing cache in use / Engine built in / &&&& PASSED

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[INT8] " << msg << std::endl;
    }
  }
};

struct Options {
  std::string onnx;
  std::string save;
  std::string cache;
  std::string list;
  std::string key_pending;
  int workspace_mb = 4096;
  std::string min_shape;
  std::string opt_shape;
  std::string max_shape;
};

bool parse_shape(const std::string& text, nvinfer1::Dims4& dims) {
  int n = 0, c = 0, h = 0, w = 0;
  char x1 = 0, x2 = 0, x3 = 0;
  std::stringstream stream(text);
  stream >> n >> x1 >> c >> x2 >> h >> x3 >> w;
  if (!stream || x1 != 'x' || x2 != 'x' || x3 != 'x' || n <= 0 || c <= 0 || h <= 0 || w <= 0) {
    return false;
  }
  dims = nvinfer1::Dims4{n, c, h, w};
  return true;
}

std::vector<std::string> read_list(const std::string& path) {
  std::vector<std::string> lines;
  std::ifstream input(path);
  std::string line;
  while (std::getline(input, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (!line.empty()) lines.push_back(line);
  }
  return lines;
}

void letterbox(const cv::Mat& bgr, int dst_w, int dst_h, std::vector<float>& chw) {
  const int src_w = bgr.cols;
  const int src_h = bgr.rows;
  const float scale = std::min(dst_w / static_cast<float>(src_w), dst_h / static_cast<float>(src_h));
  const float i2d2 = -scale * src_w * 0.5f + dst_w * 0.5f + scale * 0.5f - 0.5f;
  const float i2d5 = -scale * src_h * 0.5f + dst_h * 0.5f + scale * 0.5f - 0.5f;
  const double inv = scale == 0.f ? 0.0 : 1.0 / static_cast<double>(scale);
  cv::Mat matrix(2, 3, CV_64F);
  matrix.at<double>(0, 0) = inv;
  matrix.at<double>(0, 1) = 0.0;
  matrix.at<double>(0, 2) = -inv * i2d2;
  matrix.at<double>(1, 0) = 0.0;
  matrix.at<double>(1, 1) = inv;
  matrix.at<double>(1, 2) = -inv * i2d5;

  cv::Mat resized;
  cv::warpAffine(bgr, resized, matrix, cv::Size(dst_w, dst_h), cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  const int area = dst_w * dst_h;
  chw.assign(static_cast<size_t>(area) * 3, 0.f);
  for (int y = 0; y < dst_h; ++y) {
    const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
    for (int x = 0; x < dst_w; ++x) {
      const int index = y * dst_w + x;
      chw[index] = row[x][2] / 255.f;
      chw[area + index] = row[x][1] / 255.f;
      chw[area * 2 + index] = row[x][0] / 255.f;
    }
  }
}

class Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  Calibrator(std::vector<std::string> files, int height, int width, std::string cache_path)
      : files_(std::move(files)), height_(height), width_(width), cache_path_(std::move(cache_path)) {
    cudaMalloc(&device_, static_cast<size_t>(height_) * width_ * 3 * sizeof(float));
  }

  ~Calibrator() override {
    if (device_) cudaFree(device_);
  }

  int getBatchSize() const noexcept override { return 1; }

  bool getBatch(void* bindings[], const char* names[], int nb_bindings) noexcept override {
    while (cursor_ < files_.size()) {
      const std::string path = files_[cursor_++];
      cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
      if (image.empty()) {
        std::cout << "跳过读不了的图：" << path << std::endl;
        continue;
      }
      std::vector<float> host;
      letterbox(image, width_, height_, host);
      cudaMemcpy(device_, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
      for (int i = 0; i < nb_bindings; ++i) bindings[i] = device_;
      std::cout << "校准 " << cursor_ << "/" << files_.size() << " " << path << std::endl;
      (void)names;
      return true;
    }
    return false;
  }

  const void* readCalibrationCache(std::size_t& length) noexcept override {
    length = 0;
    return nullptr;
  }

  void writeCalibrationCache(const void* cache, std::size_t length) noexcept override {
    std::ofstream output(cache_path_, std::ios::binary);
    if (!output) {
      std::cout << "写校准缓存失败：" << cache_path_ << std::endl;
      return;
    }
    output.write(static_cast<const char*>(cache), static_cast<std::streamsize>(length));
    std::cout << "校准缓存已写入 " << cache_path_ << std::endl;
  }

 private:
  std::vector<std::string> files_;
  int height_ = 0;
  int width_ = 0;
  std::string cache_path_;
  void* device_ = nullptr;
  std::size_t cursor_ = 0;
};

void finish_key(const std::string& pending) {
  if (pending.empty()) return;
  const std::string target = pending.substr(0, pending.size() - std::string(".pending").size());
  std::remove(target.c_str());
  if (std::rename(pending.c_str(), target.c_str()) != 0) {
    std::cout << "校准完成，但标记文件没能改名：" << pending << std::endl;
  }
}

int run(const Options& options) {
  std::cout << "&&&& RUNNING" << std::endl;
  std::cout << "INT8 为主，算不了的层用 FP16。灰边 114，像素除以 255。" << std::endl;
  auto files = read_list(options.list);
  if (files.empty()) {
    std::cerr << "校准图片清单是空的：" << options.list << std::endl;
    return 2;
  }

  Logger logger;
  std::cout << "Start parsing network model" << std::endl;
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) return 3;
  const auto flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
  auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
  if (!parser->parseFromFile(options.onnx.c_str(), 2)) {
    std::cerr << "ONNX 解析失败：" << options.onnx << std::endl;
    return 4;
  }
  std::cout << "Finish parsing network model" << std::endl;

  auto* input = network->getInput(0);
  if (!input) {
    std::cerr << "ONNX 没有输入" << std::endl;
    return 4;
  }
  nvinfer1::Dims4 min_dims{};
  nvinfer1::Dims4 opt_dims{};
  nvinfer1::Dims4 max_dims{};
  auto network_dims = input->getDimensions();
  const bool dynamic = options.min_shape.size() && options.opt_shape.size() && options.max_shape.size();
  if (dynamic) {
    if (!parse_shape(options.min_shape, min_dims) || !parse_shape(options.opt_shape, opt_dims) ||
        !parse_shape(options.max_shape, max_dims)) {
      std::cerr << "shape 必须是 1x3x高x宽" << std::endl;
      return 2;
    }
  } else {
    if (network_dims.nbDims != 4) {
      std::cerr << "只支持 4 维输入，请在页面上填写动态 shape" << std::endl;
      return 2;
    }
    for (int i = 0; i < 4; ++i) {
      if (network_dims.d[i] <= 0) {
        std::cerr << "ONNX 输入尺寸不固定，请填写 min / opt / max" << std::endl;
        return 2;
      }
    }
    min_dims = opt_dims = max_dims =
        nvinfer1::Dims4{network_dims.d[0], network_dims.d[1], network_dims.d[2], network_dims.d[3]};
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                              static_cast<std::size_t>(options.workspace_mb) * 1024 * 1024);
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  config->setFlag(nvinfer1::BuilderFlag::kINT8);

  auto* profile = builder->createOptimizationProfile();
  profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, min_dims);
  profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
  profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, max_dims);
  config->addOptimizationProfile(profile);

  auto* calib_profile = builder->createOptimizationProfile();
  calib_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, opt_dims);
  calib_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
  calib_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, opt_dims);
  config->setCalibrationProfile(calib_profile);

  Calibrator calibrator(std::move(files), opt_dims.d[2], opt_dims.d[3], options.cache);
  config->setInt8Calibrator(&calibrator);

  std::cout << "Local timing cache in use" << std::endl;
  const auto started = std::chrono::steady_clock::now();
  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    std::cerr << "INT8 引擎编译失败" << std::endl;
    std::cout << "&&&& FAILED" << std::endl;
    return 5;
  }
  std::ofstream engine(options.save, std::ios::binary);
  if (!engine) {
    std::cerr << "写引擎失败：" << options.save << std::endl;
    return 5;
  }
  engine.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
  engine.close();
  const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  std::cout << "Engine built in " << seconds << " sec." << std::endl;
  finish_key(options.key_pending);
  std::cout << "&&&& PASSED" << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](std::string& slot) {
      if (i + 1 >= argc) return false;
      slot = argv[++i];
      return true;
    };
    if (arg == "--onnx") { if (!need(options.onnx)) return 2; }
    else if (arg == "--save") { if (!need(options.save)) return 2; }
    else if (arg == "--cache") { if (!need(options.cache)) return 2; }
    else if (arg == "--list") { if (!need(options.list)) return 2; }
    else if (arg == "--key-pending") { if (!need(options.key_pending)) return 2; }
    else if (arg == "--workspace") {
      if (i + 1 >= argc) return 2;
      options.workspace_mb = std::atoi(argv[++i]);
    }
    else if (arg == "--min") { if (!need(options.min_shape)) return 2; }
    else if (arg == "--opt") { if (!need(options.opt_shape)) return 2; }
    else if (arg == "--max") { if (!need(options.max_shape)) return 2; }
    else {
      std::cerr << "不认识的参数：" << arg << std::endl;
      return 2;
    }
  }
  if (options.onnx.empty() || options.save.empty() || options.cache.empty() || options.list.empty()) {
    std::cerr << "需要 --onnx --save --cache --list" << std::endl;
    return 2;
  }
  try {
    return run(options);
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << std::endl;
    return 5;
  }
}
