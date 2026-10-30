//
// Created by ubuntu on 10/25/24.
//
/**
 * @file    yolov8-obb.hpp
 * @brief   YOLOv8 OBB（Oriented Bounding Box，旋转框检测）的 TensorRT 推理封装。
 *
 * 工作流程：
 *  1. 反序列化 .engine 文件并创建 IExecutionContext；
 *  2. 解析输入/输出绑定信息（Binding）；
 *  3. make_pipe 分配 device/host 缓冲并可选预热；
 *  4. letterbox 做等比缩放+114灰边填充并转 NCHW float32，记录 PreParam；
 *  5. infer 在 CUDA 流上异步推理并把输出拷回 host；
 *  6. postprocess 解析每个 anchor → 还原原图坐标 → cv::dnn::NMSBoxes（旋转框 NMS）→ 取 topk；
 *  7. draw_objects 用 cv::boxPoints + polylines 画出旋转框与文字标签。
 *
 * 输出张量约定（YOLOv8-OBB 默认）：
 *   shape = [1, 4 + num_labels + 1, num_anchors]
 *           ↑   ↑   ↑                ↑
 *           |   |   类别打分          角度（弧度）
 *           |   xywh
 *   后处理时先转置为 [num_anchors, channels] 便于按行遍历。
 */
#ifndef OBB_NORMAL_YOLOv8_obb_HPP
#define OBB_NORMAL_YOLOv8_obb_HPP

#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>

using namespace obb;

/**
 * @class YOLOv8_obb
 * @brief 封装 OBB engine 的加载、letterbox 预处理、推理与旋转框后处理。
 */
class YOLOv8_obb {
public:
    /**
     * @brief 构造函数：读 engine → 反序列化 → 建立上下文 → 解析绑定。
     * @param engine_file_path 序列化引擎文件路径。
     */
    explicit YOLOv8_obb(const std::string& engine_file_path);

    /// 析构函数：按依赖反向释放 TensorRT 与 CUDA 资源。
    ~YOLOv8_obb();

    /**
     * @brief 给所有 binding 分配 device/host 缓冲；warmup=true 时用全 0 推理 10 次预热。
     */
    void make_pipe(bool warmup = true);

    /**
     * @brief 使用 engine 输入维度自动 letterbox + 上传到显存。
     */
    void copy_from_Mat(const cv::Mat& image);

    /**
     * @brief 指定目标 size 版本：letterbox + 上传到显存。
     */
    void copy_from_Mat(const cv::Mat& image, cv::Size& size);

    /**
     * @brief letterbox 预处理：等比 resize + 灰边填充至 size，生成 NCHW float32，
     *        同时把 ratio / dw / dh / 原图尺寸写入 pparam，供后处理还原坐标。
     * @param image 原图（BGR）
     * @param[out] out 预处理后的 NCHW float32 blob
     * @param size 目标分辨率（如 1024x1024）
     */
    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);

    /// 在 CUDA 流上异步推理并将所有输出 D2H 拷回 host_ptrs。
    void infer();

    /**
     * @brief OBB 后处理：阈值过滤 → 反 letterbox → 旋转框 NMS → 取 topk。
     * @param[out] objs       输出旋转框列表
     * @param score_thres     置信度阈值
     * @param iou_thres       NMS 的 IOU 阈值
     * @param topk            每张图保留的最大检测框数
     * @param num_labels      类别数（DOTA 默认 15）
     */
    void postprocess(std::vector<Object>& objs,
                     float                score_thres = 0.25f,
                     float                iou_thres   = 0.65f,
                     int                  topk        = 100,
                     int                  num_labels  = 15);

    /**
     * @brief 在 res 上画出每个旋转框（boxPoints+polylines）以及 "类别 概率%" 文字。
     */
    static void draw_objects(const cv::Mat&                                image,
                             cv::Mat&                                      res,
                             const std::vector<Object>&                    objs,
                             const std::vector<std::string>&               CLASS_NAMES,
                             const std::vector<std::vector<unsigned int>>& COLORS);

    int                  num_bindings;       ///< engine 全部绑定数
    int                  num_inputs  = 0;    ///< 输入绑定数
    int                  num_outputs = 0;    ///< 输出绑定数
    std::vector<Binding> input_bindings;     ///< 所有输入绑定的元信息
    std::vector<Binding> output_bindings;    ///< 所有输出绑定的元信息
    std::vector<void*>   host_ptrs;          ///< 锁页主机内存指针（与 output_bindings 一一对应）
    std::vector<void*>   device_ptrs;        ///< 设备内存指针（输入在前，输出在后）

    PreParam pparam;                         ///< 最近一次 letterbox 的预处理参数

private:
    nvinfer1::ICudaEngine*       engine  = nullptr; ///< 反序列化得到的引擎对象
    nvinfer1::IRuntime*          runtime = nullptr; ///< TensorRT 运行时
    nvinfer1::IExecutionContext* context = nullptr; ///< 执行上下文
    cudaStream_t                 stream  = nullptr; ///< CUDA 流
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR}; ///< 仅打印 ERROR 及以上
};

/**
 * @brief 构造函数实现：读取 engine 文件 → 反序列化 → 建立运行时与上下文 → 解析所有 binding。
 */
YOLOv8_obb::YOLOv8_obb(const std::string& engine_file_path)
{
    // 1. 以二进制方式将整份 engine 读入内存缓冲
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // 2. 注册 TensorRT 内置插件并创建运行时
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    // 3. 反序列化引擎、释放内存缓冲并建立执行上下文
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    // 4. 创建独立 CUDA 流，便于异步 H2D / 推理 / D2H 流水线
    cudaStreamCreate(&this->stream);

    // 5. 遍历所有 binding，分别记录到 input_bindings / output_bindings
    this->num_bindings = this->engine->getNbBindings();
    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name  = this->engine->getBindingName(i);
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs += 1;
            // 输入：取 profile 0 中允许的最大 shape，按此分配显存
            dims         = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // 先把上下文输入维度置为最大值，后续 copy_from_Mat 会按真实尺寸覆盖
            this->context->setBindingDimensions(i, dims);
        }
        else {
            // 输出：在输入维度确定后查询输出维度
            dims         = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

/**
 * @brief 析构函数：按构造顺序的反向释放 TensorRT 与 CUDA 资源。
 */
YOLOv8_obb::~YOLOv8_obb()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }
    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

/**
 * @brief 给所有输入/输出绑定分配设备/主机缓冲；warmup=true 时用全 0 输入推理 10 次预热。
 */
void YOLOv8_obb::make_pipe(bool warmup)
{
    // 输入只需要 device 内存；host 数据由 OpenCV blob 临时提供
    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    // 输出需要 device 内存接收推理结果 + host 锁页内存做异步拷回
    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    // 预热：全 0 输入连续推理 10 次，触发 kernel autotune、稳定时延曲线
    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

/**
 * @brief letterbox 预处理：保持宽高比缩放到目标 size 内，四周填充 (114,114,114)。
 *        最后用 cv::dnn::blobFromImage 完成 BGR→RGB / /255 / HWC→NCHW，
 *        并把还原所需的 ratio / dw / dh / 原图尺寸记录到 pparam。
 */
void YOLOv8_obb::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    // 取较小的缩放比例，保证图像完整放入目标分辨率
    float r    = (std::min)(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    // 等比缩放（若已是目标尺寸则只克隆）
    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    // 计算单边填充量，目标尺寸 - 缩放后尺寸 后均分到两侧
    float dw = inp_w - padw;
    float dh = inp_h - padh;
    dw /= 2.0f;
    dh /= 2.0f;

    // 用 abs+round 兼容某些边缘情况，避免出现负值导致 copyMakeBorder 报错
    //int top    = int(std::round(dh - 0.1f));
    //int bottom = int(std::round(dh + 0.1f));
    //int left   = int(std::round(dw - 0.1f));
    //int right  = int(std::round(dw + 0.1f));
    int top = int(std::abs(std::round(dh - 0.1f)));
    int bottom = int(std::abs(std::round(dh + 0.1f)));
    int left = int(std::abs(std::round(dw - 0.1f)));
    int right = int(std::abs(std::round(dw + 0.1f)));

    // 灰边填充到目标尺寸，颜色 (114,114,114) 与训练侧 letterbox 保持一致
    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    // 转 NCHW float32 blob，归一化到 [0,1]，swapRB=true（BGR→RGB）
    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);

    // 记录预处理参数，供后处理把模型坐标还原回原图坐标
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
    ;
}

/**
 * @brief 使用 engine 输入维度自动 letterbox 后上传到显存。
 */
void YOLOv8_obb::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    auto     width      = in_binding.dims.d[3];
    auto     height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    // 显式告诉 TensorRT 本次推理输入形状
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

/**
 * @brief 指定 size 版本：letterbox 到给定分辨率（OBB 默认 1024×1024）后上传到显存。
 */
void YOLOv8_obb::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

/**
 * @brief 在 CUDA 流上排队推理，并把所有输出 D2H 拷回 host_ptrs，最后做流同步。
 */
void YOLOv8_obb::infer()
{
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

/**
 * @brief OBB 后处理：解析每个 anchor → 反 letterbox → 旋转框 NMS → 取 topk。
 *
 * 输出张量原始 shape 为 [num_channels, num_anchors]；
 * 转置后按行遍历，每行布局：[cx, cy, w, h, score_0..score_{num_labels-1}, angle_rad]
 */
void YOLOv8_obb::postprocess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int num_labels)
{
    objs.clear();
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors  = this->output_bindings[0].dims.d[2];

    // 取出 letterbox 阶段记录的还原参数
    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    std::vector<cv::RotatedRect> bboxes;
    std::vector<float>           scores;
    std::vector<int>             labels;
    std::vector<int>             indices;

    // 把输出 wrap 成 cv::Mat 并转置为 [num_anchors, num_channels]，便于按行解析
    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr    = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;                  // 前 4 个：cx, cy, w, h
        auto scores_ptr = row_ptr + 4;              // 紧接 num_labels 个类别分数
        auto max_s_ptr  = std::max_element(scores_ptr, scores_ptr + num_labels);
        auto angle_ptr  = row_ptr + 4 + num_labels; // 最后 1 个：旋转角（弧度）

        float score = *max_s_ptr;
        if (score > score_thres) {
            // 反 letterbox：先减边距 dw/dh，再乘缩放比 ratio 还原到原图坐标
            float x = (*bboxes_ptr++ - dw) * ratio;
            float y = (*bboxes_ptr++ - dh) * ratio;
            float w = (*bboxes_ptr++) * ratio;
            float h = (*bboxes_ptr) * ratio;

            // 过滤极小框
            if (w < 1.f || h < 1.f) {
                continue;
            }

            // 限定到原图范围内，避免越界
            x = clamp(x, 0.f, width);
            y = clamp(y, 0.f, height);
            w = clamp(w, 0.f, width);
            h = clamp(h, 0.f, height);

            // OBB 模型输出弧度，cv::RotatedRect.angle 期望度
            float angle = *angle_ptr / CV_PI * 180.f;

            cv::RotatedRect bbox;
            bbox.center.x    = x;
            bbox.center.y    = y;
            bbox.size.width  = w;
            bbox.size.height = h;
            bbox.angle       = angle;

            bboxes.push_back(bbox);
            labels.push_back(std::distance(scores_ptr, max_s_ptr));
            scores.push_back(score);
        }
    }

    // 旋转框版本的 NMS
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);

    // 取分数最高的 topk 个填入 objs
    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        objs.push_back(obj);
        cnt += 1;
    }
}

/**
 * @brief 在原图副本 res 上：用 cv::boxPoints 计算旋转框 4 顶点 → polylines 画框；
 *        在中心点附近放置红底白字的 "类别 概率%" 文字。
 */
void YOLOv8_obb::draw_objects(const cv::Mat&                                image,
                              cv::Mat&                                      res,
                              const std::vector<Object>&                    objs,
                              const std::vector<std::string>&               CLASS_NAMES,
                              const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();
    for (auto& obj : objs) {
        // 旋转框 → 4 个顶点，转 int 画封闭多边形
        cv::Mat points;
        cv::boxPoints(obj.rect, points);
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        points.convertTo(points, CV_32S);
        cv::polylines(res, points, true, color, 2);

        // 文字内容："类别名 置信度%"
        char text[256];
        sprintf(
            text,
            "%s %.1f%%",
            CLASS_NAMES[obj.label].c_str(),
            obj.prob * 100
        );
        //sprintf(text, ""%s %.1f%%", CLASS_NAMES[].c_str(), obj.prob * 100);
        std::cout << obj.label;

        // 计算文字像素尺寸，画底色矩形时贴合
        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.center.x;
        int y = (int)obj.rect.center.y + 1;

        // 文字超出图像底边时回拉到底部
        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);
        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}

#endif  // OBB_NORMAL_YOLOv8_obb_HPP
