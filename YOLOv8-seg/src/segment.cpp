/**
 * @file    segment.cpp
 * @brief   YOLOv8 实例分割（V8Seg）演示：TensorRT 推理、框+掩码可视化、视频/批量图导出，以及 DLL 导出 C 接口。
 *
 * 流程概要：
 *  - main：load(engine, Type::V8Seg)，按路径分流单图/目录/视频；draw_objects 将检测框与按实例着色的半透明掩码叠到结果上；
 *  - Init / Detect：导出符号（__declspec(dllexport)），返回句柄并输出每目标的 xywh、类别、置信度及掩码宽高与 data 指针（内存由推理对象持有，调用方需在句柄存活期内使用）。
 *
 * engine 与输入路径在源码中硬编码，发布前请改为命令行参数或配置。
 */
#include <opencv2/opencv.hpp>

#include "yolo.hpp"

#include <iostream>
#include <filesystem>
#include "chrono"
#include <windows.h>


#include <sys/stat.h>
#include <cstring>
using namespace cv;
using namespace std;
#include <windows.h>
/// COCO 80 类名（C 字符串表，与 CLASS_NAMES 对应）。
static const char *cocolabels[] = {"person",        "bicycle",      "car",
                                   "motorcycle",    "airplane",     "bus",
                                   "train",         "truck",        "boat",
                                   "traffic light", "fire hydrant", "stop sign",
                                   "parking meter", "bench",        "bird",
                                   "cat",           "dog",          "horse",
                                   "sheep",         "cow",          "elephant",
                                   "bear",          "zebra",        "giraffe",
                                   "backpack",      "umbrella",     "handbag",
                                   "tie",           "suitcase",     "frisbee",
                                   "skis",          "snowboard",    "sports ball",
                                   "kite",          "baseball bat", "baseball glove",
                                   "skateboard",    "surfboard",    "tennis racket",
                                   "bottle",        "wine glass",   "cup",
                                   "fork",          "knife",        "spoon",
                                   "bowl",          "banana",       "apple",
                                   "sandwich",      "orange",       "broccoli",
                                   "carrot",        "hot dog",      "pizza",
                                   "donut",         "cake",         "chair",
                                   "couch",         "potted plant", "bed",
                                   "dining table",  "toilet",       "tv",
                                   "laptop",        "mouse",        "remote",
                                   "keyboard",      "cell phone",   "microwave",
                                   "oven",          "toaster",      "sink",
                                   "refrigerator",  "book",         "clock",
                                   "vase",          "scissors",     "teddy bear",
                                   "hair drier",    "toothbrush"};

const std::vector<std::string> CLASS_NAMES = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus",
	"train", "truck", "boat", "traffic light", "fire hydrant",
	"stop sign", "parking meter", "bench", "bird", "cat",
	"dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella",
	"handbag", "tie", "suitcase", "frisbee", "skis",
	"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
	"cup", "fork", "knife", "spoon", "bowl",
	"banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table",
	"toilet", "tv", "laptop", "mouse", "remote",
	"keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush" };

const std::vector<std::vector<unsigned int>> COLORS = {
	{ 0, 114, 189 }, { 217, 83, 25 }, { 237, 177, 32 },
	{ 126, 47, 142 }, { 119, 172, 48 }, { 77, 190, 238 },
	{ 162, 20, 47 }, { 76, 76, 76 }, { 153, 153, 153 },
	{ 255, 0, 0 }, { 255, 128, 0 }, { 191, 191, 0 },
	{ 0, 255, 0 }, { 0, 0, 255 }, { 170, 0, 255 },
	{ 85, 85, 0 }, { 85, 170, 0 }, { 85, 255, 0 },
	{ 170, 85, 0 }, { 170, 170, 0 }, { 170, 255, 0 },
	{ 255, 85, 0 }, { 255, 170, 0 }, { 255, 255, 0 },
	{ 0, 85, 128 }, { 0, 170, 128 }, { 0, 255, 128 },
	{ 85, 0, 128 }, { 85, 85, 128 }, { 85, 170, 128 },
	{ 85, 255, 128 }, { 170, 0, 128 }, { 170, 85, 128 },
	{ 170, 170, 128 }, { 170, 255, 128 }, { 255, 0, 128 },
	{ 255, 85, 128 }, { 255, 170, 128 }, { 255, 255, 128 },
	{ 0, 85, 255 }, { 0, 170, 255 }, { 0, 255, 255 },
	{ 85, 0, 255 }, { 85, 85, 255 }, { 85, 170, 255 },
	{ 85, 255, 255 }, { 170, 0, 255 }, { 170, 85, 255 },
	{ 170, 170, 255 }, { 170, 255, 255 }, { 255, 0, 255 },
	{ 255, 85, 255 }, { 255, 170, 255 }, { 85, 0, 0 },
	{ 128, 0, 0 }, { 170, 0, 0 }, { 212, 0, 0 },
	{ 255, 0, 0 }, { 0, 43, 0 }, { 0, 85, 0 },
	{ 0, 128, 0 }, { 0, 170, 0 }, { 0, 212, 0 },
	{ 0, 255, 0 }, { 0, 0, 43 }, { 0, 0, 85 },
	{ 0, 0, 128 }, { 0, 0, 170 }, { 0, 0, 212 },
	{ 0, 0, 255 }, { 0, 0, 0 }, { 36, 36, 36 },
	{ 73, 73, 73 }, { 109, 109, 109 }, { 146, 146, 146 },
	{ 182, 182, 182 }, { 219, 219, 219 }, { 0, 114, 189 },
	{ 80, 183, 189 }, { 128, 128, 0 }
};

/// 掩码叠加用调色板（20 色循环，与类别 id 取模配合 draw_objects）。
const std::vector<std::vector<unsigned int>> MASK_COLORS = {
	{ 255, 56, 56 }, { 255, 157, 151 }, { 255, 112, 31 },
	{ 255, 178, 29 }, { 207, 210, 49 }, { 72, 249, 10 },
	{ 146, 204, 23 }, { 61, 219, 134 }, { 26, 147, 52 },
	{ 0, 212, 187 }, { 44, 153, 168 }, { 0, 194, 255 },
	{ 52, 69, 147 }, { 100, 115, 255 }, { 0, 24, 236 },
	{ 132, 56, 255 }, { 82, 0, 133 }, { 203, 56, 255 },
	{ 255, 149, 200 }, { 255, 55, 199 }
};

/// OpenCV BGR Mat → yolo::Image。
yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

/// Windows：路径存在且为文件（非目录）。
inline bool IsFile(const std::string& path)
{
    DWORD attributes = GetFileAttributesA(path.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES)
    {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }
    if ((attributes & FILE_ATTRIBUTE_DIRECTORY) != 0)
    {
        printf("%s:%d %s is a directory\n", __FILE__, __LINE__, path.c_str());
        return false;
    }
    return true;
}

/// Windows：路径为已存在目录。
inline bool IsFolder(const std::string& path)
{
    DWORD attributes = GetFileAttributesA(path.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES)
    {
        return false;
    }
    if ((attributes & FILE_ATTRIBUTE_DIRECTORY) == 0)
    {
        return false;
    }
    return true;
}

/**
 * @brief 将单通道 mask（0–255）按 alpha 与 color 混合到 BGR 图像上（逐像素，mask 值越大越接近 color）。
 * @pre image 与 mask 同尺寸，image 为 CV_8UC3，mask 为 CV_8U。
 */
void plot_mask(cv::Mat image, cv::Mat mask, float alpha = 0.5, cv::Scalar color = cv::Scalar(0, 255, 0))
{
    CV_Assert(image.type() == CV_8UC3 && mask.type() == CV_8U && image.size() == mask.size());

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            //if (mask.at<uchar>(y, x))
            float beta = alpha*mask.at<uchar>(y, x)/255.0;
            {
                cv::Vec3b& pixel = image.at<cv::Vec3b>(y, x);
                pixel[0] = static_cast<uchar>(pixel[0] * (1 - beta) + color[0] * beta);
                pixel[1] = static_cast<uchar>(pixel[1] * (1 - beta) + color[1] * beta);
                pixel[2] = static_cast<uchar>(pixel[2] * (1 - beta) + color[2] * beta);
            }
        }
    }
}

/**
 * @brief 绘制检测框与标签；若 Box::seg 非空，将实例掩码 resize 到框内 ROI 并全图铺零后调用 plot_mask 叠加。
 * @param MASK_COLORS 长度 20 的调色板，下标为 class_label % 20。
 */
void draw_objects(const cv::Mat& image,
                  cv::Mat& res,
                  yolo::BoxArray& objs,
                  const std::vector<std::string>& CLASS_NAMES,
                  const std::vector<std::vector<unsigned int>>& COLORS,
                  const std::vector<std::vector<unsigned int>>& MASK_COLORS)
{
    res = image.clone();

    for (auto& obj : objs)
    {
        int idx = obj.class_label;
        cv::Scalar color = cv::Scalar(
            COLORS[idx][0],
            COLORS[idx][1],
            COLORS[idx][2]
        );
        cv::Scalar mask_color = cv::Scalar(
            MASK_COLORS[idx % 20][0],
            MASK_COLORS[idx % 20][1],
            MASK_COLORS[idx % 20][2]
        );

        cv::Rect rect(static_cast<int>(obj.left), static_cast<int>(obj.top),
                      static_cast<int>(obj.right - obj.left), static_cast<int>(obj.bottom - obj.top));

        cv::rectangle(
            res,
            rect,
            color,
            2
        );

        char text[256];
        sprintf(
            text,
            "%s %.1f%%",
            CLASS_NAMES[idx].c_str(),
            obj.confidence * 100
        );

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(
            text,
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            1,
            &baseLine
        );

        int x = rect.x;
        int y = rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(
            res,
            cv::Rect(x, y, label_size.width, label_size.height + baseLine),
            color, //{ 0, 0, 255 },
            -1
        );

        cv::putText(
            res,
            text,
            cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            { 255, 255, 255 },
            1
        );
        
        if (obj.seg)
        {
          // 调整 rect 以确保它不超出 image 的边界
          rect.x = (std::max)(rect.x, 0);
          rect.y = (std::max)(rect.y, 0);
          rect.width = (std::min)(rect.width, image.cols - rect.x);
          rect.height = (std::min)(rect.height, image.rows - rect.y);

          // 创建 segMask，代表一个单通道图像，使用了 obj.seg 中的数据
          cv::Mat segMask(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data);

          // 创建一个与原始图像相同大小的全黑（全零）掩膜
          cv::Mat mask(image.size(), CV_8U, cv::Scalar(0));

          // 调整 segMask 的大小以匹配调整后的 rect 的尺寸
          cv::Mat resizedMask;
          cv::resize(segMask, resizedMask, rect.size());

          // 在 mask 中提取由调整后的 rect 指定的区域
          cv::Mat roiMask = mask(rect);

          // 将调整大小后的 segMask 复制到 roiMask 中
          resizedMask.copyTo(roiMask);

          // 应用 plot_mask 函数，是将 mask 应用到 res 图像上的自定义函数
          plot_mask(res, mask, 0.5, mask_color);
    }
    }
}

/**
 * @brief 程序入口：加载 V8Seg engine，处理图像列表或导出带掩码的视频/图片到 workspace/output。
 * @return 模型加载失败返回 1；视频无法打开返回 -1；成功 0。
 */
int main(int argc, char** argv)
{
  //std::cout << "running main..." << std::endl;

  //if (argc != 3) {
  //  std::cout << "Usage: " << argv[0] << " <engine_file_path> <path_to_image_or_video>" << std::endl;
  //  return 1;
  //}

  std::string engine_file_path="D:\\YOLOv8-seg\\workspace\\model\\best.engine";
  std::string path="D:\\YOLOv8-seg\\workspace\\images\\zidane.jpg";

  std::vector<std::string> imagePathList;
  bool isVideo{ false };

  auto yolo = yolo::load(engine_file_path, yolo::Type::V8Seg);
  if (yolo == nullptr) {
    std::cout << "Failed to load YOLO model." << std::endl;
    return 1;
  }

  if (IsFile(path))
  {
    std::string suffix = path.substr(path.find_last_of('.') + 1);
    if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
    {
      imagePathList.push_back(path);
    }
    else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv")
    {
      isVideo = true;
    }
    else
    {
      printf("Suffix %s is not supported.\n", suffix.c_str());
      std::abort();
    }
  }
  else if (IsFolder(path))
  {
    cv::glob(path + "/*.jpg", imagePathList);
  }

  cv::Mat res, image;
  cv::Mat mask;

  if (isVideo)
  {
    cv::VideoCapture cap(path);

    if (!cap.isOpened())
    {
      printf("Cannot open %s\n", path.c_str());
      return -1;
    }

    // Get the frame size from the video input
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::Size frameSize(frameWidth, frameHeight);

    // Define the output video file
    
    std::string outputDirectory = "D:/YOLOv8-seg/workspace/output/";
    std::size_t fileNameStart = path.find_last_of("/\\") + 1;
    std::size_t dotPos = path.find_last_of(".");
    //std::string outputVideoPath = path.substr(0, dotPos) + "_seg.mp4";
    std::string outputVideoPath = outputDirectory + path.substr(fileNameStart, dotPos - fileNameStart) + "_seg.mp4";

    // Example for MP4V codec
    int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

    // Get the fps from the video input
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Create the VideoWriter object
    cv::VideoWriter videoWriter(outputVideoPath, codec, fps, frameSize);

    while (cap.read(image))
    {
      auto start = std::chrono::system_clock::now();
      auto objs = yolo->forward(cvimg(image));
      auto end = std::chrono::system_clock::now();
      auto tc = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
      printf("Inference: %.4lf ms\n", tc);
      draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);
      videoWriter.write(res);
    }
  }
  else
  {
    for (const auto& path : imagePathList)
    {
      image = cv::imread(path);

      auto start = std::chrono::system_clock::now();
      auto objs = yolo->forward(cvimg(image));
      auto end = std::chrono::system_clock::now();
      auto tc = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
      printf("Inference: %.4lf ms\n", tc);
      draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);
    
      std::string outputDirectory = "D:/YOLOv8-seg/workspace/output/";
      std::size_t fileNameStart = path.find_last_of("/\\") + 1;
      std::size_t dotPos = path.find_last_of(".");
      //std::string outputImagePath = path.substr(0, dotPos) + "_seg" + path.substr(dotPos);
      std::string outputImagePath = outputDirectory + path.substr(fileNameStart, dotPos - fileNameStart) + "_seg" + path.substr(dotPos);
      imshow("img", res);
      waitKey(0);
      cv::imwrite(outputImagePath, res);
    }
  }

  cv::destroyAllWindows();
  return 0;
}

/**
 * @brief DLL 导出：加载分割引擎，返回 yolo::Infer*（loadraw，Type::V8Seg，默认置信度/NMS 阈值）。
 */
extern "C" __declspec(dllexport) void* Init(const char* model_path) {
	std::string engine_name = model_path;
	float confidence_threshold = 0.25f;
	float nms_threshold = 0.5f;
	auto yolo = loadraw(engine_name, yolo::Type::V8Seg, confidence_threshold, nms_threshold);
	return yolo;
}

/**
 * @brief DLL 导出：对 BGR 图像推理，写入每目标 6 元组与掩码描述。
 * @param res_array 每行 [left, top, w, h, class_id, confidence]
 * @param widths/heights 各实例掩码尺寸（无掩码时为 0）
 * @param datas        指向锁页掩码数据的指针（与 yolo::InstanceSegmentMap 生命周期绑定，不转移所有权）
 */
extern "C" __declspec(dllexport) void Detect(
    void* p, 
    int rows, 
    int cols, 
    unsigned char* src_data, 
    float(*res_array)[6], 
    int* widths,  // 数组，存储每个目标的宽度
    int* heights, // 数组，存储每个目标的高度
    unsigned char** datas // 数组，存储指向每个目标分割图数据的指针
) {
    cv::Mat frame = cv::Mat(rows, cols, CV_8UC3, src_data);
    yolo::Infer* yolov8 = (yolo::Infer*)p;
    auto objs = yolov8->forward(cvimg(frame));

    int i = 0;
    for (auto &obj : objs) {
        res_array[i][0] = obj.left;
        res_array[i][1] = obj.top;
        res_array[i][2] = obj.right - obj.left;
        res_array[i][3] = obj.bottom - obj.top;
        res_array[i][4] = obj.class_label;
        res_array[i][5] = obj.confidence;

        if (obj.seg != nullptr) {
            widths[i] = obj.seg->width;
            heights[i] = obj.seg->height;
            datas[i] = obj.seg->data;
            // 注意：这里不转移所有权，调用方需要确保 obj.seg 指向的内存在使用期间保持有效
        } else {
            widths[i] = 0;
            heights[i] = 0;
            datas[i] = nullptr;
        }

        i++;
    }
}
