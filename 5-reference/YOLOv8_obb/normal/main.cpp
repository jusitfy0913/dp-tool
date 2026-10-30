//
// Created by ubuntu on 4/7/23.
//
/**
 * @file    main.cpp
 * @brief   YOLOv8 OBB（旋转框检测）工程的可执行入口（Windows 版本）。
 *
 * 当前 main 中将 engine 与图像路径写死，便于在 IDE 中直接 F5 调试；
 * 注释中保留了原本基于 argc/argv 的命令行用法，可按需放开。
 *
 * 流程：
 *   1) 加载 OBB engine 并调用 make_pipe(true) 完成显存分配 + 预热；
 *   2) 根据后缀判断输入是单张图、目录还是视频；
 *   3) 对每张图/每帧：copy_from_Mat → infer → postprocess → draw_objects；
 *   4) OpenCV 显示并打印每次推理耗时。
 *
 * 备注：CLASS_NAMES 与 COLORS 为 DOTA 数据集常用的 15 类标签与配色。
 */
#include "opencv2/opencv.hpp"
#include "yolov8-obb.hpp"
#include <chrono>
#include <chrono>
#include <iostream>
#include <filesystem>
#include "chrono"
#include <sys/stat.h>
#include <cstring>

/**
 * @brief 判断给定路径是否是一个存在的、且不是目录的文件（Windows 实现）。
 * @return 文件存在且非目录返回 true；不存在或为目录则打印日志并返回 false。
 */
inline bool IsFile(const std::string& path)
{
    //struct stat path_stat;
    //if (stat(path.c_str(), &path_stat) != 0)
    //{
    //    std::cerr << __FILE__ << ":" << __LINE__ << " " << path << " does not exist or error occurred" << std::endl;
    //    return false;
    //}
    //if (S_ISDIR(path_stat.st_mode))
    //{
    //    std::cerr << __FILE__ << ":" << __LINE__ << " " << path << " is a directory" << std::endl;
    //    return false;
    //}
    //return true;
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
//namespace fs = ghc::filesystem;

/// DOTA 数据集 15 类标签（与 OBB engine 输出通道顺序一致）
const std::vector<std::string> CLASS_NAMES = {"plane",
                                              "ship",
                                              "storage tank",
                                              "baseball diamond",
                                              "tennis court",
                                              "basketball court",
                                              "ground track field",
                                              "harbor",
                                              "bridge",
                                              "large vehicle",
                                              "small vehicle",
                                              "helicopter",
                                              "roundabout",
                                              "soccer ball field",
                                              "swimming pool"};

/// 与 CLASS_NAMES 一一对应的绘制颜色（B,G,R）
const std::vector<std::vector<unsigned int>> COLORS = {{0, 114, 189},
                                                       {217, 83, 25},
                                                       {237, 177, 32},
                                                       {126, 47, 142},
                                                       {119, 172, 48},
                                                       {77, 190, 238},
                                                       {162, 20, 47},
                                                       {76, 76, 76},
                                                       {153, 153, 153},
                                                       {255, 0, 0},
                                                       {255, 128, 0},
                                                       {191, 191, 0},
                                                       {0, 255, 0},
                                                       {0, 0, 255},
                                                       {170, 0, 255}};

/**
 * @brief 程序入口：加载 OBB engine，对配置好的图像/视频路径执行推理并显示结果。
 *
 * 当前 argc/argv 校验已注释掉，engine 与待推理路径在源码内硬编码，方便调试。
 */
int main(int argc, char** argv)
{
    //if (argc != 3) {
    //    fprintf(stderr, "Usage: %s [engine_path] [image_path/image_dir/video_path]\n", argv[0]);
    //    return -1;
    //}

    // cuda:0（如需多卡可放开并改为命令行参数）
    //cudaSetDevice(0);

    // 1) 引擎与待推理路径（写死，便于在 IDE 中直接调试）
    std::string engine_file_path = "D:\\YOLOv8_obb\\normal\\build\\Release\\yolov8sobb.engine";
    //std::string engine_file_path = "D:\\YOLOv8-det\\workspace\\yolov8ndetection.engine";
    std::string path = "D:\\YOLOv8_obb\\normal\\plane2.jpg";

    std::vector<std::string> imagePathList; ///< 待处理图像路径列表
    bool                     isVideo{false}; ///< 输入是否为视频

    //assert(argc == 3);

    // 2) 加载 OBB 模型并完成显存分配 + 预热
    auto yolov8_obb = new YOLOv8_obb(engine_file_path);
    yolov8_obb->make_pipe(true);

    // 3) 根据后缀分流：单张图 / 视频 / 目录
    if (IsFile(path)) {
        std::string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "peg" || suffix == "png") {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov"
                 || suffix == "mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (IsFile(path)) {
        // NOTE：此分支在原逻辑下不可能进入（前一个 if 已覆盖 IsFile=true）。
        //       若需支持目录批量推理，应改为对目录的判断。
        cv::glob(path + "/*.jpg", imagePathList);
    }

    cv::Mat  res, image;
    cv::Size size        = cv::Size{1024, 1024}; ///< OBB 默认输入分辨率
    int      num_labels  = 15;                   ///< DOTA 类别数
    int      topk        = 100;                  ///< 每图最多保留检测框数
    float    score_thres = 0.25f;                ///< 置信度阈值
    float    iou_thres   = 0.65f;                ///< NMS IOU 阈值

    std::vector<Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        // === 视频推理分支：逐帧处理 ===
        cv::VideoCapture cap(path);
        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        while (cap.read(image)) {
            objs.clear();
            yolov8_obb->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_obb->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_obb->postprocess(objs, score_thres, iou_thres, topk, num_labels);
            yolov8_obb->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            // 按 q 键提前退出循环
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
    }
    else {
        // === 图像/目录推理分支 ===
        for (auto& p : imagePathList) {
            objs.clear();
            image = cv::imread(p);
            yolov8_obb->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_obb->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_obb->postprocess(objs, score_thres, iou_thres, topk, num_labels);
            yolov8_obb->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0); // 任意按键查看下一张
        }
    }

    // 4) 清理窗口与模型对象
    cv::destroyAllWindows();
    delete yolov8_obb;
    return 0;
}
