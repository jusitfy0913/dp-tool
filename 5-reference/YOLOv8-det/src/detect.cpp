/**
 * @file    detect.cpp
 * @brief   YOLOv8 ???????????? C ????????????? ctypes ??????? DLL ??????
 *
 * ???????
 *  - main        : ???? engine?????????????????/??/???????????????????????????? output/??
 *  - Init        : ???? yolo::Infer* ?????loadraw??Type::V8????
 *  - Detect      : ?? BGR ??????????????????? res_array[i][0..5]??xywh+???+????????
 *
 * ?????main ?? engine ?? path ???????????????????????? assert(argc==3) ???????? argv ?????
 *       ?????????????????????????????????
 */
#include <opencv2/opencv.hpp>

#include "yolo.hpp"

#include <iostream>
#include <filesystem>
#include "chrono"
#include <windows.h>
#include <thread>
#include <sys/stat.h>
#include <cstring>
using namespace cv;
using namespace std;

/// COCO 80 ???????C ??????????????? CLASS_NAMES ????????????? C ?????????
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

/// ?? OpenCV BGR Mat ?? yolo::Image??????????cols/rows????
yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }


/// Windows ???? GetFileAttributes ??????????????????????????????????
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

/// ??????????????????????
inline bool IsFolder(const std::string& path)
{
    //struct stat path_stat;
    //if (stat(path.c_str(), &path_stat) != 0)
    //{
    //    return false; // Error occurred or the file/directory does not exist
    //}
    //return S_ISDIR(path_stat.st_mode);
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
 * @brief ?? res ?????????????????????????????????? COLORS[class_label]??
 * @note  ??? standard ??? 0?????? score>standard ????????????????????????
 */
void draw_objects(const cv::Mat& image,
                  cv::Mat& res,
                  yolo::BoxArray& objs,
                  const std::vector<std::string>& CLASS_NAMES,
                  const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();
    double standard = 0.0;
    for (auto& obj : objs)
    {
        int idx = obj.class_label;
        cv::Scalar color = cv::Scalar(
            COLORS[idx][0],
            COLORS[idx][1],
            COLORS[idx][2]
        );
        double score = obj.confidence;
        cv::Rect rect(static_cast<int>(obj.left), static_cast<int>(obj.top),
            static_cast<int>(obj.right - obj.left), static_cast<int>(obj.bottom - obj.top));
        if (score> standard)
        {
           

            cv::rectangle(
                res,
                rect,
                color,
                2
            );
        }
        

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

        
        if (score > standard)
        {
            if (y > res.rows)
                y = res.rows;

            cv::rectangle(
                res,
                cv::Rect(x, y, label_size.width, label_size.height + baseLine),
                color,
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

        }


    }
}

/**
 * @brief ??????????????? YOLOv8 TensorRT engine???????????????????????????
 * @return ???????? 0??????????? 1???????????? -1
 */
int main(int argc, char** argv)
{

  std::cout << "running main..." << std::endl;
std::string engine_file_path = "D:\\YOLOv8-det\\workspace\\yolov8ndetection.engine";
  //std::string engine_file_path = "D:\\YOLOv8-det\\workspace\\yolov8ndetection.engine";
  std::string path = "D:\\YOLOv8-seg\\workspace\\images\\zidane.jpg";
  const char* engine_file_pathchar = engine_file_path.c_str();
  //const std::string engine_file_path{ argv[1] };
  //const std::string path{ argv[2] };

  std::vector<std::string> imagePathList;
  bool isVideo{ false };

  assert(argc == 3);
  //????1engine?????????????? ????????nms?????????
  auto yolo = yolo::load(engine_file_path, yolo::Type::V8,0.6f,0.5f);
  //???c?????????????
  //auto yoloC = Init(engine_file_pathchar);

  if (yolo == nullptr) {
      std::cout << "Failed to load yolo model." << std::endl;
      return 1;
  }

	if (IsFile(path))
	{
		std::string suffix = path.substr(path.find_last_of('.') + 1);
		if (
			suffix == "jpg" ||
				suffix == "jpeg" ||
				suffix == "png"
			)
		{
			imagePathList.push_back(path);
		}
		else if (
			suffix == "mp4" ||
				suffix == "avi" ||
				suffix == "m4v" ||
				suffix == "mpeg" ||
				suffix == "mov" ||
				suffix == "mkv"
			)
		{
			isVideo = true;
		}
		else
		{
			printf("suffix %s is wrong !!!\n", suffix.c_str());
			std::abort();
		}
	}
	else if (IsFolder(path))
	{
		cv::glob(path + "/*.jpg", imagePathList);
	}

  cv::Mat res, image;
  //cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
  if (isVideo)
  {
		cv::VideoCapture cap(0);

		if (!cap.isOpened())
		{
			printf("can not open %s\n", path.c_str());
			return -1;
		}
        cv::Mat frame;
        while (true) {
            // ????????????
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "???????" << std::endl;
                break;
            }
            auto start = std::chrono::system_clock::now();
            auto objs = yolo->forward(cvimg(frame));
            auto end = std::chrono::system_clock::now();
            auto tc = (double)
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("inference %2.4lf ms\n", tc);
            auto startdraw = std::chrono::system_clock::now();
            draw_objects(frame, res, objs, CLASS_NAMES, COLORS);
            auto enddraw = std::chrono::system_clock::now();
            // ????????
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(enddraw - startdraw).count() / 1000.;
            auto td = (double)
            std::chrono::duration_cast<std::chrono::microseconds>(enddraw - startdraw).count() / 1000.;
            printf("Postprocessing %2.4lf ms\n", td);
            // ????
            cv::imshow("Camera", res);

            // ???? 'q' ?????
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
        // ???????????
        cap.release();
        cv::destroyAllWindows();
        //?????????????

 //   // Get the frame size from the video input
 //   int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
 //   int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
 //   cv::Size frameSize(frameWidth, frameHeight);

 //   std::string outputDirectory = "output/";
 //   std::size_t fileNameStart = path.find_last_of("/\\") + 1;
 //   std::size_t dotPos = path.find_last_of(".");
 //   std::string outputVideoPath = outputDirectory + path.substr(fileNameStart, dotPos - fileNameStart) + "_det.mp4";

 //   // Example for MP4V codec
 //   int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

 //   // Get the fps from the video input
 //   double fps = cap.get(cv::CAP_PROP_FPS);

 //   // Create the VideoWriter object
 //   cv::VideoWriter videoWriter(outputVideoPath, codec, fps, frameSize);

	//while (cap.read(image))
	// {
 //       auto start = std::chrono::system_clock::now();
 //       auto objs = yolo->forward(cvimg(image));
 //       auto end = std::chrono::system_clock::now();
 //       auto tc = (double)
 //       std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
 //       printf("inference %2.4lf ms\n", tc);
 //       draw_objects(image, res, objs, CLASS_NAMES, COLORS);
 //       //imshow("img", res);
 //       videoWriter.write(res);
	//	}
	}
	else // not video
	{
		for (auto& path : imagePathList)
		{

      image = cv::imread(path);

      auto start = std::chrono::system_clock::now();
      auto objs = yolo->forward(cvimg(image));
      int length = objs.size();
      printf("length %2.4lf \n", length);
      auto end = std::chrono::system_clock::now();
      auto tc = (double)
				std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
			printf("inference %2.4lf ms\n", tc);
      draw_objects(image, res, objs, CLASS_NAMES, COLORS);

      std::string outputDirectory = "output/";
      std::size_t fileNameStart = path.find_last_of("/\\") + 1;
      std::size_t dotPos = path.find_last_of(".");
      std::string outputImagePath = outputDirectory + path.substr(fileNameStart, dotPos - fileNameStart) + "_det" + path.substr(dotPos);
      imshow("img", res);
      waitKey(0);
      cv::imwrite(outputImagePath, res);

		}
	}
	cv::destroyAllWindows();
	return 0;
}



/**
 * @brief C ???????????��???? yolo::Infer ???
 * @note ???????????��??? 1.0f??????????????? Detect ???????????? load ???????????????
 */
extern "C" void* Init(const char* model_path) {
    std::string engine_name = model_path;
    float confidence_threshold = 1.0f;
    float nms_threshold = 0.5f;
    auto yolo = loadraw(engine_name, yolo::Type::V8, confidence_threshold, nms_threshold);
    return yolo;
}

/**
 * @brief C �ӿڣ��� CV_8UC3 ͼ�����������д���� objs ������ͬ���е� res_array��
 * @param p         Init ���ص� yolo::Infer*
 * @param res_array ÿ�� [left, top, w, h, class_id, confidence]
 */
extern "C" void Detect(void* p, int rows, int cols, unsigned char* src_data, float(*res_array)[6]) {
    cv::Mat frame = cv::Mat(rows, cols, CV_8UC3, src_data);
    yolo::Infer* yolov8 = (yolo::Infer*)p;
    auto objs = yolov8->forward(cvimg(frame));
    int i = 0;
    for (auto &obj : objs) {
        float left = obj.left;
        printf("obj.left %2.4lf \n", left) ;
        res_array[i][0] = obj.left;
        float top = obj.top;
        printf("obj.top %2.4lf \n", top);
        res_array[i][1] = obj.top;
        res_array[i][2] = obj.right - obj.left;
        res_array[i][3] = obj.bottom - obj.top;;
        res_array[i][4] = obj.class_label;
        res_array[i][5] = obj.confidence;
        i++;
    }
}


