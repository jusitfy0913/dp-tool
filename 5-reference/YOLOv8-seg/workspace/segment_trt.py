from ctypes import *
import cv2
import numpy as np
import numpy.ctypeslib as npct
import platform
import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Union
import random


random.seed(42)

# detection model classes
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# colors for per classes
COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}


# image suffixs
SUFFIXS = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff',
           '.gif', '.webp', '.pfm')
           
VIDEO_SUFFIXS = ('.mp4', '.mkv', '.flv', '.avi', '.mov', '.wmv')
           
def path_to_list(images_path: Union[str, Path]) -> List:
    if isinstance(images_path, str):
        images_path = Path(images_path)
    assert images_path.exists()
    if images_path.is_dir():
        images = [
            i.absolute() for i in images_path.iterdir() if i.suffix in SUFFIXS
        ]
    else:
        assert images_path.suffix in SUFFIXS
        images = [images_path.absolute()]
    return images
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args

class InstanceSegmentMap:
    def __init__(self, width, height, data=None):
        self.width = width
        self.height = height
        if data is None:
            self.data = np.zeros((height, width), dtype=np.uint8)
        else:
            self.data = data

class Box:
    def __init__(self, left=0.0, top=0.0, right=0.0, bottom=0.0, confidence=0.0, class_label=0):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.confidence = confidence
        self.class_label = class_label

class Detector():
    def __init__(self, model_path, dll_path):
        if platform.system().lower() == 'windows':
            self.yolov8 = CDLL(dll_path, winmode=0)
        else:
            self.yolov8 = cdll.LoadLibrary(dll_path)

        self.yolov8.Detect.argtypes = [
            c_void_p, 
            c_int, 
            c_int, 
            POINTER(c_ubyte), 
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, shape=(100, 6), flags="C_CONTIGUOUS"),
            POINTER(c_int),  # widths
            POINTER(c_int),  # heights
            POINTER(POINTER(c_ubyte))  # datas
        ]

        self.yolov8.Init.restype = c_void_p
        self.yolov8.Init.argtypes = [c_void_p]
        self.c_point = self.yolov8.Init(model_path)

    def convert_seg_data(self, bbox_array, widths, heights, datas):
        seg_maps = []
        for i, bbox in enumerate(bbox_array):
            width = widths[i]
            height = heights[i]
            data_pointer = datas[i]

            #print(f"Width: {width}, Height: {height}")  # 调试输出

            # 检查 width 和 height 是否为合理的正数
            if width > 0 and height > 0 and data_pointer:
                # 创建一个 NumPy 数组，使用 C++ 端分配的内存
                data = np.ctypeslib.as_array(data_pointer, shape=(height, width))
                seg_maps.append(InstanceSegmentMap(width, height, data))
            else:
                pass
                #print(f"Invalid dimensions or null data pointer: Width: {width}, Height: {height}")
        return seg_maps


    def predict(self, img):
        rows, cols = img.shape[0], img.shape[1]
        res_arr = np.zeros((100, 6), dtype=np.float32)

        # 创建接收宽度和高度的数组
        widths = (c_int * 100)()
        heights = (c_int * 100)()

        # 创建接收数据指针的数组
        datas = (POINTER(c_ubyte) * 100)()

        # 调用 Detect 函数
        self.yolov8.Detect(self.c_point, rows, cols, img.ctypes.data_as(POINTER(c_ubyte)), res_arr, widths, heights, datas)

        self.bbox_array = res_arr[~(res_arr == 0).all(1)]
        self.seg_maps = self.convert_seg_data(res_arr, widths, heights, datas)
    
        return self.bbox_array, self.seg_maps


def plot_mask(image, mask, alpha=0.5, color=(0, 255, 0)):
    # 检查图像尺寸是否有效
    if image.shape[1] == 0 or image.shape[0] == 0:
        print("图像尺寸中有一个维度为0，跳过掩膜绘制。")
        return image  # 如果图像的宽度或高度为0，直接返回原图
    
    # Ensure image and mask are float32
    image = image.astype(np.float32)
    mask = mask.astype(np.float32)

    # Create color image to match input image dimensions
    color_img = np.zeros_like(image, dtype=np.float32)
    color_img[:] = color

    # Compute weighted mask considering alpha
    weighted_mask = (mask / 255.0) * alpha

    # Ensure weighted_mask has three dimensions
    if len(weighted_mask.shape) == 2:
        weighted_mask = weighted_mask[:, :, np.newaxis]

    # Match the number of channels with the image
    if image.shape[2] != weighted_mask.shape[2]:
        weighted_mask = np.repeat(weighted_mask, image.shape[2], axis=2)

    # Adjust the mask size if it doesn't match the image size exactly
    if weighted_mask.shape[:2] != image.shape[:2]:
        #print(f"Adjusting mask size from {weighted_mask.shape} to {image.shape}")
        weighted_mask = cv2.resize(weighted_mask, (image.shape[1], image.shape[0]))


    # Debug: Print shapes for troubleshooting
    #print(f"Image shape: {image.shape}, Weighted mask shape: {weighted_mask.shape}, Color image shape: {color_img.shape}")

    # Apply weighted mask to the image
    weighted_image = cv2.multiply(image, 1.0 - weighted_mask)
    weighted_color_mask = cv2.multiply(color_img, weighted_mask)

    # Combine weighted images for final result
    final_image = cv2.add(weighted_image, weighted_color_mask)

    return final_image.astype(np.uint8)


def visualize(img, bbox_array, mask_array):
    assert len(bbox_array) == len(mask_array), "Length of bbox_array and mask_array must be the same"

    for bbox_data, mask_obj in zip(bbox_array, mask_array):
        x, y, w, h = bbox_data[0:4]  # Unpack xywh
        cls_id = int(bbox_data[4])
        score = bbox_data[5]
        cls = CLASSES[cls_id]     
        color = COLORS[cls]      

        # Draw bounding box and text
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        label = f"{cls} {score:.2f}"
        cv2.putText(img, label, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        if mask_obj is not None:
            # Extract mask data
            mask_data = mask_obj if isinstance(mask_obj, np.ndarray) else mask_obj.data
            mask_data = mask_data.astype(np.uint8)

            # Resize mask to match the bounding box
            mask_resized = cv2.resize(mask_data, (int(w), int(h)))

            # Directly apply the mask on the image within the bounding box
            # No need to create a full-size mask image
            x_end = min(int(x) + int(w), img.shape[1])
            y_end = min(int(y) + int(h), img.shape[0])


            # Ensure the region to apply mask is within image bounds
            img_region = img[int(y):y_end, int(x):x_end]
            masked_region = plot_mask(img_region.copy(), mask_resized, alpha=0.5, color=color)
            img[int(y):y_end, int(x):x_end] = masked_region

    
    return img

def infer_video(input_file_path, det, save_path):

    # Open the video file
    cap = cv2.VideoCapture(input_file_path)

    if not cap.isOpened():
        print(f"Cannot open {input_file_path}")
        exit(-1)

    # Get the frame size from the video input
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameSize = (frameWidth, frameHeight)

    # Assuming input_file_path is a string with the file path
    input_file_path_obj = Path(input_file_path)

    # Extract the file stem (name without extension)
    file_stem = input_file_path_obj.stem

    # Now you can create the output video path
    outputVideoPath = str(save_path / file_stem) + ".mp4"

    # Example for MP4V codec
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # Get the fps from the video input
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create the VideoWriter object
    videoWriter = cv2.VideoWriter(outputVideoPath, codec, fps, frameSize)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        result, seg_mask = det.predict(frame)
        end = time.time()
        frameprocess_time = (end - start) * 1000  
        print(f"frameprocess_time: {frameprocess_time:.2f} ms")
        
        img = visualize(frame, result, seg_mask)
        videoWriter.write(img)

    # Release resources
    cap.release()
    videoWriter.release()
    print('input->{} saved'.format(outputVideoPath))
    
def main(args: argparse.Namespace) -> None:
    model_path = args.engine.encode()  # Convert model_path to byte string
    input_file_path = args.imgs

    print("Model Path:", model_path)
    print("Images Path:", input_file_path)

    if platform.system().lower() == 'windows':
        DLL_PATH =  "./yolov8.dll"
    else:
        DLL_PATH =  "./libyolov8.so"

    # 创建 Detector 实例，传递 model_path 作为字节字符串
    det = Detector(model_path=model_path, dll_path=DLL_PATH)
    
    isFolder = False
    isImage = False
    isVideo = False
    if os.path.isdir(input_file_path):
        isFolder = True  # 是一个目录
    elif os.path.isfile(input_file_path):
        file_extension = os.path.splitext(input_file_path.lower())[1]
        if file_extension in SUFFIXS:
            isImage = True  # 是一个图像文件
        elif file_extension in VIDEO_SUFFIXS:
            isVideo = True  # 是一个视频文件
        else:
            print("It's a file, but not an image or video file.")
    else:
        print("The path is neither a directory nor a file.")

    save_path = Path(args.out_dir)
    
    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    if isImage or isFolder:
        images = path_to_list(input_file_path) 

    if isVideo:
        infer_video(input_file_path, det, save_path)
        exit()

    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Unable to load image at {image_path}")
            continue

        start = time.time()
        #print(f"start to predict ...")
        result, seg_mask = det.predict(img)
        #print(f"end predict ...")
        end = time.time()
        inference_time = (end - start) * 1000  
        print(f"Inference time for {image_path}: {inference_time:.2f} ms")
        
        img = visualize(img, result, seg_mask)
        if args.show:
            cv2.imshow(str(image_path.name), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        save_image_path = save_path / image_path.name
        cv2.imwrite(str(save_image_path), img)
        print(f"Saved image at: {save_image_path}")  # 打印保存图像的路径

if __name__ == "__main__":
    args = parse_args()
    main(args)
    # det = Detector(model_path=b"D:\YOLOv8-seg\workspace\best.engine",dll_path="D:\YOLOv8-seg\workspace\yolov8.dll")  # b'' is needed
    # img = cv2.imread("./zidane.jpg")


