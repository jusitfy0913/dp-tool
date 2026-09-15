# infer onnx model
import onnxruntime as ort
import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
def infer_onnx(onnx_path, img_path,img_name):
    ort_session = ort.InferenceSession(onnx_path)
    image = cv2.imread(img_path)    
    image_blod = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(256,256), swapRB=True, )
    # cv2.imwrite('image_blod',image_blod)
    # cv2.namedWindow("result")
    # cv2.imshow("result", image_blod)
    # cv2.waitKey(0)
    model_inputs = ort_session.get_inputs()
    model_outputs = ort_session.get_outputs()
    print('image_blod',image_blod)
    output = ort_session.run(None, {model_inputs[0].name: image_blod})
    # print('output',output)
    # np.savetxt('res.csv', output[0].squeeze(), delimiter=',', fmt='%f')
    # print(output[0])
    print('output',output)
    # print('output size ',output.shape())
    # 计算放缩比例    
    
    print('out_0:',output[0])
    # print('out_1:',output[1])
    # print('output_shape',output.shape)
    
    ratio_x = image.shape[1] / 224
    ratio_y = image.shape[0] / 224
    img_res = output[0].squeeze()
    # img_res = output[2].squeeze()
    print('out_res1:',img_res)
    cv2.imwrite('output/'+'netoutput.jpg'+img_name,img_res)
    img_res = img_res.astype(np.uint8)
    print('out_res2:',img_res)
    norm_img = np.zeros(img_res.shape)
    heatmapshow  = cv2.normalize(img_res, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    print('norm_img:',norm_img)
    print('heatmapshow:',heatmapshow)
    # norm_img = np.asarray(norm_img, dtype=np.uint8)
    # # 步骤2：应用色彩映射
    # norm_img1 = norm_img*255
    # # print(format(norm_img1, '.2f'))
    # print('norm_img1:',norm_img1)
    heat_img = cv2.applyColorMap(heatmapshow , cv2.COLORMAP_JET)
    imageCRAFT = cv2.resize(image,(224,224))
    add_img = cv2.addWeighted(imageCRAFT, 1, heat_img, 0.3, 0) 
   #五个参数分别为 图像1 图像1透明度(权重) 图像2 图像2透明度(权重) 叠加后图像亮度


    cv2.imwrite('output/'+'img_res.jpg'+img_name,add_img)
    
    # thresh 返回的二值化图像
    _, thresh = cv2.threshold(img_res, 35, 255, cv2.THRESH_BINARY) # 255白色
    cv2.imwrite('output/'+'thresh.jpg'+img_name,thresh)  # 异常图片显示（白色区域）
 
    # print('thresh',thresh) 
    # contours  轮廓列表 
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 只检测白色
    # if output[1] > 10:
    #     for cnt in contours:
    #         x, y, w, h = cv2.boundingRect(cnt) # 获取矩形左上角坐标和宽高
    #         cv2.rectangle(image, (int(x*ratio_x), int(y*ratio_y)), (int((x + w)*ratio_x), int((y + h)*ratio_y)), (0, 0, 255), 2)
 
    #     cv2.imwrite('output/'+f'{img_name}', image)
    # else:
    #     cv2.imwrite('output/'+f'{img_name}', image)
    # print(f'{img_name}:', output[1])
 
 
if __name__ == '__main__':
    onnx_path = r"E:\anomalib-main\results\Patchcore\latest\weights\onnx\model.onnx"
    img_path = r"E:\mvtec_anomaly_detection.tar\bottle\test\contamination"
    import os
    from tqdm import tqdm
    img_list = os.listdir(img_path)
    for i in tqdm(img_list):
        img_path_ = os.path.join(img_path, i)
        infer_onnx(onnx_path, img_path_,i)