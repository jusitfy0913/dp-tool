from pathlib import Path
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from torch.utils.data import DataLoader
from anomalib.utils.visualization import ImageVisualizer
from anomalib.models.image.padim import Padim
# from anomalib.visualization import Visualizer
# from anomalib.utils.visualization import Visualizer
from anomalib.models import Patchcore,EfficientAd
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.data import PredictDataset
from torchvision.transforms.v2 import Resize
import onnx_graphsurgeon as gs
import numpy as np
import onnx
# import anomalib
# print('Anomalib version: ', anomalib.__version__)
# from anomalib.metrics import F1Score, AUROC,Evaluator
# import torch
# from anomalib.data import ImageBatch

# 验证器
# f1_score = F1Score(fields=["pred_label", "gt_label"])
# auroc = AUROC(fields=["pred_score", "gt_label"])
# evaluator = Evaluator(test_metrics=[f1_score])

# val_metrics = [
#     AUROC(fields=["pred_score", "gt_label"]),     # Image-level AUROC
#     F1Score(fields=["pred_label", "gt_label"])    # Image-level F1
# ]

# # Test metrics (more comprehensive)
# test_metrics = [
#     AUROC(fields=["pred_score", "gt_label"]),     # Image-level AUROC
#     AUROC(fields=["anomaly_map", "gt_mask"]),     # Pixel-level AUROC
#     F1Score(fields=["pred_label", "gt_label"]),   # Image-level F1
#     F1Score(fields=["pred_mask", "gt_mask"])      # Pixel-level F1
# ]

# # Create evaluator with both sets
# evaluator = Evaluator(
#     val_metrics=val_metrics,
#     test_metrics=test_metrics,
#     compute_on_cpu=False
# )


datamodule = Folder(
    name="demo_1",
    root=r"E:\mvtec_anomaly_detection.tar\bottle",
    normal_dir=r"E:\mvtec_anomaly_detection.tar\bottle\train\good",
    abnormal_dir=r"E:\mvtec_anomaly_detection.tar\bottle\test\broken_small",
    normal_test_dir=r"E:\mvtec_anomaly_detection.tar\bottle\test\good",
    task="classification",
    image_size=[256, 256],
    num_workers=0, 
    val_split_ratio=0.5,
    train_batch_size = 1,
)
# visualizer2 = Visualizer()


# datamodule = Folder(
#     name="demo_1",
#     root=r"E:\mvtec_anomaly_detection.tar\bottle",
#     normal_dir=r"E:\mvtec_anomaly_detection.tar\bottle\train\good",
#     abnormal_dir=r"E:\mvtec_anomaly_detection.tar\bottle\test\broken_large",
#     num_workers=0, 
#     val_split_ratio=0.5,
# )
    # Examples:
    #     >>> from pathlib import Path
    #     >>> dataset = PredictDataset(
    #     ...     path=Path("path/to/images"),
    #     ...     image_size=(224, 224),
    #     ... )
    #     >>> len(dataset)  # doctest: +SKIP
    #     10
    #     >>> item = dataset[0]  # doctest: +SKIP
    #     >>> item.image.shape  # doctest: +SKIP
    #     torch.Size([3, 224, 224])
    # """
# Create visualizer with default settings
visualizer1 = ImageVisualizer()
dataset = PredictDataset(
    path=Path(r"E:\mvtec_anomaly_detection.tar\bottle\test\broken_small"),
    image_size=(256, 256),
)
# >>> from anomalib.data.utils import InputNormalizationMethod, get_transforms
#         >>> transform = get_transforms(
#         ...     image_size=256,
#         ...     normalization=InputNormalizationMethod.NONE
#         ... )
#         >>> dataset = FolderDataset(
#         ...     name="custom",
#         ...     normal_dir="datasets/custom/good",
#         ...     abnormal_dir="datasets/custom/defect",
#         ...     split="train",
#         ...     transform=transform
#         ... )
transform = Resize((256, 256))
# datamodule = FolderDataset(
#     name="demo_1",
#     root=r"E:\mvtec_anomaly_detection.tar\bottle",
#     normal_dir=r"E:\mvtec_anomaly_detection.tar\bottle\train\good",
#     abnormal_dir=r"E:\mvtec_anomaly_detection.tar\bottle\test\contamination", 
#     split="test",
    
# )
# transform=transform,
# Initialize components
# datamodule = MVTec()
dataloader = DataLoader(dataset)
model = Patchcore( )
# model = Padim( )

dataset1 = PredictDataset(
    path=Path(r"E:\mvtec_anomaly_detection.tar\bottle\test\broken_small"),
    image_size=(256, 256),
)

# model = EfficientAd()
engine = Engine()
predictions = engine.predict(
    model=model, 
    dataset=dataset,
    # dataloaders=[dataloader],    
    ckpt_path=r"E:\anomalib-main\results\Patchcore\demo_3\v0\weights\lightning\model.ckpt",  # 存放model.ckpt文件的路径，需要对应修改
)
# 5. Access the results
# if predictions is not None:
#     for prediction in predictions:
#         image_path = prediction.image_path
#         anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
#         pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
#         pred_score = prediction.pred_score  # Image-level anomaly score
# Process results
print("\nProcessing Results...")
# print(predictions)
if predictions is not None:
    for prediction in predictions:
        #2.0anomaly_
        # print('--start--')
        # print(prediction)
        # image_path = prediction.image_path
        # print(image_path)
        # anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
        # print(anomaly_map)
        # pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
        # print(pred_label)
        # pred_score = prediction.pred_score  # Image-level anomaly score
        # print(pred_score)
        if 'image' in prediction:
            print("键 'image' 存在于字典中")
        else:
            print("键 'image' 不存在于字典中")
        # if hasattr(prediction, 'image'):
        #             print("prediction对象具有image属性")
        # else:
        #             print("prediction对象不具有image属性")
        # if hasattr(prediction, 'image_path'):
        #     print("prediction对象具有image_path属性")
        # else:
        #     print("prediction对象不具有image_path属性")
        # if hasattr(prediction, 'pred_boxes'):
        #     print("prediction对象具有pred_boxes属性")
        # else:
        #     print("prediction对象不具有pred_boxes属性")
        #1.2anomaly
        print('anomaly_map:',prediction.get('anomaly_maps')) # Pixel-level anomaly heatmap
        print('pred_score:',prediction.get('pred_scores')) # Image-level label (0: normal, 1: anomalous)
        print('pred_boxe:',prediction.get('pred_boxes'))
        print('box_label:',prediction.get('box_labels')) # Image-level label (0: normal, 1: anomalous)
        # anomaly_score = prediction.pred_scores
        # image_path = prediction.image_path
#导出onnx        
# model:模型、
# export_type:导出类型、
# export_root: 导出根目录、
# input_size: 定义推理图像的输入宽高、
# transform: 预处理、
# compression_type: 用于指定导出模型时是否进行压缩以及压缩类型、
# datamodule: 如果导出过程中需要特定的数据集或数据处理逻辑，可以通过这个参数传递一个数据模块用于评估模型精度
# metric： 用于指定评估模型性能的度量标准
# ov_args： 用于传递额外的参数给OpenVINO优化工具

onnx_model = engine.export(
model=model,
export_type='onnx',
export_root=None,
input_size=[256, 256],
#1.2删除transform
transform=None,
compression_type=None,
datamodule=None,
metric=None,
ov_args=None,
ckpt_path=r"E:\anomalib-main\results\Patchcore\demo_3\v0\weights\lightning\model.ckpt",  # 存放model.ckpt文件的路径，需要对应修改
)
print(onnx_model)
# #导出修改gather
# graph = gs.import_onnx(onnx.load(r"E:\anomalib-main\results\EfficientAd\latest\weights\onnx\model.onnx"))
# for node in graph.nodes:
#     print('------------------------------------')
#     print(node)
#     print('------------------------------------')
#     print(node.name)
#     print('------------------------------------')
#     if node.op == "Gather_2711":
#         print(node)
#         print(node.inputs[1])
#         node.inputs[1].values = np.int64(5)
#         print(node.inputs[1])
#     if node.op == "Gather_2713":
#         print(node)
#         print(node.inputs[1])
#         node.inputs[1].values = np.int64(5)
#         print(node.inputs[1])
#     if node.op == "Gather_17":
#         print(node)
#         print(node.inputs[1])
#         node.inputs[1].values = np.int64(5)
#         print(node.inputs[1])
#     if node.op == "Gather":
#         print(node)
#         print(node.inputs[1])
#         node.inputs[1].values = np.int64(5)
#         print(node.inputs[1])

# onnx.save(gs.export_onnx(graph), r"E:\anomalib-main\results\EfficientAd\latest\weights\onnx\modelprocess.onnx")
