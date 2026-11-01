# 6-web —— TRTForge Web Studio

浏览器版工作站，用来替代 `4-qt-studio` 的 Qt 桌面程序。

## 为什么有这个目录

Qt 版本的问题是：**每台机器都要先装 Qt 6.8.3 + MSVC 才能编出界面**。本机长期没有 Qt，
`TRTForge.exe` 一直编不出来，界面等于不可用。

Web 版把界面换成浏览器，换来三件事：

| 事项 | Qt 版 | Web 版 |
| --- | --- | --- |
| 装环境 | Qt 6.8.3 + MSVC，编译几十分钟 | 装 Python 包，秒级启动 |
| 远程使用 | 只能坐那台机器前 | 局域网内任何浏览器 |
| 改界面 | 重新编译整个程序 | 改完刷新 |

**底层能力一行没重写**：训练仍走原来的 launcher，推理仍走原来编好的 DLL。

## 快速开始

### 1. 装依赖

```powershell
# 用装了 torch 的那个环境（训练需要）
D:\Anaconda3\envs\dpenv\python.exe -m pip install -r 6-web\backend\requirements.txt
```

关于 `numpy`：早期这里锁了 `numpy<2`，因为 torch 2.3.0 是按 numpy 1.x 编的。2026-10-30 实测
`torch 2.3.0 + numpy 2.5.4` 的 CUDA 前向与 numpy 桥接均正常，且 `onnxruntime 1.20.1` 已支持 numpy 2，
所以上限放宽到 `<3`。

为什么不干脆锁回 `<2`：Ultralytics 的 AutoUpdate 会在 `yolo export` 时把 numpy 自动升到 2.x，
锁 `<2` 只会让环境在每次导出后反复漂移。唯一残留告警是 `scipy 1.14.0` 声明要求 numpy<2.3，
导入时会打一行 `UserWarning`，实测不影响 anomalib 训练与推理。

### 2. 编推理 DLL（只需一次）

Web 版的推理依赖 `yolo_infer.dll` 与 `anomalib_infer.dll`。它们**不依赖 Qt**，但要 OpenCV。

```powershell
$env:OPENCV_DIR = "E:\thirdparty\opencv"
$env:NVCC_PREPEND_FLAGS = "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH --allow-unsupported-compiler"

powershell -File scripts\build_infer_dlls.ps1
```

第二个环境变量是给 **CUDA 11.8 + MSVC 14.44** 这组搭配用的，原因见「常见问题」。
产物落在 `4-qt-studio/bin/`。

### 3. 启动

**最省事的方式（推荐）**：双击仓库根目录的 `启动Web端.bat`。
它会自动找 Python、检查端口、等服务起来后再打开浏览器。

```bat
启动Web端.bat              :: 默认 127.0.0.1:8060
启动Web端.bat 8070         :: 换端口
启动Web端.bat 8070 nobrowser  :: 不自动开浏览器
启动Web端.bat 8060 reload     :: 改后端代码时热重载
```

也可以直接跑 PowerShell 脚本（参数更全）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_web_studio.ps1
```

打开 <http://127.0.0.1:8060>。顶部状态灯变绿写「环境就绪」就说明一切正常。

两个脚本都会做**端口预检**：端口被占用时会告诉你占用进程与启动时间，
交互式终端里还会问一句「结束它并继续启动？」（答 N 则中止，退出码 1）。
这条检查不是多余的——旧进程没关干净时 uvicorn 只报 10048，
用户却以为重启成功，实际连的还是旧代码。

> `scripts\start_web.bat` 内部刻意只用 ASCII：cmd.exe 解析含多字节字符的
> UTF-8 批处理时会截断/错行（码页老问题），中文提示因此全部放在
> `run_web_studio.ps1` 里。

常用参数：

```powershell
# 换端口
powershell -File scripts\run_web_studio.ps1 -Port 8070

# 指定训练解释器
powershell -File scripts\run_web_studio.ps1 -PythonExe D:\Anaconda3\envs\dpenv\python.exe

# 改前端代码时热重载
powershell -File scripts\run_web_studio.ps1 -Reload

# 让局域网别的机器也能访问
powershell -File scripts\run_web_studio.ps1 -HostName 0.0.0.0
```

## 页面说明

### 训练导出

1. 选「训练类型」：YOLO 或 Anomalib，表单会自动切字段。
2. 填 `dataset.yaml`（YOLO）或数据集 root（Anomalib），可用「浏览」按钮选。
3. 点「检测环境」先确认 Python / torch / CUDA 是否就绪（会列出缺哪个包和修复命令）。
4. 点「开始训练」，右侧实时出日志、进度条、指标。

日志是流式的，不是训练完才一次性吐出来。中间关掉浏览器训练**不会停**，重新进来点任务还能继续看。

#### 数据增强（v1.7）

表单里有「数据增强（留空则用默认值，共 15 项）」折叠区，包含
`hsv_h/s/v`、`degrees`、`translate`、`scale`、`shear`、`perspective`、
`flipud`、`fliplr`、`bgr`、`mosaic`、`mixup`、`copy_paste`、`close_mosaic`。

- **留空 = 不传 = 用 Ultralytics 默认**，行为与升级前完全一致，历史实验仍可对比。
- **填 `0` = 显式传 0**，不会被当成「没填」而忽略。
- 两个快捷按钮：
  - 「小缺陷预设」→ `mosaic=0.1, close_mosaic=20, scale=0.3, translate=0.05`
  - 「清空（用默认）」→ 所有增强字段恢复留空
- 概率类字段超出 `0~1` 或 `close_mosaic` 为负时，后端**启动前**直接返回 400，不会白等几十秒才报错。


### 推理演示

> **v1.8 起为「推理控制台」，相对 v1.7 的变化见本节末尾的「v1.8 变更点」。**

1. 选 YOLO 或 Anomalib。
2. 从「模型」下拉选一个训练产物（也可手工填路径，下拉不会强制覆盖手填值）。
   下拉数据来自后端扫描 `runs/**/weights/*.engine` 与 `results/**/*.engine`，
   选中后会自动填好引擎、同名 ONNX、标签文件、`metadata.json` 与任务类型。
3. 调参数：YOLO 有 `conf` / `nms` 滑块；Anomalib 有图像阈值、像素阈值两组滑块
   （勾上「沿用 metadata 默认值」则不覆盖）。
4. 选图片，点「开始推理」。

**v1.8 变更点**

| 项 | v1.7 | v1.8 |
| --- | --- | --- |
| 类别名 | 只显示 `cls0`/`cls1` | 读 `labels_*.json` 显示真实缺陷名（缺失字段、BOM、`null` 均容忍） |
| 检出框 | 后端画死在 PNG 上 | 后端只出**干净原图** + 前端 SVG 叠加层；后端 PNG 仅供导出 |
| 缩放时的标签 | 跟着一起放大到看不清 | 标签用逆缩放补偿，任何倍率下都是恒定字号 |
| 参数输入 | 数字框，直接敲 | 滑块 + 数字框双向联动，非法值即时回退 |
| Anomalib 阈值 | 只能用 `metadata.json` 里的 | 可覆盖，响应同时给出「生效值 / 原值」并标明是否被覆盖 |
| 热能图 | 无 | `cv2.applyColorMap` 渲染，可选色带，阈值变化即时反映 |
| 引擎句柄 | 无上限，改参数会不断累积 VRAM | LRU，上限 4，淘汰时**立即** `yi_destroy` / `ai_destroy` |
| 结果明细 | 只有一段文字 | 可排序表格，点行高亮对应框；置信度直方图；计时分解条 |
| 导出 | 只能右键存图 | JSON / CSV 客户端导出，内容与屏幕完全一致 |

结果区是**独立窗口**，排在同一排，每个都能单独操作：

| 窗口 | 内容 | 默认 |
| --- | --- | --- |
| 原图 | 上传的原始图片（YOLO 的框叠在这上面） | 开 |
| 检出结果 / 异常叠加 | YOLO 为 SVG 框层；Anomalib 为热力叠加图（可调透明度） | 开 |
| 热能图 | 仅 Anomalib，纯热力图，无原图内容 | 关 |
| 异常轮廓 | 仅 Anomalib，红色异常区域掩码 | 关 |

窗口顶部的胶囊按钮可独立开关显隐，**开关状态会记住**（存在浏览器本地，
键名带页面标识，所以不会和以后标注页的同名窗口互相覆盖）。

每个窗口内部的图像交互：

| 操作 | 效果 |
| --- | --- |
| 滚轮 | 以**光标位置为锚点**缩放（0.1× ~ 20×），光标下的那个缺陷不会跑掉 |
| 左键拖动 | 平移；拖出窗口边界也不中断 |
| 双击 / `FIT` | 复位成完整可见 |
| `1:1` | 原始像素大小，用来判断缺陷的真实像素尺寸 |
| `+` / `−` | 以视口中心为锚点缩放 |

倍率实时显示在窗口右上角。窗口内滚动不会带动整页滚动。

YOLO 返回干净原图与每个目标的类别名/置信度，框由前端画在 SVG 上
（`js/overlay.js`，`viewBox` 取图像原始像素尺寸，线宽用 `non-scaling-stroke`
保证缩放时不变粗）；Anomalib 返回异常分数、OK/NG 判定、热力叠加图、纯热力图
和红色异常轮廓图，并附带生效阈值。

> 推理需要引擎文件。仓库里没有现成的 `.trt`，得先用训练页导出 ONNX，再转 TensorRT。
>
> ⚠️ **YOLO 引擎必须用 `.transd.onnx` 编译，不能用 `best.onnx`。**
> 两者输出布局不同：
>
> | 文件 | 输出 shape | DLL 能否解析 |
> | --- | --- | --- |
> | `best.onnx` | `[1, 4+类别数, 候选框数]` | ❌ |
> | `best.transd.onnx` | `[1, 候选框数, 4+类别数]` | ✅ |
>
> 用错时 DLL **不报错**，但会返回 `confidence=957`、`top=-725` 这类垃圾框。
> Web 接口已加校验：出现置信度不在 0~1 的情况直接返回 **HTTP 400**，
> 提示「请用 `.transd.onnx` 编译」。
>
> 完整实测链路见根目录 `readme.md` 的 v1.3 记录（PCB 6 类 / mAP50 0.945 / 引擎 8.41ms）。

### 引擎构建

把 ONNX 编译成 TensorRT 引擎，参数与 `trtexec` 一一对应。

1. 选 ONNX 文件。选完会自动探测并显示输入/输出形状与文件大小。
2. 输出引擎留空即可，会按 `<onnx名>_<fp16|fp32>.engine` 自动推导。
3. 选精度（fp16 省显存，推荐）、workspace，需要变分辨率时勾「动态 shape」并填 min/opt/max。
4. 点「开始构建」，日志实时滚动，状态徽标从「构建中」变为「已完成/失败」。

选错文件不会白等：输出形状是 `[1, 4+类别数, 候选框数]` 的「未转置」YOLO 模型，
提示会变红，后端也会直接返回 400 拒绝构建。

构建完成后，引擎文件路径可直接填到「推理演示」页使用。

实测参考（PCB 6 类模型，960×960，FP16，RTX 3060 8G）：

| 项 | 数值 |
| --- | --- |
| 构建耗时 | 474.1 s |
| 引擎大小 | 10.14 MB |
| 吞吐 | 135.4 qps |

> trtexec 在 960×960 这类大分辨率上做 tactic 搜索可能长达数分钟，期间日志可能长时间不刷新，
> 这是正常的，不是卡死。GPU 占用率持续接近 100% 即为仍在编译。

### 数据集

把已有的标注重新划分、并导出成训练可直接用的格式。

1. 选数据源：YOLO 数据集目录（含 `images/`、`labels/`、`data.yaml`），或 Qt 的 `.tfanno` 工程文件。
2. 任务类型默认「自动识别」（按标签每行字段数猜 detect / segment / pose / obb），猜错可手动指定。
3. 填 train / val 比例，右侧会实时显示 `test = 100 - train - val`。
4. 点「扫描预览」：显示总图片、各集张数、空标注数、类别数，以及每张图的「现有划分 → 新划分」。
5. 选导出格式与输出目录，点「开始导出」。

**划分规则（与 Qt 版对齐，务必注意）**

- 比例是**向下取整**：`train = 总数 × train% ÷ 100` 取整、`val` 同理；
- **剩下的全部归 test**，所以 80/20 时 test 是余数（692 张 → 553 / 138 / 1），
  填 70/20/10 才能得到 70 / 20 / 剩余 10；
- 排序用 **FNV-1a 64 位 + `_2026` 盐值**，保证同一份数据在任何机器、任何进程上
  结果逐项一致。Qt 用的是 `qHash`，哈希函数不同，因此与 Qt 历史划分**不逐张一致**，
  这是为了让划分可复现而做的取舍。

**导出格式**

| 格式 | 产物 | 限制 |
| --- | --- | --- |
| YOLO | `images/{train,val,test}` + `labels/{split}/*.txt` + `dataset.yaml` | 全任务支持；`dataset.yaml` 已实测能被 Ultralytics 直接识别 |
| COCO | `annotations.json`（images / annotations / categories） | 不复制图片，图片以 `file_name` 记录；不支持 OBB |
| VOC | `Annotations/*.xml` + `JPEGImages/` | **仅 detect**（水平矩形），其余任务直接拒绝 |

**误操作防护**

- 输出目录不能放在数据源内部（会自我复制）；
- 目标目录非空时必须勾「允许覆盖」；
- `train + val > 100` 直接拒绝；
- 比例向下取整导致某集为 0 张时，导出摘要会明确告警，不会静默产出空训练集。

实测参考（PCB 6 类 / 692 张 / 3034×1586）：扫描 692 张、2950 个对象、0 空标注；
70/20/10 → 484/138/70；YOLO 标签往返逐值无损；COCO 2950 条标注；VOC 692 个 XML。
验收脚本：`python 6-web/backend/tools/test_dataset_workbench.py`（20 项）。

### 标注（v1.9 新增）

在浏览器里把 `.tfanno` 工程打开、画框、存盘、导出训练集，**不再需要装 Qt**。
格式与 `4-qt-studio` 完全互通：Web 存的工程 Qt 能开，Qt 存的工程 Web 能开。

#### 应用场景

| 场景 | 用法 |
| --- | --- |
| 人工标注新数据集 | 打开 / 新建 `.tfanno`，逐个画框、分集合，导出 YOLO 后直接训练 |
| 在已有数据集上补标 | 直接打开 Qt 的历史工程，改完存回原文件，Qt 侧无缝继续 |
| 机器预标注 + 人工修正 | 用训练好的引擎一键生成草稿，只留对的，全部接受作为一次可撤销操作 |
| 多人/多标签页协作 | 修订号乐观锁：两人同时改同一个工程，后保存的人会看到冲突弹窗而不会静默覆盖 |

#### 上手步骤

1. 切到「标注」页签。顶部填工程路径（`.tfanno`），点「打开」。
   文件不存在时后端返回空白骨架，前端进入「新建」流程，点「新建」即按当前路径落盘。
2. 左侧是图片列表（显示每张图的标注数与未保存标记）；右上角「添加图片」可扫描目录批量加入。
3. 中栏画布选工具（快捷键见下），左键画，`Esc` 取消，`Enter` 结束多边形 / 关键点。
4. 右栏选类别；选中对象后属性面板可改标签、描述、分组、集合、置信度、难例、标记、属性。
5. `Ctrl+S` 手动保存；不按也会在停止编辑 3 秒后自动保存（保存不进撤销栈）。
6. 「自动标注」展开后可生成草稿；「导出」选格式与目录。

#### 快捷键

| 键 | 作用 | 键 | 作用 |
| --- | --- | --- | --- |
| `V` | 选择 / 拖动 | `Ctrl+Z` | 撤销 |
| `R` | 矩形 | `Ctrl+Shift+Z` / `Ctrl+Y` | 重做 |
| `O` | 旋转矩形 | `Ctrl+S` | 保存 |
| `P` | 多边形 | `Delete` / `Backspace` | 删除选中 |
| `L` | 线段 | `Enter` | 结束多边形 / 关键点 |
| `T` | 点 | `Esc` | 取消绘制 / 取消选中 |
| `K` | 关键点组 | `F` | 适配窗口 |
| `Shift+拖动` | 正交约束（水平 / 垂直） | `Alt+拖动` | 临时关闭吸附 |
| `Ctrl+点击顶点` | 删除该顶点（少于 3 点自动删对象） | 点中边中点手柄拖动 | 插入 / 移动顶点 |

撤销栈深度 50，新编辑清空重做栈。拖动但没移动（只是点了一下手柄）**不会**污染撤销栈。

#### `.tfanno` 参数说明

根结构（新增键都被 Qt 静默忽略，因此双向兼容）：

```json
{
  "version": 1,
  "task": "detect",
  "classes": [{ "id": 0, "name": "missing_hole", "color": "#22d3ee" }],
  "images": [
    {
      "path": "images/01.jpg", "width": 3034, "height": 1586, "split": "train",
      "annotations": [
        {
          "id": "a1", "class_id": 0, "label": "missing_hole",
          "description": "", "group_id": 0, "shape_type": "rectangle",
          "points": [[100, 120], [180, 200]],
          "rotation": 0, "keypoint_visibility": [],
          "score": 0.87,
          "difficult": false, "flags": {}, "attributes": {}
        }
      ]
    }
  ],
  "rev": 7
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `task` | `detect`\|`segment`\|`pose`\|`obb` | 未知值回退 `detect` 并记警告 |
| `shape_type` | `rectangle` / `rotated_rect` / `polygon` / `line` / `point` / `keypoints` | 别名自动归一；未知值回退 `rectangle` |
| `points` | `[[x,y],...]` | 图像像素坐标。矩形 **只存 2 点**（左上、右下）；读入 4 点会自动折叠 |
| `keypoint_visibility` | `[0\|1\|2,...]` | 与 `points` 等长；关键点保持**单对象**，绝不拆成多个 `point` |
| `split` | `train` / `val` / `test` | 未知值回退 `train` |
| `rev` | int | 修订号，乐观锁用；每次成功保存 +1 |
| `difficult` / `flags` / `attributes` | — | 扩展字段，**默认值不落盘**，只有非默认才写出 |
| `score` | float | 仅草稿接受时写入，人工画的框不带 |

后端另返回 `meta.mtime_ns`（**字符串**）。为什么是字符串：`st_mtime_ns` 约 `1.79e18`，
超过 JS `Number.MAX_SAFE_INTEGER`（`9e15`），`JSON.parse` 会把低位四舍五入，
导致刚打开就报「文件被外部修改」。因此该值按字符串传输，服务端比对时留 `4096ns` 容差。

#### 自动标注

两条通道，结果统一成草稿列表（渲染成虚线，**不落盘、不进撤销栈**，直到被接受）：

| 通道 | 依赖 | 用途 |
| --- | --- | --- |
| 主通道 | 自有 `yolo_infer.dll`（或 onnxruntime 降级） | 用训练好的 `.engine` 出框；`obb` 任务保留 `rotation`，`pose` 任务聚合成单个关键点组 |
| 次通道 | 外部 HTTP 服务（可选） | 接 SAM / 文字提示这类重模型；**不走 import**，避免 GPL-3.0 传染 |

外部服务契约（最小集）：

```
POST {service}/predict    multipart/form-data
  file    : 图片二进制
  prompts : JSON 字符串，形如 {"text": "hole", "points": [[x,y]], "model": "sam2"}
-> 200 {"shapes": [{"label","shape_type","points","group_id","score","rotation"}]}
```

失败分类：`unreachable` / `timeout` / `http-error` / `bad-response` / `no-service`，
前端按类别给不同提示。**服务不可达时不会静默回退**，会明确写「外部服务未接入，已只用本地引擎」。

草稿接受时有两个保护：

- **脏数据清洗**：引擎输出维度与 `yolo_infer.dll` 期望不一致时（比如拿 `best.onnx` 编的引擎），
  DLL 不报错但会返回 `confidence=957` 这类垃圾框。`autolabel.sanitize_drafts()` 会把
  置信度不在 `[0,1]`、宽高非正、超出图像范围的草稿丢掉，并在响应 `warnings` 里写明原因。
- **类别自动补全**：草稿的 `label` 在工程里没有对应类别时，接受动作会按名字建类别
  （同名则复用），不会出现越界的 `class_id`。

#### 导出

「导出」会把**内存中的当前工程**（含未保存修改）交给 `dataset.py` 的同一套
`export_yolo` / `export_coco` / `export_voc`，与「数据集」页共用实现，不会出现两套导出逻辑打架。

| 格式 | 摘要显示 | 限制 |
| --- | --- | --- |
| YOLO | `train/val/test` 张数 | 全任务 |
| COCO | `images` / `annotations` / `empty` 条数 | 不支持 OBB |
| VOC | `images` 张数 | 仅 `detect` |

#### v1.9 变更点（相对 v1.8）

| 项 | v1.8 | v1.9 |
| --- | --- | --- |
| 标注能力 | 无，只能去 Qt 版 | 浏览器内完成画框 / 改类别 / 分集合 / 存盘 / 导出 |
| 布局 | 单栏页面，长页面滚动 | 标注页是**单屏固定高度**（`calc(100vh - 154px)`），三栏各自滚动，画布不抖动 |
| 撤销 | 无 | 双栈各 50 步，拖动只在真正位移后才入栈 |
| 并发 | 无概念 | `rev` 乐观锁 + `mtime_ns` 字符串指纹，冲突弹「重新加载 / 强制覆盖」二选一 |
| 自动标注 | 无 | 本地引擎为主 + 可选外部服务，草稿层独立渲染，可单条 / 全部接受与拒绝 |
| 类别名 | 只在推理页 | 标注页与推理页共用 `labels.py` 的配色与命名 |
| 导出 | 只能导出推理结果图 | 工程级导出 YOLO / COCO / VOC，与数据集页同实现 |

验收脚本：`python 6-web/backend/tools/test_annotation_project.py`（138 项），
外加浏览器实测六种形状、撤销重做、冲突弹窗、真实 `.engine` 自动标注与三种格式导出。

#### v1.12 修复（代码审查发现，按严重度从高到低）

| 严重度 | 问题 | 修复方式 |
| --- | --- | --- |
| 高 | 「批量加入图片」用 GET 调后端的 POST 扫描接口，直接 405 | 客户端改发 POST，参数仍走查询串 |
| 高 | 自动保存期间的新编辑会被丢弃（`saving` 时直接 return）且保存成功即无条件清除脏标记 | 增加编辑序号 `editSeq` 比对 + `saveQueued` 补存；`flushSave` 先等在飞请求落地 |
| 中 | 「描述」字段改完不标脏、不排期保存，关页面即丢 | 与其它属性一致地走 `afterEdit()` |
| 中 | 快速切图时旧图解码后到，画布出现「上一张的位图 + 这一张的标注」 | `imageLoadSeq` 序号比对，过期回调直接丢弃 |
| 中 | 旋转框拖边先转成 AABB 再写回，把斜放的框「摆正」 | 在框自身的局部坐标系（`R(-angle)`）里只改目标那条边 |
| 中 | 「新建」固定 `force: true` 静默覆盖磁盘上的已有工程 | 先探测是否存在并要求确认；先写盘成功再接管内存状态 |
| 中 | 撤销/重做只改数组不切视图，可能在别的图上「莫名少框」 | 快照带图片索引，索引变化时同步切图并重新解码 |
| 中 | 同进程并发保存共用 pid 临时文件，且「读 rev → 写回」无锁 | 临时名加 uuid；按绝对路径加写锁，`touch()` 共用同一把锁 |
| 低 | 校验器用同一个 `seen_ids` 混算图片 id 与标注 id，正常的 Qt 工程被误报重复 | 图片 id 全局去重、标注 id 逐图去重 |

### 环境说明

显示后端探测到的全部路径、Python 解释器、DLL 加载状态，以及可用的环境变量清单。
出问题时先看这一页。

顶栏标题下方的版本号徽标取自 `GET /api/health` 的 `version` 字段。
它不是装饰：端口被旧进程占着时徽标会停在旧值，是排查「改了代码没效果」最快的入口。

## 前端视觉体系

前端是**零构建**的：不装 npm、不打包、不引第三方库，改完刷新即可。视觉改造也遵守这条约束，
所有效果都由原生 CSS 与原生 JS 实现。

### 设计令牌

`css/style.css` 的 `:root` 里定义了全部设计令牌，**页面中不允许再出现硬编码的颜色与间距**。
令牌分三组，改主题和改尺度互不干扰：

| 组 | 内容 | 例子 |
| --- | --- | --- |
| 语义色 | 四级背景、两级文字、线条、四个强调/状态色 | `--bg` `--surface-1..3` `--text` `--cyan` `--danger` |
| 尺度 | 间距阶、字号阶、圆角、字体族 | `--sp-1..6`（4/8/12/16/24/32）、`--fs-2xs..2xl` |
| 效果 | 发光、网格与扫描线不透明度、动效时长 | `--glow-sm` `--grid-opacity` `--duration-base` |

派生色（`--cyan-a12` 之类的半透明变体）也集中定义，这样改一次主色能一次改透。

### 可读性护栏

两条硬约束，都有自动化检查兜底：

1. **正文与面板底色对比度不低于 WCAG AA 的 4.5:1**。
   `tests/test_contrast.mjs` 直接从 `style.css` 读令牌按 WCAG 公式算，
   任何一处跌破门槛就失败。
2. **装饰不覆盖图像**。网格与扫描线画在 `body::before/::after` 上（`z-index:0`、`pointer-events:none`），
   所有内容容器抬到 `z-index:1`，查看器的图像视口再额外 `isolation:isolate`。

关闭动效有两条路，效果完全相同：

- 系统偏好「减少动效」→ `@media (prefers-reduced-motion: reduce)`；
- 手动验收 → 网址后加 `?reduced-motion=1`。

两条路都只是把**效果组令牌归零**，语义色与尺度不受影响。

### 显隐约定

前端统一用 `el.hidden = true/false` 切换显隐，样式表基础层有一条：

```css
[hidden] { display: none !important; }
```

原因是浏览器给 `[hidden]` 的默认样式只是 `display: none`，优先级低于任何类选择器——
`.viewer-empty { display: grid }` 会把它盖掉，元素"藏了却还在"。这条护栏让 `.hidden` 类
与 `hidden` 属性行为一致，两种写法混用也不出错。

### 图像查看器（`js/viewer.js`）

推理页的结果窗口与后续标注页的画布共用同一个组件，避免两处实现导致「手感不一致」。

坐标系约定（全局唯一）：`p_screen = p_image * scale + offset`。
以光标为锚点缩放时必须同时改倍率与平移：

```
p_image = (cursor - offset0) / scale0
offset1 = cursor - p_image * scale1
```

只改倍率不改平移，表现就是「越缩放，光标下的东西跑得越远」。
这个公式写成纯函数 `computeZoomTransform`，并由 `tests/test_viewer_math.mjs` 覆盖
（含连续缩放 8 次后锚点漂移 ≤ 0.5px 的断言），标注页直接复用。

| 导出 | 用途 |
| --- | --- |
| `createViewer({mount, id, title})` | 单个窗口：滚轮缩放、拖动平移、工具条、倍率读数 |
| `createViewerStrip({mount, id, windows})` | 多窗口容器：自适应网格 + 开关胶囊 + 空状态 + 状态持久化 |
| `computeZoomTransform` / `screenToImage` / `clampScale` | 纯函数，可在 node 里单测 |

布局用 `repeat(auto-fit, minmax(300px,1fr))` 而不是 flex：flex 在最后一行只剩一两个元素时会把它们
拉满，导致同行窗口宽度不一致。超过三个窗口时切为定列数并允许横向滚动，不压缩窗口。

## 环境变量

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `TRTFORGE_PYTHON` | 训练用解释器 | 自动探测（当前解释器 / dpenv / pytorch） |
| `TRTFORGE_TRTEXEC` | `trtexec.exe` 绝对路径 | 自动探测（`CUDA_PATH\bin` → CUDA 根 `v*/bin` → PATH） |
| `OPENCV_DIR` | OpenCV 根目录 | 未设则推理 DLL 加载失败 |
| `CUDA_PATH` | CUDA 根目录 | 未设则推理 DLL 加载失败 |
| `TRTFORGE_REPO_ROOT` | 仓库根 | 从 `config.py` 位置向上推导 |
| `TRTFORGE_BIN_DIR` | DLL 所在目录 | `<仓库>/4-qt-studio/bin` |
| `TRTFORGE_YOLO_LAUNCHER_DIR` | YOLO launcher 目录 | `<仓库>/1-python-yolo/launcher` |
| `TRTFORGE_ANOMALIB_LAUNCHER_DIR` | Anomalib launcher 目录 | `<仓库>/2-python-anomaly/launcher` |
| `TRTFORGE_ANOMALIB_DATASET_ROOT` | Anomalib 数据集根 | `<仓库>/2-python-anomaly/anomalib-main/datasets/bottle` |
| `TRTFORGE_BROWSE_ROOTS` | 额外允许浏览的目录，分号分隔 | 仅仓库根 |
| `TRTFORGE_MAX_UPLOAD_MB` | 上传图片上限 | 32 |
| `TRTFORGE_WEB_PORT` / `TRTFORGE_WEB_HOST` | 监听端口 / 地址 | 8060 / 127.0.0.1 |

## 后端接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | 存活探针，返回 `{"ok":true,"version":"1.12.0"}`；版本号用来识别「连到了旧进程」 |
| GET | `/api/capabilities` | 环境探测（前端状态灯用） |
| POST | `/api/train/yolo` | 启动 YOLO 训练 |
| POST | `/api/train/anomalib` | 启动 Anomalib 训练 |
| POST | `/api/train/{yolo\|anomalib}/check-env` | 环境自检（开训前查缺依赖） |
| GET | `/api/runs` | 任务列表 |
| GET | `/api/runs/{run_id}` | 单任务状态 |
| POST | `/api/runs/{run_id}/stop` | 停止任务 |
| WS | `/ws/runs/{run_id}` | 事件流（训练日志 / 进度） |
| GET | `/api/browse?path=` | 目录浏览（Web 无法弹系统选择框） |
| POST | `/api/infer/yolo` | YOLO 推理 |
| POST | `/api/infer/anomalib` | Anomalib 推理 |
| GET | `/api/models` | 扫描训练产物（引擎 + ONNX + 标签 + metadata） |
| GET | `/api/infer/cache` | 查看常驻引擎句柄（调试显存用） |
| GET | `/api/engine/probe?onnx=` | 探测 ONNX 输入/输出形状，判断是否可用 |
| POST | `/api/engine/build` | 用 trtexec 把 ONNX 编译成引擎 |
| POST | `/api/dataset/scan` | 扫描数据集并按比例返回划分预览（不落盘） |
| POST | `/api/dataset/export` | 按比例导出 YOLO / COCO / VOC |
| GET | `/api/annotation/project?path=` | 读取 `.tfanno`，返回工程 + `rev` / `mtime_ns` / 统计；文件不存在返回空白骨架 |
| POST | `/api/annotation/project` | 保存工程，校验 `rev` 与 `stamp`，冲突返回 **409** 并附磁盘真实状态 |
| GET | `/api/annotation/image?path=` | 图片流（支持中文与空格路径），MIME 按扩展名推断 |
| GET | `/api/annotation/image-meta?path=` | 探测单张图片宽高，供「添加图片」填字段 |
| POST | `/api/annotation/scan-images` | 递归扫描目录里的图片，返回路径与尺寸 |
| POST | `/api/annotation/export` | 把**内存中**的工程导出 YOLO / COCO / VOC |
| POST | `/api/annotation/autolabel` | 本地引擎（必要时 + 外部服务）生成草稿，返回 `shapes` / `warnings` |

`POST /api/annotation/project` 的请求体：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `path` | str | 是 | `.tfanno` 绝对路径，必须在白名单根内 |
| `project` | object | 是 | 完整工程 JSON |
| `rev` | int | 是 | 客户端基于的修订号；新建传 `-1` |
| `stamp` | object | 否 | `{"mtime_ns":"...","size":123}`，识别「保存前文件被别人改过」 |
| `force` | bool | 否 | `true` = 强制覆盖，跳过两项校验（冲突弹窗的「强制覆盖」） |

`POST /api/annotation/autolabel` 的请求体：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `image` | str | 必填 | 当前图片绝对路径（可在工程内，也可在工程外） |
| `task` | str | `detect` | `detect` / `segment` / `pose` / `obb` |
| `engine` | str | 必填 | `.engine` 路径；主通道必填 |
| `labels` | str | 空 | `labels_*.json`，决定草稿的 `label` |
| `conf` / `nms` | float | 0.25 / 0.45 | 与推理页一致，进入句柄缓存键 |
| `service` | str | 空 | 外部服务根地址（如 `http://127.0.0.1:8500`）；空 = 只走主通道 |
| `prompts` | object | 空 | 透传给外部服务的提示（`text` / `points` / `model`） |
| `timeout` | float | 20.0 | 外部服务超时（秒），范围 `1~300` |

`POST /api/train/yolo` 的增强字段（全部可选，留空即用 Ultralytics 默认）：
`hsv_h` `hsv_s` `hsv_v` `degrees` `translate` `scale` `shear` `perspective`
`flipud` `fliplr` `bgr` `mosaic` `mixup` `copy_paste` `close_mosaic`。

`POST /api/infer/yolo` 的额外字段：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `engine` | Form str | 必填 | `.engine` 路径，必须在白名单根内 |
| `onnx` | Form str | 可空 | 只在引擎加载失败时用来提示「请用 `.transd.onnx`」 |
| `labels` | Form str | 可空 | `labels_*.json` 路径，决定 `label_name` 与 `class_colors` |
| `conf` / `nms` | Form float | 0.25 / 0.45 | 进入句柄缓存键，改了就重建句柄 |
| `draw` | Form bool | true | 是否额外产出 `annotated_png`（仅用于导出） |

`POST /api/infer/anomalib` 的额外字段：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `engine` | Form str | 必填 | `.engine` 路径 |
| `metadata` | Form str | 必填 | `metadata.json` 路径 |
| `image_threshold` | Form float | `-1.0` | `< 0` = 沿用 metadata；`>= 0` = 覆盖 |
| `pixel_threshold` | Form float | `-1.0` | 同上 |
| `heatmap_colormap` | Form str | `turbo` | OpenCV 色带名，非法名回退默认 |

> `ai_create` 的阈值是**原地写入**内部 `meta_`，所以 `ai_get_metadata` 返回的是覆盖后的值。
> 想拿文件里的原值，必须自己读 `metadata.json`——`infer.read_metadata_thresholds()` 就是干这个的。
> 响应里的 `thresholds.{image,pixel}` 同时给出 `effective`、`metadata`、`overridden` 三个字段。


## 代码结构

```
6-web/
├── backend/
│   ├── app.py            FastAPI 路由
│   ├── config.py         路径推导、环境变量与白名单（allowed_roots / is_path_allowed）
│   ├── runners.py        子进程管理 + TRTFORGE_JSON 事件解析 + 广播
│   ├── infer.py          ctypes 调 DLL（结构体定义、LRU 句柄缓存、热能图渲染）
│   ├── labels.py         标签文件解析 + 统一定义类别配色（后端绘制与前端共用）
│   ├── models.py         训练产物扫描（引擎配对 ONNX / 标签 / metadata / 任务类型）
│   ├── dataset.py        数据集扫描 / 固定哈希划分 / YOLO·COCO·VOC 导出
│   ├── annotation.py     `.tfanno` 读写与规范化（2 点矩形 / 关键点单对象 / rev 乐观锁）
│   ├── autolabel.py      自动标注双通道编排（本地引擎 + 可选外部 HTTP 服务）+ 草稿清洗
│   └── tools/
│       ├── abi_probe.cpp       打印 C 侧结构体布局，用于和 Python 对齐比对
│       ├── smoke_test.py       端到端冒烟测试（HTTP + WebSocket + 真实子进程）
│       ├── test_dataset_workbench.py  数据集工作台验收（划分 / 导出 / 错误路径）
│       ├── test_infer_params.py       推理控制台验收（阈值 / 标签 / 配色 / LRU / 扫描 / 热能图）
│       ├── test_annotation_project.py 标注工程验收（往返 / 冲突 / 回退 / 导出回读）
│       └── test_exit_codes.py  退出码语义化测试（exit 2/130 的补事件逻辑）
├── frontend/
│   ├── index.html        单页，六个页签
│   ├── css/style.css     设计令牌 + 组件样式（令牌之外无硬编码颜色）
│   ├── js/
│   │   ├── api.js        HTTP 封装
│   │   ├── ws.js         WebSocket（含断线重连）
│   │   ├── viewer.js     图像查看器：光标锚点缩放 / 拖动平移 / 多窗口容器 / 透明度
│   │   ├── overlay.js    检出框 SVG 叠加层：逆缩放标签、低分淡出、行高亮联动
│   │   ├── train.js      训练页
│   │   ├── engine.js     引擎构建页（日志流 + 探测预警）
│   │   ├── dataset.js    数据集页（扫描预览 + 导出）
│   │   ├── annotate.js   标注工作台（Canvas 2D 六形状 / 缩放平移 / 吸附 / 撤销重做 / 自动保存 / 草稿层 / 导出）
│   │   ├── infer.js      推理控制台（模型扫描 / 参数联动 / 表格 / 直方图 / 导出）
│   │   └── main.js       页签 / 环境状态 / 目录浏览
│   └── tests/            零依赖前端单测（node 直跑）
│       ├── test_viewer_math.mjs  缩放锚点、坐标互逆、滚轮归一化
│       └── test_contrast.mjs     设计令牌对比度核算（WCAG AA）
└── runtime/              上传临时文件与任务日志（已 gitignore）
```

## 验证

```powershell
# 服务已启动后
D:\Anaconda3\envs\dpenv\python.exe 6-web\backend\tools\smoke_test.py

# 退出码处理（不需要服务，独立运行）
D:\Anaconda3\envs\dpenv\python.exe 6-web\backend\tools\test_exit_codes.py

# 推理控制台（不需要服务、不需要 DLL，全部用假引擎桩）
D:\Anaconda3\envs\dpenv\python.exe 6-web\backend\tools\test_infer_params.py

# 标注工程数据层（不需要服务、不需要 DLL）
D:\Anaconda3\envs\dpenv\python.exe 6-web\backend\tools\test_annotation_project.py

# 前端单测（不需要服务，有 node 就行）
node 6-web\frontend\tests\test_viewer_math.mjs
node 6-web\frontend\tests\test_contrast.mjs
```

2026-10-29 实测结果：`smoke_test` **13 项全通过**（能力探测、参数校验、越权拦截、
真实子进程自检、WebSocket 事件流）；`test_exit_codes` **6 项全通过**。

2026-10-31 前端实测结果：`test_viewer_math` **37 条断言全通过**；
`test_contrast` **23 项全通过**（含把占位符从 3.07:1 调到 4.84:1 的一次修正）。
浏览器端另外实测了：滚轮缩放不带动页面、拖出边界不中断、双击复位、开关跨会话保留、
三窗口在 1920 下同排等宽、四窗口转横向滚动、760 窄屏单列无溢出。

2026-10-31 推理控制台实测结果：

| 脚本 | 结果 |
| --- | --- |
| `test_infer_params.py` | **71 / 71 通过** |
| `test_annotation_project.py` | **138 / 138 通过** |
| `test_augment_params.py` | 26 / 26 通过 |
| `test_dataset_workbench.py` | 20 / 20 通过 |
| `test_exit_codes.py` | 6 / 6 通过 |
| `test_onnx_fallback.py` | 36 / 36 通过 |

真实产物联调（浏览器 Playwright 驱动）：

- YOLO：PCB `01_missing_hole_01.jpg` + `best.transd_fp16.engine`，类别名显示为
  `missing_hole`（不再是 `cls0`），框线在 18%~409% 倍率下均为 1px，
  点表格行能高亮对应框，JSON/CSV 导出与屏幕数值逐项一致。
- Anomalib：`bottle.png` + `PatchCore` / `Padim` 引擎，四个窗口正常；
  把图像阈值从 metadata 的 `0.5` 覆盖成 `0.8` 后判定由 NG 翻为 OK，
  且页面同时显示「设定 0.8 / 原值 0.5 / 已覆盖」。
- LRU：连续 8 组不同 `conf` 推理，`GET /api/infer/cache` 显示常驻句柄恒为 4；
  回头用旧参数命中缓存，`rebuilt=false`，无引擎重建。

### ctypes 结构体对齐

跨语言传结构体如果对齐不一致会静默读错内存。`tools/abi_probe.cpp` 用 MSVC 打印
C 侧 `sizeof` 与 `offsetof`，与 Python `ctypes.sizeof` / `.offset` 比对，当前**逐项完全一致**：

| 结构体 | 大小 | 关键偏移 |
| --- | --- | --- |
| `yi_box` | 96 | `mask_data=48`、`kpts=64`、`is_obb=72`、`angle=92` |
| `yi_frame_stat` | 48 | — |
| `ai_metadata` | 192 | `image_threshold=160`、`input_h=184` |
| `ai_result` | 104 | `overlay_rgb=24`、`mask_rgb=48`、`raw_score=88` |
| `ai_frame_stat` | 40 | — |

## 常见问题

**顶部写「N 项待处理」**

看「环境说明」页的 JSON。常见原因：没装 `python-multipart`、launcher 路径不对、
DLL 没编、`OPENCV_DIR` 没设。

**推理接口返回 503**

DLL 没加载成功。两个必查点：

1. `OPENCV_DIR` 与 `CUDA_PATH` 是否指向真实目录；
2. Python 3.8+ **不用 PATH 找依赖 DLL**。本项目在 `infer.py` 里用
   `os.add_dll_directory` 显式加目录，如果你绕过后端直接写脚本，也要自己加。

**编译 yolo_infer 报 `STL1002: Unexpected compiler version, expected CUDA 12.4 or newer`**

MSVC 14.44 的 STL 头文件硬性拒绝 CUDA < 12.4。仓库里的 CMakeLists 虽然写了绕过宏，
但 CMake 在**探测 CUDA 编译器**那一步不会传 `CMAKE_CUDA_FLAGS`，所以宏没生效。

有效解法是环境变量（能覆盖所有 nvcc 调用，包括探测阶段）：

```powershell
$env:NVCC_PREPEND_FLAGS = "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH --allow-unsupported-compiler"
```

根治方案是升级到 CUDA 12.4+ 或换 VS2019 工具集。

**训练日志不出来**

事件只在 launcher 打印 `TRTFORGE_JSON {...}` 前缀的行时被识别，其他行走普通日志。
若完全没输出，检查 launcher 是否以无缓冲方式运行（后端已设 `PYTHONUNBUFFERED=1`）。

**只看到「退出码 2」，没有具体原因**

这是 argparse 的行为：参数不合法时直接打 usage 并 `exit(2)`，**不会**发 `error` 事件。
后端已补一层转换（`runners.py` 的 `_synthetic_error`），会额外推送一条
`ARGUMENT_ERROR` 事件并从日志尾部附上 usage 片段。若仍看不出，去
`6-web/runtime/logs/<run_id>.log` 看完整输出。
退出码 130 同理会被翻译成 `CANCELLED`。

**指标区不显示损失 / GPU 利用率**

已修正。原因是字段名与 launcher 实际结构不一致：`epoch_summary` 的
`losses`/`metrics`/`lr` 都是**字典**而非标量，GPU 利用率字段是
`utilization_gpu_pct` 而非 `utilization`。若你自己加事件渲染，请以
`1-python-yolo/launcher/tf_yolo_launcher.py` 里的 `emit` 调用为准。

**关闭浏览器后训练还在跑？**

这是设计行为。任务是后端子进程，与浏览器无关。要停就点页面「停止」，或调
`POST /api/runs/{run_id}/stop`。

**改了代码但页面行为没变**

**先确认 8060 上跑的是哪个进程。** 起服务时没关掉旧进程，新进程会因端口占用
直接退出，而浏览器连的还是几个小时前的旧代码——现象是"改了像没改"，甚至
日志里能看到已经修掉的旧异常。

```powershell
# 1) 看谁占着端口
Get-NetTCPConnection -LocalPort 8060 -State Listen | Select-Object LocalPort,OwningProcess

# 2) 用 PID 查它的启动时间与完整命令行；启动时间早于本次修改 = 旧进程
Get-CimInstance Win32_Process -Filter "ProcessId=<上一步的PID>" |
  Select-Object ProcessId,CreationDate,CommandLine

# 3) 杀掉旧进程（多个端口一起清）
Stop-Process -Id <PID> -Force

# 4) 用当前代码重启；开发期建议带 --reload，改完自动生效
D:\Anaconda3\envs\dpenv\python.exe -m uvicorn app:app --app-dir 6-web\backend `
  --host 127.0.0.1 --port 8060 --reload
```

判断标准只有一条：**`CreationDate` 必须晚于你最后一次改后端代码的时间。**
`--reload` 只监听 `.py`，改了前端 HTML/CSS/JS 仍需浏览器 `Ctrl+F5`。

更快的判断办法：**看顶栏标题下方的版本号徽标**（`v1.12.0` 这种，来自 `/api/health`）。
后端代码一改，`app.py` 里的 `APP_VERSION` 就要跟着加，页面上的徽标才会变。
徽标数字没变 = 页面连的还是旧进程，这时候改什么都不会生效。

```powershell
# 一条命令确认后端版本
Invoke-WebRequest http://127.0.0.1:8060/api/health -UseBasicParsing | Select-Object -ExpandProperty Content
# 期望输出类似 {"ok":true,"version":"1.12.0"}
```

## 与 Qt 版的关系

`4-qt-studio` 目前仍保留，作为**迁移参考实现**（参数拼装规则、数据集划分算法、
事件解析逻辑都在里面）。确认 Web 版功能覆盖后再归档到 `5-reference`。

详见提案：`openspec/changes/add-web-studio/`。
