# Design: 批量 / 视频 / 摄像头的执行模型

## 核心问题：为什么不是"一个接口三种模式"

三种形态的**时间尺度**和**失败语义**完全不同：

| | 目录批量 | 视频 | 摄像头 |
| --- | --- | --- | --- |
| 单次耗时 | 秒~小时 | 秒~小时 | 毫秒 |
| 失败影响 | 坏一张不该毁整批 | 坏一帧不该毁整段 | 坏一帧无关紧要 |
| 用户诉求 | 能停、能看进度、要报表 | 能停、要能播的产物 | 延迟低、不落盘 |
| 结论 | 子进程任务 | 子进程任务 | 同步单帧接口 |

硬把它们塞进一个接口，要么实时性被任务队列拖死，要么批量任务的取消/进度无人实现。

## 决策一：批量与视频走子进程，复用 `runners.py`

`runners.py` 已有：子进程管理、`TRTFORGE_JSON` 单行事件解析、WebSocket 广播、
日志落盘、`terminate` 取消。批量脚本只要把进度打成单行 JSON 就能进这套管道。

事件类型（与训练同构，前端不用学新协议）：

```json
{"type":"started","total":3,"kind":"batch"}
{"type":"progress","done":2,"total":3,"pct":66.7,"file":"plane2.jpg"}
{"type":"warning","message":"mask_07.jpg: decode failed"}
{"type":"summary","ok":2,"error":1,"output":"...","elapsed":0.7}
```

**取消语义**：`terminate` 之后**已写出的文件保留**、zip 不产出（`.part` 不重命名）。
理由：用户按停止通常是"够了，别再跑了"，不是"把已跑的也删掉"。

## 决策二：摄像头走同步单帧接口，且只回 JSON

`POST /api/infer/frame` 接受一帧 JPEG/PNG，返回：

```json
{"kind":"yolo","task":"detect","width":1280,"height":720,
 "boxes":[{"x1":..,"y1":..,"x2":..,"y2":..,"score":0.87,"cls":3,"label_name":"missing_hole"}],
 "class_colors":{"3":"#22d3ee"},"elapsed_ms":41.2}
```

三个"不"：

- **不回 base64 标注图**：一帧 1280×720 PNG base64 约 1~2 MB，回传解码又要吃掉主线程，
  实测这条路 8 fps 都到不了。只回坐标，带宽降到 KB 级。
- **不建运行记录**：不写 `runtime`，不留日志——这是"看一眼"，不是"跑任务"。
- **不排队**：前端 `setInterval` + 在途标志位，上一帧没回来就跳过本次 tick。
  宁可掉帧也不累积延迟。

## 决策三：产物逻辑抽到 `inferjobs.py`

批量与视频共用四件事，抽出来是为了让两边行为**不可能不一致**：

1. **文件名净化**：`板 1(2).png` → `00001_板_1_2.jpg`。Windows 不允许 `\ / : * ? " < > |`，
   中文与空格保留（用户要看得懂），序号前缀保证顺序与去重。
2. **CSV 编码**：`utf-8-sig` + `lineterminator="\n"`。不带 BOM 时 Excel 打开中文类别名是乱码；
   `\r\n` 在部分工具里会多出空行。这两点都有测试守着。
3. **原子 zip**：先写 `annotated.zip.part`，成功后再改名，避免"下载到半个包"。
4. **路径沙箱**：`resolve_artifact` 只允许 `runtime/infer` 之下的路径，且解析后再比较
   （防 `..` 与符号链接），越界一律 404，不泄露"文件是否存在"。

## 决策四：编码器可用性用"写一帧再看大小"验证

`cv2.VideoWriter.isOpened()` 在缺少 `libopenh264` 的机器上对 `avc1` 仍然返回 `True`，
写出的是 0 字节文件——错误被伪装成成功。因此：

```python
for codec in ("avc1", "H264", "mp4v"):
    writer = cv2.VideoWriter(tmp, fourcc(codec), fps, size)
    writer.write(np.zeros((h, w, 3), np.uint8))
    writer.release()
    if os.path.getsize(tmp) > 0:
        return codec
```

成本是写一帧（毫秒级），收益是"浏览器能直接播"这件事不再靠猜。

## 决策五：抽帧必须同时改输出帧率

步长 N > 1 时只处理每 N 帧，但**输出帧率写 `源帧率 ÷ N`**：

- 不改帧率 → 30 帧的 3 秒片变成 1 秒的"快进"，产物时长错、动作对不上。
- 改帧率 → 播放时长与原片一致，时间轴可直接对照。

这条不是显示细节，是"产物能不能用"的前提，所以有专门的测试断言时长不变。

## 风险与已知限制

- **显存**：批量任务与页面单张推理共用 DLL 句柄缓存，但同时跑会各建句柄。批量任务结束进程即释放。
- **大目录**：`results.json` 含每框时体积与框数成正比；几百张图建议关掉该项，只留 CSV。
- **视频音轨**：产物**无音轨**（OpenCV 不支持）。需要保留原声要另行用 ffmpeg 合流，属已知限制。
- **摄像头权限**：`getUserMedia` 需要安全上下文（localhost 或 HTTPS），局域网 HTTP 访问时浏览器会拒绝。
