# Design: 全流程串联的执行模型与精度口径

## 核心问题：串联该做成「一个大任务」还是「一串小任务」

直觉方案是把整条链做成一个子进程（`run_pipeline.py`），但它在 Windows 上有两个硬伤：

| | 单进程整链 | 一串独立 run（本方案） |
| --- | --- | --- |
| 点「停止」 | `terminate` 只杀父进程，训练/trtexec 变孤儿继续占显存 | 对**当前那一个** `terminate`，干净 |
| 日志 | 混在一个流里，靠前缀区分 | 每段独立日志，可单独回看 |
| 失败定位 | 整链一个退出码 | 每段有状态、错误、耗时 |
| 复用 | 要重写参数拼装与进度解析 | 完全复用既有 `/api/train/yolo`、`/api/export/yolo`、`/api/engine/build` 的子进程逻辑 |

代价是编排层要自己维护状态机 —— 这就是 `pipeline.py` 的全部工作。

## 决策一：只做编排，不做执行

`pipeline.py` 不 spawn 任何进程，它只回答三件事：

1. 下一段跑什么（`StagePlan(name, label, command, cwd, expect_file, report_file, callable)`）；
2. 这段结束了没有（轮询 `run_manager.get(run_id).status`）；
3. 结果怎么记账（阶段状态、产物路径、耗时、精度报告）。

轮询而不是订阅：订阅是 asyncio 的，编排跑在普通 `threading.Thread` 里；
run 结束后状态就是终态，每 0.5 s 读一次比跨线程投递事件简单且不会漏。

## 决策二：元数据步骤用 callable，分钟级步骤用 run

| 阶段 | 动作 | 形态 | 原因 |
| --- | --- | --- | --- |
| train | 训练子进程 | run | 分钟~小时，要日志流与进度 |
| export | 导出 ONNX | run | 10 s~2 min，失败要能看导出日志 |
| probe | 读 ONNX 断言转置布局 | **callable** | 纯读文件，起进程多花 1~2 s 还多一份日志噪音 |
| build | trtexec | run | 1~10 min，要日志与镜像进度 |
| labels | 写 `labels_*.json` | **callable** | 纯写文件 |
| val | 引擎精度验证 | run | 分钟级，要逐张进度 |

callable 抛异常时同样把整链判 `failed`，异常信息带类型写进阶段 `error`。

## 决策三：状态机与失败语义

```
pending → running → success
                  ↘ failed      （该段失败，后段全部 skipped，整链 failed）
                  ↘ cancelled   （被停止，后段全部 skipped，整链 cancelled）
   pending → skipped            （前面失败或已停止）
```

- **不重试**：失败就停。训练 OOM 重试一次只是再烧一遍时间，还会把真实原因埋进第二次日志。
- **产物保留**：`best.pt` / `.transd.onnx` / `.engine` / `labels_*.json` 都是磁盘上的文件，
  链断了也照样能用——页面的产物表就是给人「手工接着往下跑」用的。
- **停止只停当前段**：`_mark_remaining(skipped)` 在 `_finish` 之前执行，
  保证「停止」不会留下一个既非终态又没人推进的中间段。

## 决策四：精度口径取 COCO，但只对 detect 生效

`valmetrics.py` 独立于推理实现，只吃两个数组：预测框与 GT 框。

- IoU 匹配按**置信度降序**贪心，同一 GT 只匹配一次（COCO 的 `match` 语义）；
- AP 用**全点插值**（101 点太粗，COCO 2014 之后就是全点）；
- IoU 阈值 0.50:0.95 步长 0.05，取均值得到 mAP50-95；
- 阈值比较带 `1e-9` 容差：`IoU = 8/16` 在浮点里是 `0.4999999999999999`，
  严格 `>= 0.5` 会让「恰好命中」的用例失败——这是测试里真实踩到的坑。

只对 `detect` 生效，因为 segment / pose / obb 的 GT 解码口径完全不同
（多边形 IoU、关键点 OKS、旋转框角度），套检测口径会给出**看起来很像**但完全不可比的数字。

## 决策五：持久化与重启清理

记录写成 `runtime/pipelines/{id}.json`，先写 `.json.part` 再改名（断电不会留下半个 JSON）。

启动时扫一遍存量记录，把 `status ∈ {pending, running}` 的标成：
`cancelled` + `error = "服务重启，任务已中断"`，未开始的段 `skipped`，进行中的段 `cancelled`。

不这么做的话，前端按 `status` 轮询会永远转圈，而「停止」只能拿到 409（内存里没有这条）。
同时 `GET /api/pipeline/{id}` 返回 `live` 字段：`live=false` 且状态仍是 running 时，
前端停止轮询并在提示里说明「这条记录来自上一次服务进程」。历史列表也会把磁盘记录合并进来，
否则刷新页面后历史会突然变空，看起来像数据丢了。

## 决策六：历史阶段的日志读文件，而不是复用实时流

实时日志面板（`WS /ws/runs/{id}`）只认内存里的 run。用它去看一条已结束、或上一个
服务进程留下的阶段，只会得到「未知任务」——而这恰恰是排障最需要的那一段。

所以阶段行有两个入口：该段还在跑 → `focusRun(run_id)` 走实时流；否则 →
`GET /api/pipeline/{id}/log?stage=&lines=` 直接读 `runtime/logs/{run_id}.log` 的尾部，
渲染在卡片内的 `#pipeLog`。几点实现约束：

- 只拼接 `runtime/logs/{run_id}.log` 路径，`run_id` 取自服务端记录而非客户端参数，
  且拼接结果必须直接位于 `runtime/logs` 下（`run_id` 带 `..` 也跳不出去）；
- 阶段名必须在已知阶段清单内，`lines` 夹取到 1~5000，避免一次读成百 MB；
- 从文件尾向前读，命中行数即停，不做全文读取（trtexec 日志动辄数十 MB）。

## 风险与已知限制

- **磁盘增长**：每个串联一条 JSON（几 KB），失败率高的机器上会积累；目前不做自动清理，
  需要时手工删 `runtime/pipelines/`。
- **精度验证耗时**：逐张走 DLL，1000 张 val 集约数分钟；大 val 集建议先 `split=test` 抽小批试跑。
- **fp16 构建更慢**：tactic 搜索空间不同，实测 `--fp16` 可能比 `--fp32` 慢数倍，
  且期间日志完全静默（trtexec 行为），页面靠心跳推进度。
- **模块缓存**：升级后浏览器可能仍持有旧 `js/*.js`（ES module 图缓存），
  已给 `/static/` 加 `no-cache`；极端情况仍需硬刷新。
