# Change: 加 CI 与测试总入口（工具链改动，不影响功能规格）

## Why

v1.15 开发过程中漏掉了一个参数拼装错误（`--simplify true` 被 argparse 判成多余参数），
本地 12 个测试脚本全部通过也发现不了——因为那一条根本没人跑。根因不是缺测试，而是
**缺一个"每次都跑"的机制**：

- 444 项断言只能靠人肉逐个执行，改一行代码忘了跑某个脚本，问题要到用户点开页面才暴露；
- 新增脚本（如 `test_transd_onnx.py`）不会自动进入任何流水线；
- 前端两个 `.mjs` 单测同样靠记忆。

## What Changes

- 新增 `6-web/backend/tools/run_all_tests.py`：扫描同目录 `test_*.py` 自动收录，串行执行，
  逐条打印耗时与通过数，任一退出码非 0 即整体失败。支持按关键字过滤（`run_all_tests.py transd`）。
- 新增 `.github/workflows/tests.yml`：
  - `backend` job：Windows + Python 3.11，装 `requirements.txt` + onnx + pillow，跑总入口；
  - `frontend` job：Node 20，跑两个 `.mjs` 单测 + 所有 `js/*.js` 的 `node --check`。
- `requirements.txt` 补 `onnx>=1.14` 与 `pillow>=9`：前者是 `.transd.onnx` 生成的实际依赖，
  后者是数据集概览读图像头的依赖，之前只存在于开发机环境里、没有声明。

## Impact

- Affected specs: 无（工具链改动，不改变任何用户可见行为）
- Affected code:
  - 新增 `6-web/backend/tools/run_all_tests.py`、`.github/workflows/tests.yml`
  - 修改 `6-web/backend/requirements.txt`
  - 文档：`6-web/README.md` 的「验证」一节

## Non-Goals

- 不引入 pytest/tox：现有脚本是自包含的 `check()` 风格，改造成本大于收益。
- 不在 CI 里跑真实推理：没有 GPU、也没有 TensorRT DLL，保持纯逻辑测试。
- 不把 `openspec validate` 接进 CI：CLI 是 npm 全局包，版本漂移会让流水线无故变红。

## 验收

- 本地 `python 6-web/backend/tools/run_all_tests.py` → 12 个脚本全通过、返回 0。
- 故意把某个断言改错 → 返回 1 且输出尾部直接给出失败行。
- CI 工作流 YAML 可被解析，两个 job 的 steps 数量与设计一致。
