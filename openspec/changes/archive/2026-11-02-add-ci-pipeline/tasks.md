# Tasks: 加 CI 与测试总入口

## 1. 总入口

- [x] 1.1 新增 `6-web/backend/tools/run_all_tests.py`：扫描 `test_*.py`、串行执行、单脚本 600s 超时
- [x] 1.2 失败时附上 stdout/stderr 尾部 12 行，CI 日志里能直接看到哪条断言挂了
- [x] 1.3 支持按关键字过滤（`run_all_tests.py transd`），便于本地只跑一块
- [x] 1.4 本地实跑：12 个脚本、444 项断言全通过，总耗时约 23 秒

## 2. CI

- [x] 2.1 新增 `.github/workflows/tests.yml`：`backend`（Windows + py3.11）+ `frontend`（Node 20）
- [x] 2.2 backend job 用 `requirements.txt` + `onnx` + `pillow`，不装 torch / 不碰 GPU
- [x] 2.3 frontend job 跑 `test_contrast.mjs` / `test_viewer_math.mjs`，并对所有 `js/*.js` 做 `node --check`
- [x] 2.4 用 YAML 解析确认两个 job 结构正确

## 3. 依赖声明

- [x] 3.1 `requirements.txt` 补 `onnx>=1.14`（`.transd.onnx` 生成）与 `pillow>=9`（数据集概览读图像头）
- [x] 3.2 说明为什么不在 CI 装 torch：所有测试用假引擎桩

## 4. 文档

- [x] 4.1 `6-web/README.md`「验证」一节改成「一条命令跑完」，补 CI 说明与代码结构树
- [x] 4.2 根 `readme.md` v1.15 小节补「配套的流程改进」
