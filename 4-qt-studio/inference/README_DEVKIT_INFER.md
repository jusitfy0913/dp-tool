# 本目录仅提供预编译推理库

本开发包不包含封装后的 TensorRT / OCR 推理实现源码。

请使用：

- 头文件：`include/*_infer_c.h`
- 动态库：`4-qt-studio/bin/*_infer.dll`
- 导入库：`4-qt-studio/bin/*_infer.lib`

主工程 CMake 已改为导入上述预编译库，不再编译推理源码。
