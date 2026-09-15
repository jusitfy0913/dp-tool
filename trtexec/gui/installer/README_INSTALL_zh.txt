DefectVision AI Studio（TRTForge）安装说明
========================================

本安装程序会将已编译好的程序文件安装到您选择的目录（默认在 Program Files 下）。

【系统要求】
- 64 位 Windows 10 或更高版本
- 支持 CUDA 的 NVIDIA 显卡驱动（使用 TensorRT 引擎构建与推理时需要）
- Visual Studio 2015–2022 可再发行组件（x64）。若启动提示缺少 VCRUNTIME 等 DLL，请从微软官网安装 “Microsoft Visual C++ Redistributable for Visual Studio 2015–2022 (x64)”。

【与源码编译目录的关系】
安装包内容来自开发阶段的 trtexec\gui\bin 目录（内含 TRTForge.exe、Qt 运行库、推理 DLL、以及构建时复制到 bin 的 TensorRT/CUDA/cuDNN/OpenCV 等 DLL）。请确保在制包前已在您的环境完成一次 Release 编译，否则可能缺少运行库。

【卸载】
- 使用 Windows「设置」→「应用」中找到本软件卸载，或从开始菜单卸载项执行。
