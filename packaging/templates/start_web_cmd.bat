@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 没有 PowerShell 时用本文件启动。端口和地址改下面两行，不读取 config.json。
set "PORT=8060"
set "WEBHOST=127.0.0.1"

set "ROOT=%~dp0"
set "TRTFORGE_HOME=%ROOT%"
set "TRTFORGE_REPO_ROOT=%ROOT%"
set "TRTFORGE_FRONTEND_DIR=%ROOT%frontend"
set "TRTFORGE_BIN_DIR=%ROOT%bin"
set "TRTFORGE_YOLO_LAUNCHER_DIR=%ROOT%launchers"
set "TRTFORGE_ANOMALIB_LAUNCHER_DIR=%ROOT%launchers"
set "TRTFORGE_MODEL_DIR=%ROOT%models"
set "TRTFORGE_OCR_MODEL_DIR=%ROOT%models\ocr"
set "TRTFORGE_TRTEXEC=%ROOT%bin\trtexec.exe"
set "TRTFORGE_TOOL_EXE=%ROOT%app\TRTForgeWeb.exe"
set "OPENCV_BIN_DIR=%ROOT%bin"
set "CUDA_BIN_DIR=%ROOT%bin"
set "TRTFORGE_WEB_PORT=%PORT%"
set "TRTFORGE_WEB_HOST=%WEBHOST%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PATH=%ROOT%bin;%PATH%"

if exist "%ROOT%train-env\python.exe" set "TRTFORGE_PYTHON=%ROOT%train-env\python.exe"
if exist "%ROOT%app\tcl8.6" set "TCL_LIBRARY=%ROOT%app\tcl8.6"
if exist "%ROOT%app\tk8.6" set "TK_LIBRARY=%ROOT%app\tk8.6"

if not exist "%TRTFORGE_TOOL_EXE%" (
  echo 找不到程序：%TRTFORGE_TOOL_EXE%
  echo 请确认这是完整的 TRTForgeWeb 文件夹，不要只拷单个文件。
  pause
  exit /b 1
)

set "OPENHOST=%WEBHOST%"
if "%WEBHOST%"=="0.0.0.0" set "OPENHOST=127.0.0.1"

echo TRTForge Web Studio
echo   地址     : http://%OPENHOST%:%PORT%/
if defined TRTFORGE_PYTHON (
  echo   训练环境 : %TRTFORGE_PYTHON%
) else (
  echo   训练环境 : 未配置。只能推理，不能训练。
)
echo 本窗口请保持打开。关掉它，网页就停止。
echo.

start "" cmd /c "ping 127.0.0.1 -n 12 >nul & start http://%OPENHOST%:%PORT%/"
"%TRTFORGE_TOOL_EXE%"
echo.
echo 程序已退出，退出码 %ERRORLEVEL%。
pause
