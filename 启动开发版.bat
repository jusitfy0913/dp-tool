@echo off
rem ===========================================================================
rem  TRTForge Web Studio - start from source, in a WebView2 window (dev only)
rem
rem  ASCII only on purpose: cmd.exe mis-parses UTF-8 .bat files with multi-byte
rem  characters on some code pages. All Chinese messages live on the Python side
rem  (launcher.py shows message boxes and writes runtime\logs\launcher.log).
rem
rem  Why TRTFORGE_HOME points at 6-web\.devroot instead of the repo root:
rem  launcher.py treats its root as an "install root" and derives
rem  frontend\ bin\ launchers\ models\ from it. The repo root has none of those,
rem  which would give a blank page. .devroot only holds junctions to the real
rem  folders plus a config.json, so the install layout is satisfied while the
rem  code and frontend still come from the source tree.
rem
rem  .devroot is created on first run and is machine-local (junctions use
rem  absolute paths), so it is git-ignored. Delete it any time; this file
rem  recreates it.
rem
rem  Normal users should double-click app\TRTForgeWeb.exe in the packaged folder.
rem ===========================================================================
setlocal EnableExtensions

set "REPO=%~dp0"
set "DEVROOT=%REPO%6-web\.devroot"
set "PYW=D:\Anaconda3\envs\dpenv\pythonw.exe"
if not exist "%PYW%" set "PYW=pythonw.exe"

if not exist "%REPO%6-web\backend\launcher.py" (
    echo [ERROR] 6-web\backend\launcher.py not found next to this file.
    pause
    exit /b 1
)

rem ---------------------------------------------------------------------------
rem  Install-layout shim. Only rebuild when something is actually missing, so a
rem  user-edited config.json (custom port, data_roots) is never overwritten.
rem ---------------------------------------------------------------------------
if not exist "%DEVROOT%\config.json" (
    if not exist "%DEVROOT%" mkdir "%DEVROOT%"
    > "%DEVROOT%\config.json" echo { "port": 8060, "host": "127.0.0.1", "python": "", "data_roots": [], "ui": "window" }
)
if not exist "%DEVROOT%\frontend" mklink /J "%DEVROOT%\frontend" "%REPO%6-web\frontend" >nul
if not exist "%DEVROOT%\bin" mklink /J "%DEVROOT%\bin" "%REPO%4-qt-studio\bin" >nul
if not exist "%DEVROOT%\models" (
    if not exist "%REPO%6-web\models" mkdir "%REPO%6-web\models"
    mklink /J "%DEVROOT%\models" "%REPO%6-web\models" >nul
)
if not exist "%DEVROOT%\runtime" mkdir "%DEVROOT%\runtime"

rem Launchers: hardlinks to the source scripts, so editing them in the repo is
rem what the dev instance actually runs (the packaged launchers folder is a
rem stale copy from the last packaging run).
if not exist "%DEVROOT%\launchers" mkdir "%DEVROOT%\launchers"
if not exist "%DEVROOT%\launchers\tf_yolo_launcher.py" mklink /H "%DEVROOT%\launchers\tf_yolo_launcher.py" "%REPO%1-python-yolo\launcher\tf_yolo_launcher.py" >nul
if not exist "%DEVROOT%\launchers\tf_anomalib_launcher.py" mklink /H "%DEVROOT%\launchers\tf_anomalib_launcher.py" "%REPO%2-python-anomaly\launcher\tf_anomalib_launcher.py" >nul
if not exist "%DEVROOT%\launchers\tf_anomalydino_launcher.py" mklink /H "%DEVROOT%\launchers\tf_anomalydino_launcher.py" "%REPO%2-python-anomaly\launcher\tf_anomalydino_launcher.py" >nul
if not exist "%DEVROOT%\launchers\anomalydino_core.py" mklink /H "%DEVROOT%\launchers\anomalydino_core.py" "%REPO%2-python-anomaly\launcher\anomalydino_core.py" >nul
if not exist "%DEVROOT%\launchers\yolov8n.pt" mklink /H "%DEVROOT%\launchers\yolov8n.pt" "%REPO%1-python-yolo\launcher\yolov8n.pt" >nul

rem ---------------------------------------------------------------------------
rem  Start. pythonw.exe = no console window next to the app window; everything
rem  that matters lands in 6-web\.devroot\runtime\logs\launcher.log.
rem ---------------------------------------------------------------------------
set "TRTFORGE_HOME=%DEVROOT%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

pushd "%REPO%6-web\backend"
start "" "%PYW%" launcher.py %*
popd

exit /b 0
