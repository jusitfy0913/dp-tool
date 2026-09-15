@echo off
setlocal EnableExtensions
set "LOGDIR=D:\test1\deeplearning\third_party\Paddle\build_logs"
mkdir "%LOGDIR%" 2>nul
set "LOG=%LOGDIR%\paddle_infer_build.log"

echo [%date% %time%] === Paddle Inference: build + sync (configure already done) === >> "%LOG%"

echo [%date% %time%] Step 2: build inference_lib_dist >> "%LOG%"
call "D:\test1\deeplearning\third_party\Paddle\build_infer_trt_release.bat" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [%date% %time%] BUILD FAILED >> "%LOG%"
  exit /b 2
)

set "INSTALL=D:\test1\deeplearning\third_party\Paddle\build_infer_trt\paddle_inference_install_dir"
if not exist "%INSTALL%\paddle\lib\paddle_inference.dll" (
  echo [%date% %time%] INSTALL DIR MISSING paddle_inference.dll >> "%LOG%"
  exit /b 3
)

echo [%date% %time%] Step 3: sync to trtexec/gui/bin >> "%LOG%"
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\test1\deeplearning\scripts\sync_paddle_inference_to_gui_bin.ps1" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [%date% %time%] SYNC FAILED >> "%LOG%"
  exit /b 4
)

echo [%date% %time%] ALL DONE >> "%LOG%"
exit /b 0
