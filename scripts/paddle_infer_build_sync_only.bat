@echo off
rem ===================================================================
rem TRTForge - Paddle Inference: build only (skip CMake configure) + sync
rem Use this after the first configure, when only sources changed.
rem
rem All paths are derived from this script location (no hard-coded
rem machine paths):
rem   REPO_ROOT        = parent of this scripts directory
rem   PADDLE_THIRD_PARTY_ROOT = %REPO_ROOT%\third_party
rem   PADDLE_WORK_ROOT = %PADDLE_THIRD_PARTY_ROOT%\Paddle
rem   PADDLE_INSTALL_DIR = %PADDLE_WORK_ROOT%\build_infer_trt\paddle_inference_install_dir
rem
rem Override any of them with the environment variable of the same name.
rem Sync target: %REPO_ROOT%\4-qt-studio\bin
rem Encoding: UTF-8 (ASCII-only on purpose so cmd.exe never garbles it)
rem ===================================================================
setlocal EnableExtensions

rem REPO_ROOT: %~dp0 is the scripts dir with a trailing backslash
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"

if not defined PADDLE_THIRD_PARTY_ROOT set "PADDLE_THIRD_PARTY_ROOT=%REPO_ROOT%\third_party"
if not defined PADDLE_WORK_ROOT set "PADDLE_WORK_ROOT=%PADDLE_THIRD_PARTY_ROOT%\Paddle"
if not defined PADDLE_LOG_DIR set "PADDLE_LOG_DIR=%PADDLE_WORK_ROOT%\build_logs"
if not defined PADDLE_INSTALL_DIR set "PADDLE_INSTALL_DIR=%PADDLE_WORK_ROOT%\build_infer_trt\paddle_inference_install_dir"

set "LOG=%PADDLE_LOG_DIR%\paddle_infer_build.log"
mkdir "%PADDLE_LOG_DIR%" 2>nul

echo [%date% %time%] === Paddle Inference: build + sync (configure already done) === >> "%LOG%"
echo [%date% %time%] REPO_ROOT=%REPO_ROOT% >> "%LOG%"
echo [%date% %time%] PADDLE_WORK_ROOT=%PADDLE_WORK_ROOT% >> "%LOG%"

echo [%date% %time%] Step 1: ninja build (SkipConfigure) >> "%LOG%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_paddle_inference_win_gpu_trt.ps1" -SkipConfigure >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [%date% %time%] BUILD FAILED >> "%LOG%"
  exit /b 2
)

if not exist "%PADDLE_INSTALL_DIR%\paddle\lib\paddle_inference.dll" (
  echo [%date% %time%] INSTALL DIR MISSING paddle_inference.dll: %PADDLE_INSTALL_DIR% >> "%LOG%"
  exit /b 3
)

echo [%date% %time%] Step 2: sync to 4-qt-studio\bin >> "%LOG%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync_paddle_inference_to_bin.ps1" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [%date% %time%] SYNC FAILED >> "%LOG%"
  exit /b 4
)

echo [%date% %time%] ALL DONE >> "%LOG%"
exit /b 0
