@echo off
rem ===========================================================================
rem  TRTForge Web Studio - quick start (ASCII only, see note below)
rem
rem  Usage:
rem    start_web.bat                  -> http://127.0.0.1:8060 + open browser
rem    start_web.bat 8070             -> another port
rem    start_web.bat 8070 nobrowser   -> do not open the browser
rem    start_web.bat 8060 reload      -> auto reload when backend code changes
rem
rem  NOTE: this file is intentionally ASCII-only. cmd.exe mis-parses UTF-8
rem  batch files containing multi-byte characters (a well known codepage bug),
rem  which silently corrupts lines. All Chinese messages live in
rem  run_web_studio.ps1, which PowerShell reads correctly as UTF-8.
rem ===========================================================================
setlocal EnableExtensions
title TRTForge Web Studio

set "PS1=%~dp0run_web_studio.ps1"
if not exist "%PS1%" (
    echo [ERROR] run_web_studio.ps1 not found next to this script.
    pause
    exit /b 1
)

set "ARGS="
set "OPEN=-OpenBrowser"
set "RELOAD="

:parse
if "%~1"=="" goto launch
if /i "%~1"=="nobrowser" (
    set "OPEN="
    shift
    goto parse
)
if /i "%~1"=="reload" (
    set "RELOAD=-Reload"
    shift
    goto parse
)
set "ARGS=%ARGS% -Port %~1"
shift
goto parse

:launch
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %ARGS% %OPEN% %RELOAD%
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" (
    echo.
    echo [ERROR] startup failed, exit code %CODE%.
    pause
)
exit /b %CODE%
