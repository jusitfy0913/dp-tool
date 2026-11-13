@echo off
rem ASCII only on purpose: cmd.exe parses .bat with the system code page.
rem
rem Double-click entry. The real launcher is app\TRTForgeWeb.exe (backend\launcher.py):
rem it reads config.json, sets the environment variables, checks for an existing
rem instance and opens the browser. This file only forwards TRTFORGE_HOME so the
rem exe can find its install root, and keeps the window open when something fails.
rem
rem Users normally double-click app\TRTForgeWeb.exe directly. This bat exists for
rem the cases where that is blocked (download protection, shortcut, scripted start).

setlocal
cd /d "%~dp0"
set "TRTFORGE_HOME=%~dp0"
set "EXE=%~dp0app\TRTForgeWeb.exe"

if not exist "%EXE%" (
  echo Cannot find: %EXE%
  echo This folder is not a complete TRTForgeWeb install. Do not copy a single file.
  pause
  exit /b 1
)

"%EXE%" %*
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
  echo.
  echo TRTForgeWeb exited with code %CODE%.
  echo See runtime\logs\launcher.log for details.
  pause
)
exit /b %CODE%
