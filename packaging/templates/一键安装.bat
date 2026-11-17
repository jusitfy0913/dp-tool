@echo off
rem ASCII only. cmd.exe reads .bat with the system code page.
rem Double-click this once: install the bundled training environment,
rem put a desktop shortcut, and open TRTForge Web.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0一键安装.ps1"
if errorlevel 1 pause
exit /b %ERRORLEVEL%
