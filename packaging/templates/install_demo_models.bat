@echo off
rem Install the demo models into the TRTForgeWeb install folder.
rem Keep this file ASCII-only: cmd.exe reads .bat with the system ANSI code page,
rem and non-ASCII bytes here would garble on some machines.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_demo_models.ps1"
if errorlevel 1 pause
