@echo off
rem Double click this file to start TRTForge Web Studio.
rem The real logic lives in scripts\start_web.bat (ASCII-only launcher) and
rem scripts\run_web_studio.ps1 (Chinese messages, port checks, browser opening).
rem Arguments are forwarded as-is, e.g.  启动Web端.bat 8070
call "%~dp0scripts\start_web.bat" %*
