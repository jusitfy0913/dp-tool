@echo off
rem Kept only so old shortcuts on user machines keep working.
rem It used to carry its own port/address settings (PORT / WEBHOST at the top),
rem which is how users ended up with two different ports to remember.
rem Now it is the same behaviour as start_web.bat: config.json is the only place
rem port and host are set.

call "%~dp0start_web.bat" %*
exit /b %ERRORLEVEL%
