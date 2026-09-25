@echo off
REM Double-click to start PixelPaws without the desktop shortcut. Runs
REM installer\run.bat, which activates the pixelpaws environment and logs the
REM launch to %LOCALAPPDATA%\PixelPaws\run.log.
call "%~dp0installer\run.bat"
