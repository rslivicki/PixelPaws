@echo off
REM Double-click to install PixelPaws. Runs installer\install.bat, which asks
REM where to install, builds the Python environment, and puts a shortcut on
REM the desktop. See INSTALL.txt for details and troubleshooting.
REM
REM In the release zip this file sits next to a PixelPaws\ folder that holds
REM the app; in an installed copy it sits inside the app folder itself.
if exist "%~dp0installer\install.bat" (
    call "%~dp0installer\install.bat"
) else if exist "%~dp0PixelPaws\installer\install.bat" (
    call "%~dp0PixelPaws\installer\install.bat"
) else (
    echo.
    echo   PixelPaws cannot install from here: the rest of the app is missing.
    echo.
    echo %~dp0 | findstr /i ".zip\\" >nul && (
        echo   It looks like you opened the zip and ran this file from inside it.
        echo   Windows only unpacked this one file.
        echo.
    )
    echo   1. Close this window.
    echo   2. Right-click the downloaded zip and choose Extract All, then Extract.
    echo   3. Open the extracted folder and double-click Install PixelPaws.bat there.
    echo.
    pause
)
