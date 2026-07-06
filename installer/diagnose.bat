@echo off
REM ============================================================================
REM  PixelPaws install diagnostics — run this on the test machine after a
REM  failed install.bat to see what state things are in.
REM ============================================================================
setlocal EnableDelayedExpansion

set "PIXELPAWS_ROOT=%~dp0.."
pushd "%PIXELPAWS_ROOT%"
set "PIXELPAWS_ROOT=%CD%"
popd

echo ============================================================
echo   PixelPaws install diagnostics
echo   Run on: %DATE% %TIME%
echo ============================================================
echo.
echo PIXELPAWS_ROOT: %PIXELPAWS_ROOT%
echo USERPROFILE:    %USERPROFILE%
echo LOCALAPPDATA:   %LOCALAPPDATA%
echo TEMP:           %TEMP%
echo PATH:           %PATH:;=^

%
echo.
echo ============================================================
echo  Disk space:
echo ============================================================
wmic logicaldisk where "DriveType=3" get Caption,FreeSpace,Size /format:table 2>nul
echo.
echo ============================================================
echo  Conda installs visible:
echo ============================================================
for %%D in (
    "%LOCALAPPDATA%\PixelPaws\miniforge3"
    "%USERPROFILE%\miniforge3"
    "%USERPROFILE%\mambaforge"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%ProgramData%\miniforge3"
    "%ProgramData%\Anaconda3"
    "%ProgramData%\Miniconda3"
    "C:\miniforge3"
    "C:\Anaconda3"
    "C:\Miniconda3"
    "C:\ProgramData\Anaconda3"
) do (
    if exist %%D (
        echo   %%D
        if exist %%D\Scripts\mamba.exe echo     - has mamba.exe
        if exist %%D\Scripts\conda.exe echo     - has conda.exe
        if exist %%D\envs\pixelpaws    echo     - has pixelpaws env
    )
)
echo.
where mamba.exe 2>nul
where conda.exe 2>nul
echo.
echo ============================================================
echo  PixelPaws app data:
echo ============================================================
if exist "%LOCALAPPDATA%\PixelPaws" (
    echo   %LOCALAPPDATA%\PixelPaws exists. Contents:
    dir /B /S "%LOCALAPPDATA%\PixelPaws" 2>nul
) else (
    echo   %LOCALAPPDATA%\PixelPaws does not exist.
)
echo.
echo ============================================================
echo  Last 60 lines of install.log (if present):
echo ============================================================
if exist "%PIXELPAWS_ROOT%\install.log" (
    powershell -ExecutionPolicy Bypass -Command "Get-Content -Tail 60 -Path '%PIXELPAWS_ROOT%\install.log'"
) else (
    echo   install.log not found at %PIXELPAWS_ROOT%\install.log
)
echo.
echo ============================================================
echo  Diagnostics complete. Share this output ^(and install.log^).
echo ============================================================
pause
endlocal
