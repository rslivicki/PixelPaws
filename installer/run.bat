@echo off
REM Launch PixelPaws in its dedicated conda env.

setlocal EnableDelayedExpansion
set "PIXELPAWS_ROOT=%~dp0.."
pushd "%PIXELPAWS_ROOT%"
set "PIXELPAWS_ROOT=%CD%"
popd

set "MAMBA_ROOT="
call :check_conda_root "%LOCALAPPDATA%\PixelPaws\miniforge3"
if not defined MAMBA_ROOT call :check_conda_root "%USERPROFILE%\miniforge3"
if not defined MAMBA_ROOT call :check_conda_root "%USERPROFILE%\mambaforge"
if not defined MAMBA_ROOT call :check_conda_root "%USERPROFILE%\anaconda3"
if not defined MAMBA_ROOT call :check_conda_root "%USERPROFILE%\miniconda3"
if not defined MAMBA_ROOT call :check_conda_root "%ProgramData%\miniforge3"
if not defined MAMBA_ROOT call :check_conda_root "%ProgramData%\Anaconda3"
if not defined MAMBA_ROOT call :check_conda_root "%ProgramData%\Miniconda3"
if not defined MAMBA_ROOT call :check_conda_root "C:\miniforge3"
if not defined MAMBA_ROOT call :check_conda_root "C:\Anaconda3"
if not defined MAMBA_ROOT call :check_conda_root "C:\Miniconda3"
if not defined MAMBA_ROOT call :check_conda_root "C:\ProgramData\Anaconda3"

REM Also honor whatever the install.bat used (saved in a side-pointer file).
if not defined MAMBA_ROOT if exist "%LOCALAPPDATA%\PixelPaws\conda_root.txt" (
    set /p "MAMBA_ROOT="<"%LOCALAPPDATA%\PixelPaws\conda_root.txt"
)

if not defined MAMBA_ROOT (
    echo ============================================================
    echo  ERROR: Could not find a Miniforge / Mambaforge / Anaconda install.
    echo  Run %~dp0install.bat first.
    echo ============================================================
    pause
    exit /b 1
)

call "!MAMBA_ROOT!\Scripts\activate.bat" pixelpaws
if errorlevel 1 (
    echo ============================================================
    echo  ERROR: Could not activate conda env "pixelpaws" under !MAMBA_ROOT!.
    echo  Run %~dp0install.bat to rebuild it.
    echo ============================================================
    pause
    exit /b 1
)

cd /d "%PIXELPAWS_ROOT%"
REM Everything python prints to stderr (tracebacks included) is also kept in
REM run.log so a crash can be reported even if this window closes.
if not exist "%LOCALAPPDATA%\PixelPaws" mkdir "%LOCALAPPDATA%\PixelPaws" >nul 2>&1
set "PP_LOG=%LOCALAPPDATA%\PixelPaws\run.log"
> "%PP_LOG%" echo PixelPaws launch %DATE% %TIME%
>> "%PP_LOG%" echo root=%PIXELPAWS_ROOT%
>> "%PP_LOG%" echo conda=!MAMBA_ROOT!
>> "%PP_LOG%" where python
python PixelPaws_GUI.py %* 2>> "%PP_LOG%"
set "PP_RC=%errorlevel%"
>> "%PP_LOG%" echo exit code %PP_RC%
if not "%PP_RC%"=="0" (
    echo.
    echo ============================================================
    echo  PixelPaws exited with code %PP_RC%
    echo ============================================================
    type "%PP_LOG%"
    echo.
    echo  This output is saved at: %PP_LOG%
    echo  Please send that file when reporting the problem.
    pause
)
endlocal
exit /b 0

:check_conda_root
if "%~1"=="" exit /b 0
if not exist "%~1" exit /b 0
if exist "%~1\Scripts\mamba.exe" ( set "MAMBA_ROOT=%~1" & exit /b 0 )
if exist "%~1\Scripts\conda.exe" ( set "MAMBA_ROOT=%~1" & exit /b 0 )
exit /b 0
