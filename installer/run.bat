@echo off
REM Launch PixelPaws in its dedicated conda env.

setlocal EnableDelayedExpansion
REM NOTE: %-expansion happens when a whole if (...) block is parsed, so an
REM unquoted path containing ")" (e.g. "...win64(1)\PixelPaws") closes the
REM block early and cmd aborts. Inside blocks use !delayed! expansion only.
set "PP_INSTALLER_DIR=%~dp0"
set "PIXELPAWS_ROOT=%~dp0.."
pushd "%PIXELPAWS_ROOT%"
set "PIXELPAWS_ROOT=%CD%"
popd

REM run.log records every launch from the very first step, so a failure
REM anywhere in this launcher (not just in the app) can be reported.
if not exist "%LOCALAPPDATA%\PixelPaws" mkdir "%LOCALAPPDATA%\PixelPaws" >nul 2>&1
set "PP_LOG=%LOCALAPPDATA%\PixelPaws\run.log"
> "%PP_LOG%" echo PixelPaws launch %DATE% %TIME%
>> "%PP_LOG%" echo root=%PIXELPAWS_ROOT%

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
    >> "%PP_LOG%" echo ERROR: no conda install found
    echo ============================================================
    echo  ERROR: Could not find a Miniforge / Mambaforge / Anaconda install.
    echo  Run !PP_INSTALLER_DIR!install.bat first.
    echo ============================================================
    pause
    exit /b 1
)

call "!MAMBA_ROOT!\Scripts\activate.bat" pixelpaws
if errorlevel 1 (
    >> "%PP_LOG%" echo ERROR: could not activate env pixelpaws under !MAMBA_ROOT!
    echo ============================================================
    echo  ERROR: Could not activate conda env "pixelpaws" under !MAMBA_ROOT!.
    echo  Run !PP_INSTALLER_DIR!install.bat to rebuild it.
    echo ============================================================
    pause
    exit /b 1
)

cd /d "%PIXELPAWS_ROOT%"
REM Everything the app prints (tracebacks included) goes to run.log so a
REM crash can be reported even if this window closes.
>> "%PP_LOG%" echo conda=!MAMBA_ROOT!
>> "%PP_LOG%" where python
>> "%PP_LOG%" echo --- app output ---
call :now PP_T0
python PixelPaws_GUI.py %* >> "%PP_LOG%" 2>&1
set "PP_RC=%errorlevel%"
call :now PP_T1
set /a "PP_DT=PP_T1-PP_T0"
if !PP_DT! LSS 0 set /a "PP_DT+=86400"
>> "%PP_LOG%" echo --- exit code %PP_RC% after !PP_DT! s ---
REM A "success" exit within 10 s means the window would have closed before
REM anyone could read an error - keep it open in that case too.
if !PP_DT! LSS 10 set "PP_RC=%PP_RC% (closed after !PP_DT! s)"
if not "!PP_RC!"=="0" (
    echo.
    echo ============================================================
    echo  PixelPaws exited with code !PP_RC!
    echo ============================================================
    type "!PP_LOG!"
    echo.
    echo  This output is saved at: !PP_LOG!
    echo  Please send that file when reporting the problem.
    pause
)
endlocal
exit /b 0

:now
REM seconds since midnight -> variable named in %1 (leading zeros are not octal)
for /f "tokens=1-3 delims=:.," %%a in ("%TIME: =0%") do set /a "%~1=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)"
exit /b 0

:check_conda_root
if "%~1"=="" exit /b 0
if not exist "%~1" exit /b 0
if exist "%~1\Scripts\mamba.exe" ( set "MAMBA_ROOT=%~1" & exit /b 0 )
if exist "%~1\Scripts\conda.exe" ( set "MAMBA_ROOT=%~1" & exit /b 0 )
exit /b 0
