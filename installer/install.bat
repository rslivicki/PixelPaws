@echo off
REM ============================================================================
REM  PixelPaws installer (Windows)
REM
REM  Creates a self-contained `pixelpaws` conda environment using Miniforge3,
REM  drops a desktop shortcut, and seeds the default DLC model bundle to
REM  %LOCALAPPDATA%\PixelPaws\bundles\pixelpaws_v1\.
REM
REM  End-user prerequisites:
REM    - Windows 10 or 11 (64-bit)
REM    - For GPU: NVIDIA driver >= 525  (NO CUDA Toolkit install needed)
REM    - ~6 GB free disk space
REM ============================================================================
setlocal EnableDelayedExpansion

set "PIXELPAWS_ROOT=%~dp0.."
pushd "%PIXELPAWS_ROOT%"
set "PIXELPAWS_ROOT=%CD%"
popd

set "LOGFILE=%PIXELPAWS_ROOT%\install.log"
set "STAGE=START"

echo. > "%LOGFILE%"
call :log "=================================================="
call :log "  PixelPaws installer"
call :log "  Started: %DATE% %TIME%"
call :log "  Install root: %PIXELPAWS_ROOT%"
call :log "  Log file:     %LOGFILE%"
call :log "=================================================="

REM -- Find an existing mamba/conda anywhere on this machine ----------------
set "STAGE=miniforge-detect"
set "MAMBA_ROOT="

call :log "Looking for an existing Miniforge / Mambaforge / Anaconda / Miniconda install..."

REM 1. Honor explicit override.
if defined PIXELPAWS_MAMBA_ROOT (
    if exist "%PIXELPAWS_MAMBA_ROOT%\Scripts\mamba.exe" set "MAMBA_ROOT=%PIXELPAWS_MAMBA_ROOT%"
    if exist "%PIXELPAWS_MAMBA_ROOT%\Scripts\conda.exe" set "MAMBA_ROOT=%PIXELPAWS_MAMBA_ROOT%"
)

REM 2. Common per-user / system install locations on every drive.
if not defined MAMBA_ROOT call :check_conda_root "%LOCALAPPDATA%\PixelPaws\miniforge3"
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

REM 3. Whatever the PATH has.
if not defined MAMBA_ROOT (
    for /f "tokens=*" %%i in ('where mamba.exe 2^>nul') do (
        for %%j in ("%%~dpi..") do call :check_conda_root "%%~fj"
    )
)
if not defined MAMBA_ROOT (
    for /f "tokens=*" %%i in ('where conda.exe 2^>nul') do (
        for %%j in ("%%~dpi..") do call :check_conda_root "%%~fj"
    )
)

if defined MAMBA_ROOT (
    call :log "Found existing conda install at: !MAMBA_ROOT!"
    goto :have_mamba
)

REM ------------------------------------------------------------------ #
REM  Nothing found — ask the user where to install Miniforge3.
REM ------------------------------------------------------------------ #
set "STAGE=miniforge-pickdir"
call :log "No existing Miniforge / Mambaforge / Anaconda detected."
echo.
echo ============================================================
echo   Miniforge3 needs to be installed (~80 MB download,
echo   ~500 MB on disk after extraction).
echo.
echo   Detected drives:
REM wmic /format:csv emits "Node,Caption,FreeSpace" — we want the 2nd and 3rd
REM tokens. `set /a` overflows past 2 GB for byte counts, so we approximate
REM GB by string-slicing off the last 9 digits.
for /f "skip=1 tokens=2,3 delims=," %%a in ('wmic logicaldisk where "DriveType=3" get Caption^,FreeSpace /format:csv 2^>nul') do (
    if not "%%b"=="" (
        set "FREE_BYTES=%%b"
        set "FREE_GB_X=!FREE_BYTES:~0,-9!"
        if "!FREE_GB_X!"=="" set "FREE_GB_X=0"
        echo     %%a   !FREE_GB_X! GB free
    )
)
echo.
echo   Where would you like Miniforge3 installed?
echo     1) %LOCALAPPDATA%\PixelPaws\miniforge3   (default, no admin needed)
echo     2) Choose another drive letter (e.g. D, G).
echo.
set "DRIVE_CHOICE="
set /p "DRIVE_CHOICE=Press Enter for the default, or type a drive letter (e.g. D): "
if "%DRIVE_CHOICE%"=="" (
    set "MAMBA_ROOT=%LOCALAPPDATA%\PixelPaws\miniforge3"
) else (
    REM strip trailing colon if user typed "D:" and uppercase it
    set "DRIVE_CHOICE=%DRIVE_CHOICE::=%"
    set "MAMBA_ROOT=!DRIVE_CHOICE!:\PixelPaws\miniforge3"
)
call :log "Install location: !MAMBA_ROOT!"

REM If the target already exists (leftover from a failed prior attempt),
REM try to remove it. If we can't, auto-pick a fresh path with a numeric
REM suffix — Miniforge3's NSIS installer refuses to install over a
REM non-empty directory, so we have to give it virgin ground.
if exist "!MAMBA_ROOT!" (
    call :log "Target !MAMBA_ROOT! already exists. Attempting to remove..."
    rmdir /S /Q "!MAMBA_ROOT!" 1>>"%LOGFILE%" 2>&1
    if exist "!MAMBA_ROOT!" (
        call :log "Could not remove !MAMBA_ROOT! (likely in use). Picking a fresh path..."
        set "MAMBA_BASE=!MAMBA_ROOT!"
        set /a "PP_SUFFIX=2"
        :pick_fresh_loop
        set "MAMBA_ROOT=!MAMBA_BASE!_!PP_SUFFIX!"
        if exist "!MAMBA_ROOT!" (
            set /a "PP_SUFFIX+=1"
            if !PP_SUFFIX! GTR 50 call :fatal "Tried 50 fresh paths and they were all occupied. Manually delete leftover miniforge3* directories under !MAMBA_BASE!'s parent and rerun."
            goto :pick_fresh_loop
        )
        call :log "Using fresh path: !MAMBA_ROOT!"
    )
)

set "STAGE=miniforge-download"
call :log "Downloading Miniforge3 installer..."
set "MF_INSTALLER=%TEMP%\Miniforge3-Windows-x86_64.exe"
set "MF_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
powershell -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest '%MF_URL%' -OutFile '%MF_INSTALLER%'" 1>>"%LOGFILE%" 2>&1
if not exist "%MF_INSTALLER%" (
    call :fatal "Miniforge3 download failed. Check internet connection and antivirus. URL: %MF_URL%"
)

set "STAGE=miniforge-install"
call :log "Installing Miniforge3 silently to !MAMBA_ROOT! ..."
call :log "Command: ""%MF_INSTALLER%"" /InstallationType=JustMe /RegisterPython=0 /S /D=!MAMBA_ROOT!"

REM Verify download integrity before running.
for %%F in ("%MF_INSTALLER%") do set "MF_SIZE=%%~zF"
call :log "Installer size: !MF_SIZE! bytes"
if "!MF_SIZE!"=="0" call :fatal "Downloaded installer is 0 bytes. Antivirus likely deleted it. Add an exception for %TEMP% and retry."
if "!MF_SIZE!"=="" call :fatal "Could not stat the downloaded installer at %MF_INSTALLER%."

REM Belt-and-braces: the picker stage already handled an existing dir, but
REM if anything (AV scan, file-watcher) recreated files in the meantime,
REM bump to a fresh suffix one more time before NSIS sees a non-empty path.
if exist "!MAMBA_ROOT!" (
    call :log "Target reappeared at !MAMBA_ROOT! between picker and install. Bumping suffix..."
    set "MAMBA_BASE2=!MAMBA_ROOT!"
    set /a "PP_SUFFIX2=2"
    :pick_fresh_loop2
    set "MAMBA_ROOT=!MAMBA_BASE2!_alt!PP_SUFFIX2!"
    if exist "!MAMBA_ROOT!" (
        set /a "PP_SUFFIX2+=1"
        if !PP_SUFFIX2! GTR 50 call :fatal "All 50 fallback paths under !MAMBA_BASE2! are occupied. Reboot the machine and rerun."
        goto :pick_fresh_loop2
    )
    call :log "Using fallback fresh path: !MAMBA_ROOT!"
)

"%MF_INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /S /D=!MAMBA_ROOT! 1>>"%LOGFILE%" 2>&1
set "MF_RC=!errorlevel!"
call :log "Installer exit code: !MF_RC!"

if not exist "!MAMBA_ROOT!\Scripts\mamba.exe" (
    if not exist "!MAMBA_ROOT!\Scripts\conda.exe" (
        echo. >> "%LOGFILE%"
        echo MAMBA_ROOT was: [!MAMBA_ROOT!] >> "%LOGFILE%"
        if exist "!MAMBA_ROOT!" (
            echo Files in MAMBA_ROOT: >> "%LOGFILE%"
            dir /B "!MAMBA_ROOT!" >> "%LOGFILE%" 2>&1
            if exist "!MAMBA_ROOT!\Scripts" dir /B "!MAMBA_ROOT!\Scripts" >> "%LOGFILE%" 2>&1
        ) else (
            echo MAMBA_ROOT does not exist on disk at all. >> "%LOGFILE%"
        )
        call :fatal "Miniforge3 silent install failed at !MAMBA_ROOT! (installer exit=!MF_RC!). See %LOGFILE% for details. Common causes: antivirus blocked /S flag, write-protected path, or out of disk space."
    )
)

:have_mamba
set "STAGE=conda-activate"
call :log "Conda root: !MAMBA_ROOT!"

REM Use mamba if present, otherwise fall back to conda. (Plain Anaconda
REM installs may not have mamba.)
set "PKG_MGR=mamba"
if not exist "!MAMBA_ROOT!\Scripts\mamba.exe" (
    if exist "!MAMBA_ROOT!\Scripts\conda.exe" (
        set "PKG_MGR=conda"
        call :log "mamba.exe not found — using conda instead (slower, but works)."
    )
)
call :log "Package manager: !PKG_MGR!"

call "!MAMBA_ROOT!\Scripts\activate.bat" 1>>"%LOGFILE%" 2>&1
if errorlevel 1 call :fatal "Failed to activate conda from !MAMBA_ROOT!."

REM -- Create or update the env ----------------------------------------------
set "STAGE=env-create"
!PKG_MGR! env list | findstr /B "pixelpaws " >nul
if errorlevel 1 (
    call :log "Creating conda env 'pixelpaws' from environment.yml (this can take 10-20 min)..."
    call !PKG_MGR! env create -f "%PIXELPAWS_ROOT%\installer\environment.yml" 1>>"%LOGFILE%" 2>&1
    set "MAMBA_RC=!errorlevel!"
) else (
    call :log "Updating existing 'pixelpaws' env from environment.yml..."
    call !PKG_MGR! env update -n pixelpaws -f "%PIXELPAWS_ROOT%\installer\environment.yml" 1>>"%LOGFILE%" 2>&1
    set "MAMBA_RC=!errorlevel!"
)
if not "!MAMBA_RC!"=="0" (
    call :fatal "!PKG_MGR! env create/update failed with exit code !MAMBA_RC!. See %LOGFILE% for details."
)

REM Verify python landed in the env.
set "STAGE=env-verify"
set "PP_PY=!MAMBA_ROOT!\envs\pixelpaws\python.exe"
if not exist "!PP_PY!" (
    call :fatal "Python not found at !PP_PY! after env create."
)
call :log "Verified python at: !PP_PY!"

REM Persist the conda root so run.bat can find it even if the user
REM installed Miniforge3 to a non-standard drive.
if not exist "%LOCALAPPDATA%\PixelPaws" mkdir "%LOCALAPPDATA%\PixelPaws" 1>>"%LOGFILE%" 2>&1
> "%LOCALAPPDATA%\PixelPaws\conda_root.txt" echo !MAMBA_ROOT!

REM -- Seed the default bundle to %LOCALAPPDATA% -----------------------------
set "STAGE=bundle-seed"
set "PP_BUNDLES=%LOCALAPPDATA%\PixelPaws\bundles"
if not exist "%PP_BUNDLES%" mkdir "%PP_BUNDLES%" 1>>"%LOGFILE%" 2>&1

if exist "%PIXELPAWS_ROOT%\default_bundle\pixelpaws_v1\manifest.json" (
    if not exist "%PP_BUNDLES%\pixelpaws_v1\manifest.json" (
        call :log "Seeding default model bundle to %PP_BUNDLES%\pixelpaws_v1 ..."
        xcopy /E /I /Y /Q "%PIXELPAWS_ROOT%\default_bundle\pixelpaws_v1" "%PP_BUNDLES%\pixelpaws_v1\" 1>>"%LOGFILE%" 2>&1
        if errorlevel 1 (
            call :fatal "Bundle copy via xcopy failed. See %LOGFILE%."
        )
    ) else (
        call :log "Default bundle already present; skipping."
    )
) else (
    call :log "NOTE: default_bundle\pixelpaws_v1\manifest.json missing in install root."
    call :log "      The app will look for an installed bundle on first run."
)

REM -- Create desktop shortcut (best-effort) ---------------------------------
set "STAGE=shortcut"
call :log "Creating desktop shortcut ..."
set "PS_TMP=%TEMP%\pp_shortcut.ps1"
> "%PS_TMP%" echo $ErrorActionPreference = 'Stop'
>> "%PS_TMP%" echo $sh = New-Object -ComObject WScript.Shell
>> "%PS_TMP%" echo $lnk = $sh.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\PixelPaws.lnk')
>> "%PS_TMP%" echo $lnk.TargetPath = '%PIXELPAWS_ROOT%\installer\run.bat'
>> "%PS_TMP%" echo $lnk.WorkingDirectory = '%PIXELPAWS_ROOT%'
>> "%PS_TMP%" echo if (Test-Path '%PIXELPAWS_ROOT%\pixelpaws_icon.ico') { $lnk.IconLocation = '%PIXELPAWS_ROOT%\pixelpaws_icon.ico' }
>> "%PS_TMP%" echo $lnk.Save()
>> "%PS_TMP%" echo Write-Host 'Shortcut created on Desktop.'
powershell -ExecutionPolicy Bypass -File "%PS_TMP%" 1>>"%LOGFILE%" 2>&1
if errorlevel 1 (
    call :log "WARNING: failed to create desktop shortcut. Launch via installer\run.bat instead."
)
del "%PS_TMP%" >nul 2>&1

REM -- Done ------------------------------------------------------------------
set "STAGE=done"
echo.
echo ============================================================
echo   PixelPaws installed successfully.
echo.
echo   Launch via the new "PixelPaws" desktop shortcut,
echo   or run:  %PIXELPAWS_ROOT%\installer\run.bat
echo.
echo   Full install log: %LOGFILE%
echo ============================================================
echo.
call :log "Install completed successfully."
pause
endlocal
exit /b 0


REM ============================== subroutines ==============================
:check_conda_root
REM Sets MAMBA_ROOT to %~1 if that directory contains a usable conda/mamba.
if "%~1"=="" exit /b 0
if not exist "%~1" exit /b 0
if exist "%~1\Scripts\mamba.exe" (
    set "MAMBA_ROOT=%~1"
    exit /b 0
)
if exist "%~1\Scripts\conda.exe" (
    set "MAMBA_ROOT=%~1"
    exit /b 0
)
exit /b 0

:log
echo %~1
echo %~1 >> "%LOGFILE%"
exit /b 0

:fatal
echo.
echo ============================================================
echo   INSTALL FAILED  (stage: %STAGE%^)
echo   %~1
echo.
echo   Full log:  %LOGFILE%
echo   Please share the log if you need help diagnosing.
echo ============================================================
echo.
echo %~1 >> "%LOGFILE%"
pause
REM "exit 1" (not "exit /b 1") terminates the whole cmd window — otherwise
REM the call returns to the caller and the script keeps running past failure.
exit 1
