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

REM Channel notices are cosmetic, and writing their cache file has crashed
REM conda AFTER a fully successful env build (PermissionError on
REM notices.cache, conda 26.x, 2026-08-28). Turn them off entirely.
set "CONDA_NUMBER_CHANNEL_NOTICES=0"

REM -- Create or update the env ----------------------------------------------
set "STAGE=env-create"

REM environment.yml stays the single source of truth, but it is split here so
REM the two slow phases run separately and each one shows live progress:
REM   conda part (python + pip)   -> %TEMP%\pp_env_base.yml
REM   pip part (everything else)  -> %TEMP%\pp_requirements.txt
set "PP_YML_BASE=%TEMP%\pp_env_base.yml"
set "PP_REQS=%TEMP%\pp_requirements.txt"
set "PS_SPLIT=%TEMP%\pp_split.ps1"
> "%PS_SPLIT%" echo $ErrorActionPreference = 'Stop'
>> "%PS_SPLIT%" echo $lines = @(Get-Content -LiteralPath '%PIXELPAWS_ROOT%\installer\environment.yml')
>> "%PS_SPLIT%" echo $i = [array]::IndexOf([string[]]($lines ^| ForEach-Object { $_.Trim() }), '- pip:')
>> "%PS_SPLIT%" echo if ($i -lt 0) { throw 'environment.yml has no pip section' }
>> "%PS_SPLIT%" echo $lines[0..($i-1)] ^| Set-Content -LiteralPath '%PP_YML_BASE%' -Encoding ASCII
>> "%PS_SPLIT%" echo $lines[($i+1)..($lines.Count-1)] ^| ForEach-Object { $_.Trim() } ^| Where-Object { $_ -like '- *' } ^| ForEach-Object { $_.Substring(2).Trim() } ^| Set-Content -LiteralPath '%PP_REQS%' -Encoding ASCII
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SPLIT%" 1>>"%LOGFILE%" 2>&1
if errorlevel 1 call :fatal "Could not parse installer\environment.yml (see %LOGFILE%)."
set "PP_NREQ=?"
for /f %%N in ('type "%PP_REQS%" ^| %SystemRoot%\System32\find.exe /c /v ""') do set "PP_NREQ=%%N"
set "PP_PY=!MAMBA_ROOT!\envs\pixelpaws\python.exe"

set "PP_MODE=create"
!PKG_MGR! env list | findstr /B "pixelpaws " >nul
if not errorlevel 1 (
    REM An env from an older PixelPaws (or an interrupted install) already
    REM exists. A fresh rebuild is the reliable upgrade path; in-place
    REM update is offered for users who added their own packages.
    call :log "An existing 'pixelpaws' environment was found (older version or interrupted install)."
    echo.
    echo   An existing PixelPaws environment was found - likely from an
    echo   older version, or from an install that did not finish.
    echo     1^) Fresh reinstall - remove it and rebuild clean  (recommended^)
    echo     2^) Update in place - faster, keeps extra packages you added
    choice /C 12 /N /D 1 /T 30 /M "  Choose 1 or 2 (auto-picks 1 after 30s): "
    if errorlevel 2 (
        set "PP_MODE=update"
    ) else (
        call :log "Removing the old 'pixelpaws' env for a clean rebuild..."
        call !PKG_MGR! env remove -n pixelpaws -y 1>>"%LOGFILE%" 2>&1
    )
)

echo.
echo ============================================================
echo   Step 1 of 3 - base Python 3.11 environment   (about 1-2 min)
echo ============================================================
if "!PP_MODE!"=="update" (
    call :log "Updating existing 'pixelpaws' env (python/pip layer)..."
    set "PP_CMD="!MAMBA_ROOT!\Scripts\!PKG_MGR!.exe" env update -n pixelpaws -f "%PP_YML_BASE%""
) else (
    call :log "Creating conda env 'pixelpaws' (python/pip layer)..."
    set "PP_CMD="!MAMBA_ROOT!\Scripts\!PKG_MGR!.exe" env create -f "%PP_YML_BASE%""
)
call :run_stream
set "MAMBA_RC=!errorlevel!"
if not "!MAMBA_RC!"=="0" (
    REM conda can exit non-zero from post-install housekeeping (notices
    REM cache, plugin hooks) after the env itself built fine. Trust the
    REM env over the exit code.
    "!MAMBA_ROOT!\Scripts\conda.exe" env list 2>nul | findstr /B "pixelpaws " >nul
    if errorlevel 1 call :fatal "!PKG_MGR! env create/update failed with exit code !MAMBA_RC!. See %LOGFILE% for details."
    call :log "NOTE: !PKG_MGR! exited with code !MAMBA_RC! but the env exists; continuing."
)

REM conda does not always put the env under !MAMBA_ROOT!\envs - when that
REM directory is not writable (system-wide Anaconda in ProgramData) it
REM silently uses %USERPROFILE%\.conda\envs instead. Ask conda where the
REM env's python actually is rather than guessing.
set "PP_ENVPY_TXT=%TEMP%\pp_envpy.txt"
"!MAMBA_ROOT!\Scripts\conda.exe" run -n pixelpaws python -c "import sys;print(sys.executable)" > "%PP_ENVPY_TXT%" 2>>"%LOGFILE%"
set "PP_PY_FOUND="
set /p PP_PY_FOUND=<"%PP_ENVPY_TXT%"
if defined PP_PY_FOUND set "PP_PY=!PP_PY_FOUND!"
if not exist "!PP_PY!" (
    call :fatal "Env 'pixelpaws' was created but its python.exe could not be located (looked at !PP_PY!). See %LOGFILE%."
)
call :log "Env python: !PP_PY!"

echo.
echo ============================================================
echo   Step 2 of 3 - Python packages   (!PP_NREQ! packages, about 3 GB)
echo   Includes PyTorch with CUDA and DeepLabCut - typically 10-20 min
echo   on a fast connection. Each line below is one package being
echo   fetched or installed; the "Installing collected packages" line
echo   near the end is the last long pause.
echo ============================================================
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

REM -- 2a: PyTorch from its own CUDA index -----------------------------------
REM With --extra-index-url PyPI's CPU-only torch wins the version tie (that
REM shipped CPU-only pose to every user until 2026-08-28). Install torch and
REM torchvision first, from the CUDA index alone, so the rest of the
REM requirements see them already satisfied.
set "PP_TORCH_INDEX="
for /f "tokens=2" %%U in ('%SystemRoot%\System32\findstr.exe /B /C:"--extra-index-url" "%PP_REQS%"') do set "PP_TORCH_INDEX=%%U"
if not defined PP_TORCH_INDEX set "PP_TORCH_INDEX=https://download.pytorch.org/whl/cu126"
REM RTX 50-series (Blackwell) needs CUDA 12.8 kernels; older cards keep cu126.
set "PS_GPU=%TEMP%\pp_gpu.ps1"
set "PP_GPU_FLAGS=%TEMP%\pp_gpu_flags.bat"
> "%PS_GPU%" echo $ErrorActionPreference = 'Continue'
>> "%PS_GPU%" echo $gpus = @(Get-CimInstance Win32_VideoController)
>> "%PS_GPU%" echo $n = ($gpus ^| Select-Object -ExpandProperty Name) -join '; '
>> "%PS_GPU%" echo Write-Host ('GPUs: ' + $n)
>> "%PS_GPU%" echo $nv = $gpus ^| Where-Object { $_.Name -match 'NVIDIA^|GeForce^|Quadro^|Tesla' } ^| Select-Object -First 1
>> "%PS_GPU%" echo $drv = ''
>> "%PS_GPU%" echo if ($nv -and $nv.DriverVersion) { $d = ($nv.DriverVersion -replace '\.', ''); if ($d.Length -ge 5) { $drv = $d.Substring($d.Length-5, 3) + '.' + $d.Substring($d.Length-2) } }
>> "%PS_GPU%" echo $bw = [int](($n -match 'RTX (50\d\d^|PRO \d+ Blackwell)') -or ($n -match 'Blackwell'))
>> "%PS_GPU%" echo Set-Content -LiteralPath '%PP_GPU_FLAGS%' -Encoding ASCII -Value @("set `"PP_HAS_NVIDIA=$([int][bool]$nv)`"", "set `"PP_NV_DRIVER=$drv`"", "set `"PP_BLACKWELL=$bw`"")
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_GPU%" 2>>"%LOGFILE%"
set "PP_HAS_NVIDIA=0"
set "PP_NV_DRIVER="
set "PP_BLACKWELL=0"
if exist "%PP_GPU_FLAGS%" call "%PP_GPU_FLAGS%"
set "PP_DRV_MIN=528.33"
if "!PP_BLACKWELL!"=="1" (
    set "PP_TORCH_INDEX=!PP_TORCH_INDEX:cu126=cu128!"
    set "PP_DRV_MIN=570.00"
    call :log "RTX 50-series GPU detected - using the CUDA 12.8 PyTorch build."
)
if "!PP_HAS_NVIDIA!"=="1" call :log "NVIDIA driver version: !PP_NV_DRIVER! (PyTorch build needs >= !PP_DRV_MIN!)"
del "%PS_GPU%" >nul 2>&1
call :log "PyTorch index: !PP_TORCH_INDEX!"
set "PP_TORCH_SPEC="
for /f "usebackq delims=" %%T in (`%SystemRoot%\System32\findstr.exe /B /R /C:"torch[<>=]" "%PP_REQS%"`) do set "PP_TORCH_SPEC=%%T"
if not defined PP_TORCH_SPEC set "PP_TORCH_SPEC=torch>=2.7.0,<3.0"
set "PP_CMD="!PP_PY!" -m pip install --progress-bar off --upgrade --index-url !PP_TORCH_INDEX! "!PP_TORCH_SPEC!" torchvision"
call :run_stream
set "PIP_RC=!errorlevel!"
if not "!PIP_RC!"=="0" (
    call :fatal "PyTorch install failed with exit code !PIP_RC! (index !PP_TORCH_INDEX!). Scroll up for the first line starting with ERROR, or see %LOGFILE%."
)

REM -- 2b: everything else from PyPI -----------------------------------------
echo.
echo   ... PyTorch done. Now the remaining packages (DeepLabCut, OpenCV, ...).
echo.
set "PP_CMD="!PP_PY!" -m pip install --progress-bar off -r "%PP_REQS%""
call :run_stream
set "PIP_RC=!errorlevel!"
if not "!PIP_RC!"=="0" (
    call :log "pip exited with code !PIP_RC!; checking whether the env is nevertheless complete..."
    set "PP_ENV_OK="
    "!PP_PY!" -c "import torch, cv2, ttkbootstrap, deeplabcut" 1>>"%LOGFILE%" 2>&1
    if not errorlevel 1 set "PP_ENV_OK=1"
    if defined PP_ENV_OK (
        call :log "Environment verified complete despite the exit code; continuing."
    ) else (
        call :fatal "Package install failed with exit code !PIP_RC!. Scroll up for the first line starting with ERROR, or see %LOGFILE%."
    )
)

REM Verify python landed in the env.
set "STAGE=env-verify"
if not exist "!PP_PY!" (
    call :fatal "Python not found at !PP_PY! after env create."
)
call :log "Verified python at: !PP_PY!"
set "STAGE=gpu-check"
"!PP_PY!" -c "import torch; ok=torch.cuda.is_available(); print('GPU: ' + (torch.cuda.get_device_name(0) + ' ready (CUDA ' + str(torch.version.cuda) + ')' if ok else 'no CUDA GPU available to PyTorch ' + str(torch.__version__) + ' - pose tracking will run on CPU'))" > "%TEMP%\pp_gpu.txt" 2>>"%LOGFILE%"
set "PP_GPU_MSG="
set /p PP_GPU_MSG=<"%TEMP%\pp_gpu.txt"
if defined PP_GPU_MSG call :log "!PP_GPU_MSG!"
set "PP_CUDA_OK=0"
if defined PP_GPU_MSG if not "!PP_GPU_MSG:ready=!"=="!PP_GPU_MSG!" set "PP_CUDA_OK=1"
if "!PP_HAS_NVIDIA!"=="1" if "!PP_CUDA_OK!"=="0" (
    call :log "NVIDIA GPU present but PyTorch cannot use it - the NVIDIA driver is most likely too old."
    echo.
    echo ============================================================
    echo   Your NVIDIA graphics driver needs an update
    echo.
    echo   Installed driver : !PP_NV_DRIVER!
    echo   Required         : !PP_DRV_MIN! or newer
    echo.
    echo   PixelPaws is installed and will work, but pose tracking will
    echo   run on the CPU ^(20-30x slower^) until the driver is updated.
    echo   After updating the driver just relaunch PixelPaws - no
    echo   reinstall is needed.
    echo.
    echo   Download: https://www.nvidia.com/drivers
    echo ============================================================
    choice /C YN /N /D Y /T 60 /M "  Open the NVIDIA driver download page now? [Y/N] (auto-Y in 60s): "
    if not errorlevel 2 start "" "https://www.nvidia.com/drivers"
)

REM Persist the conda root so run.bat can find it even if the user
REM installed Miniforge3 to a non-standard drive.
if not exist "%LOCALAPPDATA%\PixelPaws" mkdir "%LOCALAPPDATA%\PixelPaws" 1>>"%LOGFILE%" 2>&1
> "%LOCALAPPDATA%\PixelPaws\conda_root.txt" echo !MAMBA_ROOT!

REM -- Detect a previous install in a different folder ----------------------
set "PREV_ROOT="
if exist "%LOCALAPPDATA%\PixelPaws\install_root.txt" set /p PREV_ROOT=<"%LOCALAPPDATA%\PixelPaws\install_root.txt"
if defined PREV_ROOT if /I not "!PREV_ROOT!"=="%PIXELPAWS_ROOT%" (
    call :log "Previous PixelPaws install detected at: !PREV_ROOT!"
    call :log "The desktop shortcut will now point at THIS copy. The old folder"
    call :log "was left untouched (it may contain your project data). Once the"
    call :log "new version runs, the old app folder can be deleted."
)
> "%LOCALAPPDATA%\PixelPaws\install_root.txt" echo %PIXELPAWS_ROOT%

REM -- Seed the default bundle to %LOCALAPPDATA% -----------------------------
echo.
echo ============================================================
echo   Step 3 of 3 - model bundle and desktop shortcut   (under a minute)
echo ============================================================
set "STAGE=bundle-seed"
set "PP_BUNDLES=%LOCALAPPDATA%\PixelPaws\bundles"
if not exist "%PP_BUNDLES%" mkdir "%PP_BUNDLES%" 1>>"%LOGFILE%" 2>&1

if exist "%PIXELPAWS_ROOT%\default_bundle\pixelpaws_v1\manifest.json" (
    set "SEED_BUNDLE=1"
    if exist "%PP_BUNDLES%\pixelpaws_v1\manifest.json" (
        REM Refresh only when the shipped bundle differs from the installed
        REM one - an old install must not pin users to an outdated model.
        fc /B "%PIXELPAWS_ROOT%\default_bundle\pixelpaws_v1\manifest.json" "%PP_BUNDLES%\pixelpaws_v1\manifest.json" >nul 2>&1
        if not errorlevel 1 set "SEED_BUNDLE="
    )
    if defined SEED_BUNDLE (
        call :log "Seeding default model bundle to %PP_BUNDLES%\pixelpaws_v1 ..."
        xcopy /E /I /Y /Q "%PIXELPAWS_ROOT%\default_bundle\pixelpaws_v1" "%PP_BUNDLES%\pixelpaws_v1\" 1>>"%LOGFILE%" 2>&1
        if errorlevel 1 (
            call :fatal "Bundle copy via xcopy failed. See %LOGFILE%."
        )
    ) else (
        call :log "Default bundle already up to date; skipping."
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
>> "%PS_TMP%" echo $ico = @('%PIXELPAWS_ROOT%\assets\pixelpaws_icon.ico', '%PIXELPAWS_ROOT%\pixelpaws_icon.ico') ^| Where-Object { Test-Path $_ } ^| Select-Object -First 1
>> "%PS_TMP%" echo if ($ico) { $lnk.IconLocation = $ico }
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

:run_stream
REM Run %PP_CMD% with its output shown live AND appended to the log, and
REM report how long it took. Exit code is the command's own.
set "PS_RUN=%TEMP%\pp_run.ps1"
set "PP_LOGFILE=%LOGFILE%"
> "%PS_RUN%" echo $ErrorActionPreference = 'Continue'
>> "%PS_RUN%" echo $t0 = Get-Date
>> "%PS_RUN%" echo ^& cmd.exe /c $env:PP_CMD 2^>^&1 ^| ForEach-Object { $s = [string]$_; if ($s.Trim().Length -gt 0) { Write-Host $s; Add-Content -LiteralPath $env:PP_LOGFILE -Value $s } }
>> "%PS_RUN%" echo $rc = $LASTEXITCODE
>> "%PS_RUN%" echo $dt = (Get-Date) - $t0
>> "%PS_RUN%" echo Write-Host ('  [done in {0}m {1:00}s, exit code {2}]' -f [int]$dt.TotalMinutes, $dt.Seconds, $rc)
>> "%PS_RUN%" echo exit $rc
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_RUN%"
exit /b %errorlevel%

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
