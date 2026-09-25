@echo off
REM build.bat - build PawCapture for distribution.
REM
REM Produces:
REM   dist\PawCapture\                       portable folder (PawCapture.exe + ffmpeg.exe + runtime)
REM   dist\PawCapture-<ver>_win64.zip        the portable folder, zipped
REM   installer_output\PawCapture_Setup_v<ver>.exe   only if Inno Setup 6 is installed
REM
REM Usage:  build.bat            (incremental)
REM         build.bat --clean    (wipe build\ and dist\ first)
REM
REM Needs Python 3.10+ (the "py" launcher or python on PATH; set PAWCAPTURE_PYTHON
REM to pick a specific interpreter) and internet on the first run (pip packages,
REM ffmpeg download). The version is read from PAWCAPTURE_VERSION in pawcapture_app.py.

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PY=%PAWCAPTURE_PYTHON%"
if "%PY%"=="" set "PY=py -3"
%PY% --version >nul 2>&1
if errorlevel 1 set "PY=python"
%PY% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: no Python found. Install Python 3.10+ from python.org, or set PAWCAPTURE_PYTHON.
    exit /b 1
)
for /f "delims=" %%v in ('%PY% --version') do echo Using %%v

for /f "tokens=2 delims==" %%v in ('findstr /b "PAWCAPTURE_VERSION" pawcapture_app.py') do set "VER=%%v"
set "VER=%VER:"=%"
set "VER=%VER: =%"
if "%VER%"=="" (
    echo ERROR: could not read PAWCAPTURE_VERSION from pawcapture_app.py
    exit /b 1
)
echo Building PawCapture %VER%

if /i "%~1"=="--clean" (
    echo Cleaning build\ and dist\ ...
    if exist build rmdir /s /q build
    if exist dist  rmdir /s /q dist
)

REM ---- ffmpeg (bundled next to PawCapture.exe) --------------------------------
if not exist "ffmpeg\ffmpeg.exe" (
    echo Downloading ffmpeg release-essentials build from gyan.dev ...
    if not exist ffmpeg mkdir ffmpeg
    powershell -NoProfile -Command ^
        "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $zip='ffmpeg\ffmpeg_dl.zip'; Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile $zip -UseBasicParsing; Expand-Archive -Path $zip -DestinationPath 'ffmpeg\ex' -Force; $exe=Get-ChildItem -Path 'ffmpeg\ex' -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1; Copy-Item $exe.FullName -Destination 'ffmpeg\ffmpeg.exe'; $lic=Get-ChildItem -Path 'ffmpeg\ex' -Recurse -Filter 'LICENSE*' | Select-Object -First 1; if ($lic) { Copy-Item $lic.FullName -Destination 'ffmpeg\FFMPEG_LICENSE.txt' }; Remove-Item $zip -Force; Remove-Item 'ffmpeg\ex' -Recurse -Force"
    if errorlevel 1 goto :err
    if not exist "ffmpeg\ffmpeg.exe" (
        echo ERROR: ffmpeg.exe missing after download
        exit /b 1
    )
)
for /f "tokens=1-3" %%a in ('ffmpeg\ffmpeg.exe -version 2^>^&1') do (
    echo Bundling %%a %%b %%c
    goto :ffdone
)
:ffdone

REM ---- Python deps + PyInstaller ---------------------------------------------
echo Installing/refreshing dependencies ...
%PY% -m pip install --quiet --upgrade pip
if errorlevel 1 goto :err
%PY% -m pip install --quiet -r requirements.txt
if errorlevel 1 goto :err

echo Running PyInstaller ...
%PY% -m PyInstaller --noconfirm --clean PawCapture.spec
if errorlevel 1 goto :err
if not exist "dist\PawCapture\PawCapture.exe" (
    echo ERROR: build did not produce dist\PawCapture\PawCapture.exe
    exit /b 1
)

REM ---- Portable zip -----------------------------------------------------------
set "ZIP=dist\PawCapture-%VER%_win64.zip"
echo Zipping portable folder to %ZIP% ...
REM Python's zipfile, not Compress-Archive: Compress-Archive skipped a file it
REM could not open (base_library.zip, 2026-09-18) and still exited 0, so the
REM zip is checked against the folder's file count before it counts as built.
%PY% -c "import os,shutil,sys,zipfile; z=r'%ZIP%'; os.path.exists(z) and os.remove(z); shutil.make_archive(z[:-4],'zip','dist','PawCapture'); n=sum(len(f) for _,_,f in os.walk(r'dist\PawCapture')); m=sum(1 for i in zipfile.ZipFile(z).infolist() if not i.is_dir()); print(f'zip holds {m} of {n} files'); sys.exit(0 if m==n else 1)"
if errorlevel 1 goto :err

REM ---- Installer (optional) ---------------------------------------------------
set "ISCC="
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe"      set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if "%ISCC%"=="" (
    echo Inno Setup 6 not found - skipping PawCapture_Setup_v%VER%.exe ^(portable zip is complete^).
    echo   Install it from https://jrsoftware.org/isdl.php ^(or: winget install JRSoftware.InnoSetup^) and rerun.
) else (
    echo Building installer with Inno Setup ...
    if not exist installer_output mkdir installer_output
    "%ISCC%" /Q /DAppVersion=%VER% installer.iss
    if errorlevel 1 goto :err
    echo Installer: installer_output\PawCapture_Setup_v%VER%.exe
)

echo.
echo Built: dist\PawCapture\PawCapture.exe
echo Zip:   %ZIP%
exit /b 0

:err
echo.
echo BUILD FAILED - see the output above.
exit /b 1
