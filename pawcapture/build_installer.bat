@echo off
:: Always run from the folder this .bat file lives in
cd /d "%~dp0"
setlocal EnableDelayedExpansion
title CamSync Pro — Build Installer

:: Log everything to build_log.txt as well as the console
set LOG=build_log.txt
echo Build started: %DATE% %TIME% > %LOG%

echo.
echo ============================================================
echo   CamSync Pro — Windows Installer Builder
echo   All output is also saved to build_log.txt
echo ============================================================
echo.

:: ── Check Python ─────────────────────────────────────────────────────────────
echo [1/6] Checking Python...
python --version >> %LOG% 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH. >> %LOG%
    echo [ERROR] Python not found in PATH.
    echo         Download from https://python.org
    goto :fail
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do (
    echo [OK] %%v
    echo [OK] %%v >> %LOG%
)

:: ── Install dependencies ──────────────────────────────────────────────────────
echo.
echo [2/6] Installing Python dependencies (this may take a minute)...
echo [2/6] pip install >> %LOG%
python -m pip install --upgrade pip >> %LOG% 2>&1
python -m pip install opencv-python PyQt5 numpy pyinstaller pygrabber >> %LOG% 2>&1
if errorlevel 1 (
    echo [ERROR] pip install failed - check build_log.txt for details
    echo [ERROR] pip install failed >> %LOG%
    goto :fail
)
echo [OK] Dependencies installed.
echo [OK] Dependencies installed >> %LOG%

:: ── Check Inno Setup ─────────────────────────────────────────────────────────
set ISCC=""
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe"      set ISCC="%ProgramFiles%\Inno Setup 6\ISCC.exe"
if %ISCC%=="" (
    echo [WARNING] Inno Setup 6 not found - will skip installer creation.
    echo [WARNING] Inno Setup not found >> %LOG%
    set SKIP_INNO=1
) else (
    echo [OK] Inno Setup: !ISCC!
    echo [OK] Inno Setup: !ISCC! >> %LOG%
    set SKIP_INNO=0
)

:: ── Download FFmpeg ───────────────────────────────────────────────────────────
echo.
echo [3/6] Checking for FFmpeg...
echo [3/6] FFmpeg check >> %LOG%
if exist "ffmpeg\ffmpeg.exe" (
    echo [OK] ffmpeg\ffmpeg.exe already present.
    echo [OK] FFmpeg already present >> %LOG%
) else (
    echo      Downloading FFmpeg (~60MB^)...
    if not exist "ffmpeg" mkdir ffmpeg
    powershell -NoProfile -Command ^
        "try { $url='https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'; $zip='ffmpeg\ffmpeg_dl.zip'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Write-Host 'Downloading...'; Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing; Write-Host 'Extracting...'; Expand-Archive -Path $zip -DestinationPath 'ffmpeg\ex' -Force; $exe=Get-ChildItem -Path 'ffmpeg\ex' -Recurse -Filter 'ffmpeg.exe'|Select-Object -First 1; Copy-Item $exe.FullName -Destination 'ffmpeg\ffmpeg.exe'; Remove-Item $zip -Force; Remove-Item 'ffmpeg\ex' -Recurse -Force; Write-Host 'FFmpeg ready.' } catch { Write-Host ('FAILED: ' + $_.Exception.Message); exit 1 }" >> %LOG% 2>&1
    if errorlevel 1 (
        echo [ERROR] FFmpeg download failed - check build_log.txt
        echo [ERROR] FFmpeg download failed >> %LOG%
        goto :fail
    )
    if not exist "ffmpeg\ffmpeg.exe" (
        echo [ERROR] ffmpeg.exe not found after download
        echo [ERROR] ffmpeg.exe missing after download >> %LOG%
        goto :fail
    )
    echo [OK] FFmpeg downloaded.
    echo [OK] FFmpeg downloaded >> %LOG%
)

:: ── Generate installer artwork ────────────────────────────────────────────────
echo.
echo [4/6] Generating installer artwork...
echo [4/6] Artwork >> %LOG%
if not exist "installer" mkdir installer
if not exist "installer\icon.ico" (
    python -c "import struct,zlib,pathlib;p=lambda t,d:struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xFFFFFFFF);s=32;cx=cy=s//2;r=s//2-1;rows=[[((255,122,0,255)if(x-cx)**2+(y-cy)**2<=r*r else(0,0,0,0))for x in range(s)]for y in range(s)];raw=b''.join(b'\x00'+bytes([v for px in row for v in px])for row in rows);png=b'\x89PNG\r\n\x1a\n'+p(b'IHDR',struct.pack('>IIBBBBB',s,s,8,6,0,0,0))+p(b'IDAT',zlib.compress(raw,9))+p(b'IEND',b'');entry=struct.pack('<BBBBHHII',0,0,0,0,1,32,len(png),22);pathlib.Path('installer/icon.ico').write_bytes(struct.pack('<HHH',0,1,1)+entry+png);w,h,c=497,314,(13,13,24);row=bytes(list(reversed(c)))*w;stride=row+b'\x00'*((4-(w*3)%4)%4);data=stride*h;fs=54+len(data);pathlib.Path('installer/wizard_banner.bmp').write_bytes(struct.pack('<HIHHI',0x4D42,fs,0,0,54)+struct.pack('<IiiHHIIiiII',40,w,-h,1,24,0,len(data),2835,2835,0,0)+data);w2,h2,c2=55,58,(255,122,0);row2=bytes(list(reversed(c2)))*w2;stride2=row2+b'\x00'*((4-(w2*3)%4)%4);data2=stride2*h2;fs2=54+len(data2);pathlib.Path('installer/wizard_icon.bmp').write_bytes(struct.pack('<HIHHI',0x4D42,fs2,0,0,54)+struct.pack('<IiiHHIIiiII',40,w2,-h2,1,24,0,len(data2),2835,2835,0,0)+data2)" >> %LOG% 2>&1
    if errorlevel 1 (
        echo [WARNING] Artwork generation failed - installer will use defaults.
        echo [WARNING] Artwork failed >> %LOG%
    ) else (
        echo [OK] Artwork generated.
        echo [OK] Artwork generated >> %LOG%
    )
) else (
    echo [OK] Artwork already present.
)

:: ── PyInstaller ───────────────────────────────────────────────────────────────
echo.
echo [5/6] Bundling with PyInstaller (may take 2-4 minutes)...
echo [5/6] PyInstaller >> %LOG%
python -m PyInstaller --clean --noconfirm camsync.spec >> %LOG% 2>&1
if errorlevel 1 (
    echo [ERROR] PyInstaller failed - check build_log.txt for details
    echo [ERROR] PyInstaller failed >> %LOG%
    goto :fail
)
if not exist "dist\CamSyncPro.exe" (
    echo [ERROR] dist\CamSyncPro.exe not found after PyInstaller
    echo [ERROR] EXE missing after PyInstaller >> %LOG%
    goto :fail
)
for %%F in ("dist\CamSyncPro.exe") do set /a SIZE_MB=%%~zF / 1048576
echo [OK] EXE built: dist\CamSyncPro.exe (~!SIZE_MB! MB)
echo [OK] EXE built ~!SIZE_MB! MB >> %LOG%

:: ── Inno Setup ────────────────────────────────────────────────────────────────
echo.
if "%SKIP_INNO%"=="1" (
    echo [5b/6] Skipping Inno Setup ^(not installed^).
    echo        Standalone EXE ready at: dist\CamSyncPro.exe
) else (
    echo [6/6] Building installer with Inno Setup...
    echo [6/6] Inno Setup >> %LOG%
    if not exist "installer_output" mkdir installer_output
    !ISCC! installer.iss >> %LOG% 2>&1
    if errorlevel 1 (
        echo [ERROR] Inno Setup failed - check build_log.txt
        echo [ERROR] Inno Setup failed >> %LOG%
        goto :fail
    )
    echo [OK] Installer built.
    echo [OK] Installer built >> %LOG%
)

echo.
echo ============================================================
if "%SKIP_INNO%"=="0" echo   Installer : installer_output\CamSyncPro_Setup_v0.6.0.exe
echo   Standalone: dist\CamSyncPro.exe
echo   FFmpeg is fully bundled.
echo   Full log saved to: %LOG%
echo ============================================================
echo Build completed successfully: %DATE% %TIME% >> %LOG%
echo.
pause
goto :eof

:fail
echo.
echo ============================================================
echo   BUILD FAILED
echo   Open build_log.txt in this folder for the full error details.
echo ============================================================
echo Build FAILED: %DATE% %TIME% >> %LOG%
echo.
pause
exit /b 1
