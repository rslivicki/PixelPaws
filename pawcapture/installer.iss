; ─────────────────────────────────────────────────────────────────────────────
; CamSync Pro — Inno Setup Installer Script
; Requires: Inno Setup 6 — https://jrsoftware.org/isdl.php
; FFmpeg is bundled automatically — no user action required.
; ─────────────────────────────────────────────────────────────────────────────

#define AppName      "CamSync Pro"
#define AppVersion   "0.6.0"
#define AppPublisher "CamSync Pro"
#define AppExeName   "CamSyncPro.exe"

[Setup]
AppId={{A3F1C9D2-7B4E-4A0F-9C3D-E5F2B8A71234}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
DefaultDirName={autopf}\CamSyncPro
DefaultGroupName={#AppName}
AllowNoIcons=yes
OutputDir=installer_output
OutputBaseFilename=CamSyncPro_Setup_v{#AppVersion}
SetupIconFile=installer\icon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
MinVersion=10.0
WizardImageFile=installer\wizard_banner.bmp
WizardSmallImageFile=installer\wizard_icon.bmp
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main application executable (built by PyInstaller)
Source: "dist\CamSyncPro.exe"; DestDir: "{app}"; Flags: ignoreversion

; FFmpeg — bundled directly, no user install needed
Source: "ffmpeg\ffmpeg.exe"; DestDir: "{app}"; Flags: ignoreversion

; README
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppName}";              Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}";    Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";        Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; \
    Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
