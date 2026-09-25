; PawCapture - Inno Setup 6 installer script (https://jrsoftware.org/isdl.php)
; build.bat runs this after PyInstaller and passes /DAppVersion=<ver>.
; Packages the whole portable folder dist\PawCapture\ (ffmpeg included).

#define AppName      "PawCapture"
#ifndef AppVersion
  #define AppVersion "0.8.0"
#endif
#define AppPublisher "PixelPaws"
#define AppURL       "https://github.com/rslivicki/PixelPaws"
#define AppExeName   "PawCapture.exe"

[Setup]
; Same AppId as the earlier "CamSync Pro" installer so lab machines upgrade in place.
AppId={{A3F1C9D2-7B4E-4A0F-9C3D-E5F2B8A71234}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\PawCapture
DefaultGroupName={#AppName}
AllowNoIcons=yes
OutputDir=installer_output
OutputBaseFilename=PawCapture_Setup_v{#AppVersion}
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
; The PyInstaller portable folder, verbatim (PawCapture.exe, ffmpeg.exe, Qt/Python runtime)
Source: "dist\PawCapture\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme skipifsourcedoesntexist

[InstallDelete]
; Leftovers from the single-file "CamSync Pro" builds this installer replaces
Type: files; Name: "{app}\CamSyncPro.exe"

[Icons]
Name: "{group}\{#AppName}";              Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}";    Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";        Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
