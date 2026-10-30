; TRTForge / DefectVision AI Studio — Windows 安装包
; 需要先完整编译 TRTForge（使 4-qt-studio/bin 内含 exe、Qt 依赖、推理 DLL、TensorRT 等）
; 编译本脚本：安装 Inno Setup 6 后运行 build_installer.ps1，或在 IDE 中打开本文件并 Build

#define MyAppName "DefectVision AI Studio"
#define MyAppNameShort "TRTForge"
#define MyAppVersion "1.0.0"
#define MyAppExeName "TRTForge.exe"
#define MyAppPublisher "DefectVision"
; 相对本 .iss 文件：installer\..\bin
#define BinDir "..\\bin"

#ifndef MyAppId
#define MyAppId "{{7E3A9F2C-51D8-4B0E-A631-2F8C9E317A15}"
#endif

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppVerName={#MyAppName} {#MyAppVersion}
DefaultDirName={autopf}\{#MyAppNameShort}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=TRTForge_Setup_{#MyAppVersion}_x64
SetupIconFile=
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
CloseApplications=yes
RestartApplications=no
LicenseFile=
InfoBeforeFile=README_INSTALL_zh.txt
UninstallDisplayIcon={app}\{#MyAppExeName}
VersionInfoVersion={#MyAppVersion}

[Languages]
; 若本机 Inno 带 Languages\ChineseSimplified.isl，可自行追加一行中文语言
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; 打包已编译目录（含子目录 platforms、configs 等），排除调试符号以减小体积
Source: "{#BinDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.pdb,*.ilk,*.exp,*.lib,*.old"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "{#MyAppName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; Comment: "{#MyAppName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
