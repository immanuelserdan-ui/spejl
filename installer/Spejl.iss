#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef PayloadDir
  #define PayloadDir SourcePath + "payload"
#endif
#ifndef ReleaseDir
  #define ReleaseDir SourcePath + "..\releases\" + AppVersion
#endif

[Setup]
AppId={{8509B369-4A67-4DB3-AB58-2F25AB8D9DA4}
AppName=Spejl
AppVersion={#AppVersion}
AppVerName=Spejl {#AppVersion}
DefaultDirName={localappdata}\Programs\Spejl
DefaultGroupName=Spejl
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible and not arm64
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.19045
OutputDir={#ReleaseDir}
OutputBaseFilename=Spejl-{#AppVersion}-Windows-x64-Setup
SetupIconFile={#PayloadDir}\spejl-icon.ico
UninstallDisplayIcon={app}\Spejl.exe
UninstallDisplayName=Spejl
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
LZMAUseSeparateProcess=yes
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
VersionInfoVersion={#AppVersion}.0
VersionInfoDescription=Spejl full offline installer

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Spejl"; Filename: "{app}\Spejl.exe"; WorkingDir: "{app}"
Name: "{group}\Spejl - Getting started"; Filename: "{app}\GETTING-STARTED.txt"
Name: "{autodesktop}\Spejl"; Filename: "{app}\Spejl.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Spejl.exe"; Description: "Launch Spejl"; Flags: nowait postinstall skipifsilent
