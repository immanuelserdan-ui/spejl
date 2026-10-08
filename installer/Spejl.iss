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

[Code]
// A Spejl.exe left running -- often invisible, e.g. after OmniBIM closed while
// hosting it -- keeps Spejl.exe and its DLLs locked, and copying failed halfway
// with "DeleteFile failed; code 5. Access is denied." Find any Spejl.exe of
// the current user (old versions included) and offer to close it first.

function SpejlRunning(): Boolean;
var
  ResultCode: Integer;
  Output: AnsiString;
  ListFile: String;
begin
  Result := False;
  ListFile := AddBackslash(GetTempDir) + 'spejl-setup-tasks.txt';
  if Exec(ExpandConstant('{cmd}'),
          '/C tasklist /NH /FI "IMAGENAME eq Spejl.exe" /FI "USERNAME eq ' + GetUserNameString + '" > "' + ListFile + '"',
          '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if LoadStringFromFile(ListFile, Output) then
      Result := Pos('spejl.exe', Lowercase(String(Output))) > 0;
  end;
  DeleteFile(ListFile);
end;

function CloseSpejl(): Boolean;
var
  ResultCode, Attempt: Integer;
begin
  Result := not SpejlRunning();
  if Result then
    exit;
  if SuppressibleMsgBox('Spejl is still running on this PC, possibly in the background without a window '
                        + '(for example after OmniBIM was closed).' + #13#10#13#10
                        + 'Setup needs to close it to update Spejl. Unsaved mirrored plans will be lost. '
                        + 'Close Spejl now?', mbConfirmation, MB_YESNO, IDYES) <> IDYES then
    exit;
  Exec(ExpandConstant('{sys}\taskkill.exe'),
       '/F /IM Spejl.exe /FI "USERNAME eq ' + GetUserNameString + '"',
       '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  for Attempt := 1 to 20 do
  begin
    if not SpejlRunning() then
    begin
      Result := True;
      exit;
    end;
    Sleep(250);
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if not CloseSpejl() then
    Result := 'Spejl is still running. Open Task Manager, go to the Details tab, end every '
              + 'Spejl.exe, then run Setup again.';
end;

function InitializeUninstall(): Boolean;
begin
  Result := CloseSpejl();
  if not Result then
    MsgBox('Spejl is still running. Open Task Manager, go to the Details tab, end every '
           + 'Spejl.exe, then run the uninstaller again.', mbError, MB_OK);
end;
