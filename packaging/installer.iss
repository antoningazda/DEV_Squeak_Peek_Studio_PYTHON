; Inno Setup script for Squeak Peek Studio (Windows installer).
; Built in CI via: ISCC.exe /DMyAppVersion=<version> packaging\installer.iss
; Expects the PyInstaller onedir build to already exist at dist\SqueakPeekStudio.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

[Setup]
AppId={{6C6F5B0A-6C79-4B7B-9F1E-3B6C6E6E6B1D}}
AppName=Squeak Peek Studio
AppVersion={#MyAppVersion}
AppPublisher=NUDZ
DefaultDirName={autopf}\Squeak Peek Studio
DefaultGroupName=Squeak Peek Studio
UninstallDisplayIcon={app}\SqueakPeekStudio.exe
SetupIconFile=..\src\squeak_peek\gui\assets\icon\app_icon.ico
OutputDir=installer_output
OutputBaseFilename=SqueakPeekStudio-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "..\dist\SqueakPeekStudio\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Squeak Peek Studio"; Filename: "{app}\SqueakPeekStudio.exe"
Name: "{autodesktop}\Squeak Peek Studio"; Filename: "{app}\SqueakPeekStudio.exe"; Tasks: desktopicon
Name: "{group}\Uninstall Squeak Peek Studio"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\SqueakPeekStudio.exe"; Description: "Launch Squeak Peek Studio"; Flags: nowait postinstall skipifsilent
