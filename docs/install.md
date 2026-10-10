# Install

Squeak Peek Studio ships as a standalone application for macOS, Windows and
Linux. You do **not** need Python installed — the interpreter and every
dependency are bundled.

[Go to the latest release](https://github.com/antoningazda/squeak-peek-studio-releases/releases/latest){ .md-button .md-button--primary }

## Download

Every release publishes these files:

| File | Platform | Notes |
|---|---|---|
| `SqueakPeekStudio-macOS.dmg` | macOS, Apple Silicon (M1 and later) | Disk image — drag to Applications |
| `SqueakPeekStudio-macOS-Intel.dmg` | macOS, Intel | Disk image — drag to Applications |
| `SqueakPeekStudio-windows-Setup.exe` | Windows 10/11 | Installer, adds a Start-menu entry |
| `SqueakPeekStudio-windows.zip` | Windows 10/11 | Portable — unzip and run, no installer |
| `SqueakPeekStudio-linux.tar.gz` | 64-bit Linux | Extract and run the binary |

=== "macOS"

    1. Download `SqueakPeekStudio-macOS.dmg` (Apple Silicon) or
       `SqueakPeekStudio-macOS-Intel.dmg` (Intel Mac) and open it. Not sure
       which? Apple menu → **About This Mac** → *Chip* (Apple M…) or
       *Processor* (Intel).
    2. Drag **Squeak Peek Studio** into your **Applications** folder.
    3. **The first launch is special.** Right-click (or Control-click) the
       app and choose **Open**, then click **Open** in the dialog.

    !!! warning "“Squeak Peek Studio is damaged and can't be opened”"

        The app is not damaged. Current builds are not yet signed with an
        Apple Developer ID, and macOS shows this message for any such app.
        Right-click → **Open** is the normal way past it, and you only need
        to do it once per installed version.

        If the dialog gives you no **Open** button, run this once in
        Terminal:

        ```bash
        xattr -dr com.apple.quarantine "/Applications/Squeak Peek Studio.app"
        ```

=== "Windows"

    **With the installer**

    1. Download and run `SqueakPeekStudio-windows-Setup.exe`.
    2. SmartScreen will say **“Windows protected your PC”**. Click
       **More info**, then **Run anyway** — the builds are not yet
       Authenticode-signed.
    3. Follow the installer; launch from the Start menu.

    **Portable**

    Download `SqueakPeekStudio-windows.zip`, extract it anywhere, and run
    `SqueakPeekStudio.exe` from the extracted folder.

=== "Linux"

    ```bash
    tar -xzf SqueakPeekStudio-linux.tar.gz
    cd SqueakPeekStudio
    ./SqueakPeekStudio
    ```

    The bundle needs the usual Qt runtime libraries. On Debian/Ubuntu:

    ```bash
    sudo apt-get install -y libegl1 libopengl0 libxkbcommon0 \
      libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
      libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
      libxcb-xinerama0 libdbus-1-3
    ```

    For audio playback you also want a working PortAudio/ALSA setup
    (`libportaudio2`).

---

## Install from source

Use this if you want the CLI, want to train models with the optional neural
network components, or intend to modify the code.

**Requirements:** Python 3.11 or 3.12.

```bash
git clone https://github.com/antoningazda/DEV_Squeak_Peek_Studio_PYTHON.git
cd DEV_Squeak_Peek_Studio_PYTHON

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .
```

This gives you two commands:

```bash
squeak-peek          # launch the desktop application
squeak-peek-cli      # headless batch processing
```

### Neural-network components

Nothing extra to do. The [CNN detector](methods.md#cnn-faster-r-cnn) and the
[call-type classifier](guide/classification.md) need PyTorch, and
`pip install -e .` already installs it — torch and torchvision are core
dependencies, because the Classification tab is a core feature. The
downloadable installers ship with PyTorch inside too.

!!! note "The old `cnn` extra still works"

    PyTorch used to be an optional `cnn` extra. The extra is still there as
    an empty alias so existing install commands and scripts keep working —
    it just does nothing now.

### Optional extras

| Extra | Install | Gives you |
|---|---|---|
| `dev` | `pip install -e ".[dev]"` | pytest, ruff, mypy, hypothesis |
| `package` | `pip install -e ".[package]"` | PyInstaller, for building the standalone bundles |

### Verify the install

```bash
squeak-peek-cli --version
squeak-peek-cli detect data/example/single/USV_Example_Short.wav --detector psd
```

The second command writes a label file next to the example recording and
prints how many calls it found.

---

## Where your settings live

Squeak Peek Studio reads its defaults from a JSON settings file
(`settings/default.json` in a source checkout; bundled inside the app
otherwise). The **Settings** tab edits it, and you can load or save settings
files per experiment. See [Settings](guide/settings.md) and
[File formats](file-formats.md#settings-json).

Window geometry, theme and keyboard shortcuts are stored separately by the
operating system (via Qt's `QSettings`), so they survive upgrades
independently of the JSON file.
