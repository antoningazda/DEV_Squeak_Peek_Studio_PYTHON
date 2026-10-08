# Code signing & notarization

The release workflow signs what it can with whatever credentials are
configured and degrades gracefully when there are none. **With no secrets
set it still produces working installers** — they just carry the OS warnings
described in "Unsigned builds" below. Adding signing later is a secrets
change, not a code change.

| Platform | No secrets (today) | With secrets |
|---|---|---|
| macOS | Ad-hoc signature, hardened runtime | Developer ID signature, notarized, stapled |
| Windows | Unsigned | Authenticode-signed `.exe` + installer |
| Linux | Unsigned (normal for `.tar.gz`) | n/a |

---

## macOS

### What the build does

1. `packaging/sign_macos.sh` signs the bundle **inside-out**: every nested
   Mach-O first, then nested `.framework` bundles deepest-first, then the
   `.app` itself.

   It deliberately does not use `codesign --deep`. Apple documents `--deep`
   as a verification convenience and advises against signing with it — it
   applies the top-level entitlements to nested code and skips bundles it
   does not recognise. (`--deep` *is* correct for `--verify`, which is where
   the script uses it.)

2. Everything is signed with `--options runtime` (the hardened runtime),
   which notarization requires. This applies to ad-hoc builds too, so the
   unsigned path exercises the same runtime restrictions the signed one
   will — a hardened-runtime bug cannot hide until the day you buy a
   certificate.

3. `packaging/notarize_macos.sh` signs the DMG, submits it to Apple with
   `notarytool --wait`, and staples the ticket. Stapling matters: it embeds
   the ticket in the DMG so first launch works on a machine with no network
   path to Apple. With no identity configured the script exits 0 and does
   nothing.

### Entitlements

`packaging/entitlements.plist` must contain **no XML comments** — the
kernel's AMFI plist parser rejects them with `AMFIUnserializeXML: syntax
error`, which fails the build. The rationale therefore lives here:

- **`com.apple.security.cs.disable-library-validation` — verified required.**
  Without it the hardened runtime refuses to load the bundled
  `libpython*.dylib`: *"mapping process and mapped file (non-platform) have
  different Team IDs"*. Ad-hoc signatures carry no Team ID, so every bundled
  binary looks like a different team. A Developer ID build signs everything
  with one identity and might not strictly need this, but PyInstaller also
  `dlopen`s its own extension modules and the ctypes/cffi backends
  (`soxr`, portaudio via `sounddevice`), which is the usual reason
  PyInstaller apps keep it.

- **`com.apple.security.cs.allow-jit` and
  `com.apple.security.cs.allow-unsigned-executable-memory` — defensive.**
  numba `@njit` compiles the BSCD and RBD per-sample loops to machine code at
  runtime (`squeak_peek/detectors/{bscd,rbd}.py`). Tested on arm64, numba
  JITs fine under the hardened runtime *without* these, because llvmlite
  allocates RW and then `mprotect`s to RX rather than mapping W+X. They are
  kept for x86_64, where LLVM's allocation path historically does need
  them, and because the failure mode is a hard crash on the user's first
  detection run rather than something CI would catch.

Deliberately **not** requested:

- `com.apple.security.device.audio-input` — the app only ever plays audio
  back (`sd.play`); it never opens an input stream, so it needs no
  microphone access and should not prompt for it.
- `com.apple.security.cs.allow-dyld-environment-variables` — tested and not
  needed; it would let `DYLD_*` variables inject libraries into a signed
  process, which is a real (if small) weakening of the hardened runtime.

### Secrets to add

Apple requires a **paid** Developer Program membership ($99/yr) for a
Developer ID certificate. There is no free tier: a free Apple ID issues only
"Personal Team" development certificates, which cannot notarize and cannot
be distributed. Apple does operate a **fee waiver for nonprofits,
accredited educational institutions and government entities** in eligible
countries — worth checking whether NUDZ qualifies before paying personally.

Once you have the membership, export the *Developer ID Application*
certificate from Keychain Access as a `.p12`, then add these repository
secrets:

| Secret | Value |
|---|---|
| `MACOS_CERTIFICATE_P12` | `base64 -i cert.p12 \| pbcopy` |
| `MACOS_CERTIFICATE_PASSWORD` | the password set during the `.p12` export |
| `MACOS_SIGN_IDENTITY` | e.g. `Developer ID Application: Jane Doe (AB12CD34EF)` |
| `MACOS_NOTARY_APPLE_ID` | the Apple ID owning the membership |
| `MACOS_NOTARY_PASSWORD` | an **app-specific password** from appleid.apple.com — not the account password |
| `MACOS_NOTARY_TEAM_ID` | the 10-character team identifier |

The workflow imports the certificate into a throwaway keychain that is
deleted in an `if: always()` cleanup step, and calls
`security set-key-partition-list` so `codesign` does not block on an
interactive "allow access to key?" prompt no one can answer on CI.

---

## Windows

`packaging/sign_windows.ps1` signs the payload `.exe` before it is zipped or
fed to Inno Setup, and signs the installer after it is built. Bundled
DLLs/PYDs are left unsigned on purpose: SmartScreen judges the installer and
the launched `.exe`, and timestamping ~400 extra files costs a timestamp
server round-trip each for no user-visible gain.

| Secret | Value |
|---|---|
| `WINDOWS_CERTIFICATE_PFX` | base64-encoded `.pfx` code-signing certificate |
| `WINDOWS_CERTIFICATE_PASSWORD` | its password |
| `WINDOWS_TIMESTAMP_URL` | optional; defaults to `http://timestamp.digicert.com` |

Timestamping is not optional when a certificate is present. Without it every
signature stops validating the day the certificate expires — including on
copies users have already installed.

### Which certificate

There is no free option here either.

- **Azure Trusted Signing** (~$10/month) is the cheapest route and needs no
  certificate file or hardware token, but requires an Azure subscription and
  an organisation identity validation. If you go this way, replace the
  `sign_windows.ps1` call with Microsoft's `azure/trusted-signing-action`.
- **OV certificate** (~$200-400/yr): since the 2023 CA/Browser Forum rules
  the private key must live on a hardware token or cloud HSM, which does not
  fit the `.pfx`-in-a-secret model used here without a cloud-HSM option from
  the issuer. **An OV certificate also does not stop the SmartScreen warning
  immediately** — reputation accrues over downloads.
- **EV certificate** (~$400+/yr) is the only thing that clears SmartScreen
  from the first download.

For an internal lab tool, documenting the "More info → Run anyway" click is
a defensible alternative to any of these.

---

## Unsigned builds: what users see

Until certificates are configured, tell users this in the release notes.

**macOS** — "Squeak Peek Studio is damaged and can't be opened" or
"unidentified developer". The app is not damaged; macOS says this for any
app without a Developer ID signature. To open it:

1. Drag the app to `/Applications`.
2. Right-click (or Control-click) it and choose **Open**.
3. Click **Open** in the dialog.

Only needed once per installed version. If the dialog offers no Open button,
run `xattr -dr com.apple.quarantine "/Applications/Squeak Peek Studio.app"`
once in Terminal.

**Windows** — "Windows protected your PC". Click **More info**, then
**Run anyway**.

**Linux** — no signing expectations; `tar -xzf` and run.
