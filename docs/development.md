# Development

For contributors and anyone extending the application.

## Setting up

```bash
git clone https://github.com/antoningazda/DEV_Squeak_Peek_Studio_PYTHON.git
cd DEV_Squeak_Peek_Studio_PYTHON

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"          # PyTorch is a core dependency, nothing extra needed
```

Python **3.11 or 3.12**.

```bash
squeak-peek        # GUI
squeak-peek-cli    # CLI
```

## Project structure

```
src/squeak_peek/       Main installable package
    config.py          Pydantic settings (mirrors settings/default.json)
    plugins.py         Self-registering plugin base
    audio/             WAV I/O, filters, STFT, sonification
    detectors/         PSD, BSCD, RBD, ML, CNN
    features/          12-D acoustic features, tonality scoring
    labels/            Label I/O, post-processing, metrics
    ml/                Random Forest training & Optuna optimisation
    cnn/               Faster R-CNN training, dataset conversion
    usv_classifier/    CNN + RF call-type model
                       (core/ = vendored USV_Klasifikace pipeline)
    video/             Sync, spectrogram panel, ffmpeg export
    gui/               PyQt6 desktop application
cli/                   Click-based headless CLI
tests/                 pytest suite
settings/              default.json (shared with the MATLAB project)
packaging/             PyInstaller spec, installers, signing scripts
docs/                  This documentation site
data/                  Example WAV files and labels
```

## Tests and linting

```bash
pytest                       # the suite
pytest --cov=squeak_peek     # with coverage
ruff check src/ tests/ cli/  # lint
mypy src/                    # types (non-strict)
```

[CI](https://github.com/antoningazda/DEV_Squeak_Peek_Studio_PYTHON/actions)
runs the suite on Python 3.11 and 3.12 across macOS, Ubuntu and Windows,
plus a ruff pass. Linux runs the GUI tests under `Xvfb`.

Ruff config: line length 100, rules `E`, `F`, `I`, `UP`. The vendored
`usv_classifier/core/*` is exempt — it is kept byte-comparable with
upstream rather than reformatted.

---

## Writing a detector

Detectors are **self-registering plugins**. Everything downstream — the
Detection tab's list and tooltips, the CLI's `--detector` choices, the
Settings tab's parameter form — is generated from the registry. Adding one
means adding one file.

Create `src/squeak_peek/detectors/mydetector.py`:

```python
from pydantic import BaseModel, Field

from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.labels.model import Label


class MyParams(BaseModel):
    fcutMin: float = Field(
        40_000, ge=0, le=250_000,
        description="Lower bound of the analysed band.",
        json_schema_extra={"unit": "Hz", "group": "freq_band"},
    )
    threshold: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Detection sensitivity.",
    )


class MyDetector(AbstractDetector):
    id = "MYDET"
    display_name = "My detector"
    description = "One sentence shown as the tooltip and Settings caption."
    Params = MyParams

    def detect(self, signal, fs: int) -> list[Label]:
        ...
        return labels
```

That is the whole integration. The package's `__init__.py` imports every
module in the directory for its registration side effect.

Things the framework reads:

| Attribute | Used for |
|---|---|
| `id` | Registry key, CLI `--detector` value, settings key, exported filename |
| `display_name` | Settings sub-tab title |
| `description` | Tooltip in the Detection tab, caption in Settings |
| `Params` | Generates the Settings form — field types, ranges and descriptions all come from the pydantic model |

`json_schema_extra` keys the form understands: `unit`, `group`, `decimals`,
`caption`.

`PSDDetector` is the complete worked example.

## Writing a classifier

Same pattern, in `src/squeak_peek/classifiers/`. A classifier takes labels
a detector already produced plus the audio, and returns labels with the
`.label` field filled in.

```python
class MyClassifier(AbstractClassifier):
    id = "MYCLS"
    display_name = "My classifier"
    description = "..."
    Params = MyParams

    def classify(self, labels, signal, fs) -> list[Label]:
        ...
```

**A classifier never invents events** — it only types the ones it is given.
`classifiers/duration.py` is a minimal worked example.

## Settings

`config.py` holds the pydantic models mirroring `settings/default.json`.
Per-plugin parameters are stored as raw dictionaries captured through
pydantic's `extra` mechanism, so **a new plugin's parameters round-trip
through JSON with no schema change** — and a settings file from a build
with an extra detector loads cleanly here.

---

## Packaging

```bash
pip install -e ".[package]"
./build_app.sh
```

PyInstaller config: `packaging/SqueakPeekStudio.spec`. The Windows
installer is built from `packaging/installer.iss` (Inno Setup).

## Releases

Push a `v*` tag. `.github/workflows/release.yml` builds installers for
macOS, Windows and Linux and publishes them to the public releases repo,
[`antoningazda/squeak-peek-studio-releases`](https://github.com/antoningazda/squeak-peek-studio-releases).

The version lives in **one place**: `squeak_peek.__version__`.
`pyproject.toml` reads it dynamically, and so do the Info tab and the CLI.

### Code signing

Signing is **gated on repository secrets**. With none set, the workflow
produces an ad-hoc-signed `.app` (hardened runtime enabled) and an unsigned
Windows installer. With the secrets present, the same workflow produces a
Developer ID-signed, notarized and stapled DMG and an Authenticode-signed
installer. Adding certificates later needs no code change.

| Platform | No secrets | With secrets |
|---|---|---|
| macOS | Ad-hoc signature, hardened runtime | Developer ID, notarized, stapled |
| Windows | Unsigned | Authenticode-signed |
| Linux | Unsigned | n/a |

The hardened runtime is applied to ad-hoc builds too, so the unsigned path
exercises the same runtime restrictions the signed one will — a
hardened-runtime bug cannot hide until the day a certificate is bought.

See `packaging/SIGNING.md` for secret names and required entitlements.

---

## Documentation

This site is [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).
The source is `docs/` in this repository; CI builds it and publishes to the
releases repo's GitHub Pages.

```bash
pip install -r docs/requirements.txt
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build      # static site into site/
```

Docs live next to the code deliberately: a change to a detector's
parameters and the documentation of those parameters belong in the same
commit.

## Contributing

1. Branch from `main`.
2. Keep `pytest` and `ruff check src/ tests/ cli/` green.
3. Update the docs in the same commit as the behaviour change.
4. Open a pull request.

Issues and questions:
[GitHub issues](https://github.com/antoningazda/DEV_Squeak_Peek_Studio_PYTHON/issues).

## Licence

MIT. Original MATLAB application: Bc. Antonín Gazda, Master's thesis, CTU
Prague FEE, May 2025.
