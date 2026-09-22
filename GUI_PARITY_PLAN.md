# MATLAB → Python GUI/UX Parity Plan

`PORT_PLAN.md` covered the numerical core (detectors, features, labels, ML) —
that's done. This is a second deep-dive: a feature-by-feature comparison of
the **MATLAB App Designer UI** (`../DEV_Squeak-Peek-Studio/SqueakPeekStudio_exported.m`,
5063 lines) against the current PyQt6 GUI. The signal-processing engines match;
the *application* around them is a simplified reinterpretation, not a port.
Everything below is missing or behaves differently.

Frozen/shared, don't touch without flagging: `squeak_peek.config.*` (already
has every field these WPs need — this is a GUI-wiring gap, not a config
gap), `squeak_peek.labels.model.Label` is the one exception — WP20 extends
it, and every other WP that touches `Label` must rebase on that change.

Each WP = one agent, one branch, owns the files listed. Tier A items are
independent of each other except where noted. Read the cited MATLAB line
ranges in full before starting — this doc summarizes, it doesn't replace the
source.

---

## Gap inventory (what's actually missing)

| Area | MATLAB behavior | Python today |
|---|---|---|
| Keyboard shortcuts | ←/→ navigate segments/labels; `c`/`Shift+c` accept/reject classification; `d`/`Shift+d` accept/reject detection; `space` accepts both + advances (Label Edit tab only) | None at all |
| Label accept/reject | Two independent states per label — `DetectionState` and `ClassificationState`, each `None`/`Accepted`/`Rejected`; counters for all four; rejected labels are **kept**, not deleted | One "Accept"/"Reject" pair that just renames or **deletes** the label; no state concept |
| Export encoding | State round-trips through the label text as a 2-char suffix (`d_Dc`) + filename gets `_RD{n}_RC{m}` rejected-counts | No state to encode; plain export |
| Draggable labels | Label Edit spectrogram: start/end are `drawline` ROIs you literally drag to resize the label, live-updating the model | Static overlay lines only |
| Manual labels | Right-click (secondary click) on either spectrogram creates a new label centered on the click, sized by `ManualLabelLength` | No click handler |
| Sonification | `SonifyButton` → phase-vocoder pitch-shift (semitones) + time-stretch (custom algorithm, not a library call) → `audioplayer` playback with a live moving cursor on both waveform and spectrogram | Doesn't exist — no button, no audio playback anywhere |
| Spectrogram colormap | 15 selectable colormaps (`parula`, `turbo`, `hsv`, `jet`, `invgray`, …) | Hardcoded single reversed-grayscale, not selectable |
| Label colors | 7 colors each for detected/reference labels, dropdown-selectable | Hardcoded theme colors |
| Batch mode | Full second UI panel: folder-based USV/labels/reference-labels dirs, file-count validation, Detection tab iterates every WAV in the folder | Doesn't exist — single-file only |
| Multi-detector run | Multi-select list (run several detectors in one pass), multi-select post-processing (`None`/`Merge`/`Remove Short`) in chosen order, progress dialog with per-detector ETA + cancel, timestamped export filenames | Single detector via radio buttons, fixed two checkboxes, no progress/cancel, non-timestamped export (overwrites on rerun) |
| Settings coverage | Sub-tabs for Data Input, Visualization, PSD, BSCD, RBD, ML (model path + Sensitivity + min duration), Post-processing, Label Edit, Other/Theme | Only Visualization, PSD, Post-processing, and a System/Light/Dark toggle exist. **BSCD, RBD, ML, Label Edit, Data Input settings tabs don't exist** |
| Theme | `Light`/`Gray`/`Custom` presets; Custom reveals 4 `QColorDialog`-style pickers (Background/Text/Primary/Accent) bound to `AppSettings.Theme` | A separate, newer System/Light/Dark design-token system (`_theme.py`) unrelated to `config.ThemeSettings` — the two never merged |
| Startup | Auto-loads `settings/default.json` if present, then auto-loads the last-used files | `AppSettings.defaults()` only, no file ever touched, no auto-load of files |
| Info tab | Hyperlinks to thesis PDF, documentation, source repo, author email; CTU/NIMH logos; app + MATLAB version fields; logo click toggles a rat GIF (Easter egg) | Static text only, no links, no logos |

---

## Tier A — independent, parallel

### WP19 — Keyboard shortcuts
**Owns:** `src/squeak_peek/gui/app.py` (event filter / shortcut wiring only),
`src/squeak_peek/gui/_tab_visualization.py`, `_tab_label_edit.py`
**Port:** `SqueakPeekStudioUIFigureKeyPress` (`SqueakPeekStudio_exported.m:2497-2561`).
Implement as a `QMainWindow.keyPressEvent` (or `QShortcut`s scoped to the
active tab) dispatching to the same handlers the buttons already call — do
not duplicate accept/reject logic, call the same slot the button's
`clicked` signal calls. **Blocked on WP20 landing first** for the `c`/`d`/
`space` handlers (they need `DetectionState`/`ClassificationState`); the
←/→ segment/label navigation shortcuts have no such dependency and can land
immediately.

### WP20 — Label accept/reject state model + export encoding
**Owns:** `src/squeak_peek/labels/model.py` (add fields), `labels/io.py`
(encode/decode the suffix), `gui/_tab_label_edit.py` (rewrite the
accept/reject UI to match)
**Port:** `decodeStateSuffix` (`:954-1023`), `AcceptDetectionButtonPushed`/
`RejectDetectionButtonPushed` (`:2154-2192`), `AcceptClassificationButtonPushed`/
`RejectClassificationButtonPushed` (`:2562-2604`), `ExportLabelsButtonPushed`
(`:2193-2249`), `DetectionStateDropDownValueChanged` (`:2414-2440`),
`ClassificationStateDropdownValueChanged` (`:2650-2674`), `ShowClassificationEdit`
(`:837-857`).
This is the highest-impact WP — nearly everything else in the Label Edit
tab depends on it. Concretely:
- Add `detection_state: Literal["None","Accepted","Rejected"] = "None"` and
  `classification_state: Literal["None","Accepted","Rejected"] = "None"` to
  `Label`.
- `labels/io.py`: on export, append `f"_{det_code}{cls_code}"` to the label
  text (`D`/`d`/`x` × `C`/`c`/`x`, per the table in `ExportLabelsButtonPushed`);
  on import, detect a 2-char suffix after the last `_` and decode it back
  into the two state fields (`decodeStateSuffix`'s logic exactly — including
  the "not exactly 2 chars → leave alone" guard).
- `_tab_label_edit.py`: replace the single Accept/Reject pair with **two**
  independent actions (Accept/Reject Detection, Accept/Reject Classification),
  four running counters, and a "Rejected → free-text classification override"
  field (`ShowClassificationEdit`) that appears only when
  `classification_state == "Rejected"`. Rejecting no longer deletes the
  label — it's still shown, still exportable, just flagged.

### WP21 — Spectrogram colormaps + label colors
**Owns:** `src/squeak_peek/gui/_spectrogram_widget.py` (colormap builder),
`gui/_tab_settings.py` (dropdowns wired to `settings.visualization.colormap`/
`label_color`/`reference_label_color` — all three fields already exist in
`config.py`, this is pure GUI wiring)
**Port:** the 15-item list at `SqueakPeekStudio_exported.m:3894` and the
7-item label-color lists at `:3908`/`:3922`; `SpectrogramColormapDropDownValueChanged`
(`:2094-2101`).
Build each MATLAB colormap as a `pg.ColorMap` (MATLAB's `parula`, `turbo`,
`hsv`, `jet` etc. don't ship in pyqtgraph — use `matplotlib.cm` colormaps
where names match 1:1, e.g. `hsv`/`jet`/`hot`/`cool`/`spring`/`summer`/
`autumn`/`winter`/`gray`/`bone`/`copper`/`pink` are all matplotlib names
too; `parula` and `turbo` need their RGB lookup tables ported directly —
MATLAB ships `parula.m`'s table, and `turbo` is a published Google LUT,
both public). `invgray` is just `gray` flipped, matches the MATLAB special
case at `:1447`. Don't add a new dependency (`matplotlib`) without checking
it isn't already pulled in transitively — pyqtgraph doesn't need it, so
this may be a genuinely new dependency; flag it in the PR rather than
silently adding it, since it changes the packaged installer size.

### WP22 — Draggable label boundaries + manual label creation
**Owns:** `src/squeak_peek/gui/_spectrogram_widget.py` (add drag support +
click handler), `gui/_tab_label_edit.py`, `gui/_tab_visualization.py`
**Port:** `labelEditUpdateStartTime`/`labelEditUpdateEndTime` (`:790-819`),
`AnySpectrogramUIAxesButtonDown` (`:2441-2490`).
Use `pg.InfiniteLine(movable=True)` for the current label's start/end in
Label Edit, with `sigPositionChangeFinished` updating `Label.start_time`/
`end_time` on the model (mirrors the MATLAB ROI listener). For manual
label creation, connect to the plot's mouse-click signal, filter to
right-click (`event.button() == Qt.MouseButton.RightButton`), create a
`Label` centered on the clicked time with duration
`settings.visualization.manual_label_length`, insert it sorted by
`start_time` — matches `AnySpectrogramUIAxesButtonDown` exactly, including
that it fires from *both* the Visualization and Label Edit spectrograms.

### WP23 — Sonification
**Owns:** new `src/squeak_peek/audio/sonify.py`, `gui/_tab_visualization.py`
(Sonify button + playback), `tests/unit/test_sonify.py`
**Port:** `sonifySegment` (`:858-928`) and `SonifyButtonPushed` (`:2342-2391`)
**exactly** — this is a specific phase-vocoder implementation (pitch-shift
via STFT frame-timeline resampling, then a second independent time-stretch
pass, both via manual overlap-add), not equivalent to `librosa.effects.
pitch_shift`/`time_stretch` (different algorithm, will sound different).
Port the math as written: bandpass 40-120kHz → STFT with Hamming window →
pitch-shift by resampling the analysis-frame timeline by `2^(ST/12)` →
overlap-add → repeat STFT/resample/overlap-add for the `slowdown` factor →
normalize.
Playback: use `sounddevice` (already a dependency) instead of MATLAB's
`audioplayer`; the moving playback-position line on both waveform and
spectrogram needs a `QTimer` polling how many frames have played (sounddevice's
callback-based stream gives you this) rather than MATLAB's `isplaying()`
polling loop — same visual effect, different mechanism since Qt can't block
the event loop like MATLAB's `while isplaying(player)` does.

---

## Tier B — depends on Tier A pieces landing first

### WP24 — Batch mode
**Owns:** `src/squeak_peek/gui/_tab_data_input.py` (add batch panel + mode
toggle), `gui/_tab_detection.py` (iterate all WAVs in a folder)
**Depends on:** nothing structurally, but touches the same two files WP17
(below) also touches — coordinate order, don't run both at once.
**Port:** the `BatchModePanel` component tree (`:19-35` in the properties
block) + `SingleBatchModeButtonPushed` (`:1087-1118`) +
`checkFileCount` (`:759-789`) for the "N files found" validation labels, +
the batch branch of `RunDetectorsButtonPushed` (`:1710-1740`).
Add a Single/Batch toggle to Data Input; in Batch mode, three folder
pickers (USVs/labels/referenceLabels) replace the three file pickers, each
showing a live "N .wav found" / "N .txt found" count. Detection tab's Run
must then iterate every `.wav` in the folder, sorted by name, matching
`RunDetectorsButtonPushed`'s batch loop.

### WP25 — Multi-detector run pipeline
**Owns:** `src/squeak_peek/gui/_tab_detection.py`
**Depends on:** WP24 landing first (shares the batch-iteration code path;
avoid two agents rewriting `_run()` at once — sequence these two, don't
parallelize them against each other even though both are otherwise
independent).
**Port:** `RunDetectorsButtonPushed` in full (`:1639-1935`) — multi-select
`QListWidget` (Qt equivalent of MATLAB's multi-select `ListBox`) for
detectors (`PSD`/`BSCD`/`RBD`/`ML`) and post-processing
(`None`/`Merge Close Labels`/`Remove Short Labels`, applied in list order),
a `QProgressDialog` with a Cancel button wired to check between every
detector/file (MATLAB's `CancelRequested` poll), the same per-detector ETA
heuristic (`estTime = k * sigLen / 161999424` — the constants at `:1780-1787`
are MATLAB's own benchmark numbers; profile Python's actual speed instead of
reusing them verbatim, they won't transfer directly since the ported
detectors' performance characteristics differ, especially BSCD/RBD post-numba),
and timestamped export filenames (`{baseName}_{det}_{yyyyMMdd_HHMMSS}_detected.txt`)
instead of the current fixed name that overwrites on rerun.

### WP26 — Remaining Settings tabs (BSCD, RBD, ML, Label Edit, Data Input)
**Owns:** `src/squeak_peek/gui/_tab_settings.py`
**Port:** the corresponding MATLAB sub-tab component blocks + their
`*ValueChanged` handlers (all one-liners — `RBD_fCutMinEditFieldValueChanged`
etc. at `:2702-2900`, ML at `:4391-4444`). Every field these need already
exists on `config.py` (`RBDParams`, `BSCDParams`, `MLParams` — including the
`sensitivity` field that landed with the ML detector). This is pure
`QFormLayout` wiring, no new logic — follow the exact pattern already used
for the PSD tab in `_make_psd_tab`/`_apply`/`_reload_values`. Add a model
`QLineEdit` + Browse button + `QFileDialog` for `ml.modelPath` (`.joblib`
files instead of MATLAB's `.mat`, matches WP6/WP7's model format), and a
Sensitivity spinbox.

### WP27 — Theme presets (Light/Gray/Custom) reconciled with the design-token system
**Owns:** `src/squeak_peek/gui/_theme.py`, `gui/_tab_settings.py`
(Appearance sub-tab)
**Read first, don't guess:** a design-token light/dark theme system already
landed (`_theme.py`, System/Light/Dark toggle) *after* `PORT_PLAN.md`'s
Tier 3, unrelated to `config.ThemeSettings`. This WP's job is to decide
and implement how MATLAB's `Light`/`Gray`/`Custom` (4 raw RGB pickers,
`ThemeDropDownValueChanged` at `:1273-1318`) maps onto that system —
options: (a) treat `config.ThemeSettings` as a legacy/import-export-only
concern satisfied by the existing settings JSON round-trip and don't build
UI for it, since the new design-token system is arguably a *better*
replacement, not a gap; or (b) add a 4th "Custom" mode to `_theme.py` that
reads `QColorDialog`-picked colors into `config.ThemeSettings` and
generates tokens from them. **Don't implement (b) without checking with
the user first** — it's a real product decision (two theme systems doing
overlapping jobs), not a mechanical port.

---

## Tier C — polish, low priority, do last

### WP28 — Startup auto-load
**Owns:** `src/squeak_peek/gui/app.py`, `_state.py`
**Port:** `loadDefaultSettingsIfExists` (`:716-758`) + the
`app.LoadFilesButtonPushed()` call at the end of `startupFcn` (`:1034-1056`).
On launch, if `settings/default.json` exists relative to the app, load it;
if its `DataInput.DefaultUSVFieldSingle` etc. point at files that exist,
load them too. Keep this behind a try/except that degrades silently to
current behavior (empty state) — this is a convenience, not a requirement,
and must never crash startup on a stale/missing path.

### WP29 — Info tab content
**Owns:** `src/squeak_peek/gui/_tab_info.py`
**Port:** the Info tab component block (`:4900-5029` region) +
`OpenDocumentationButtonPushed`/`OpenMastersThesisButtonPushed`/
`OpenSourceCodeButtonPushed` (`:1237-1251`) — each just `web(url)`/`system(open ...)`,
port to `QDesktopServices.openUrl`. Add the CTU/NIMH logos (check
`assets/` in the MATLAB repo for source images, re-export at a reasonable
resolution) and the author-email `mailto:` hyperlink
(`AuthorEmailHyperlink`). The logo-click rat GIF swap (`ImageClicked`,
`:2328-2336`) is a pure Easter egg — nice-to-have, not required, do it last
if there's time.

---

## Suggested dispatch order

1. **WP20** first, alone (everything Label-Edit-shaped depends on it).
2. Once WP20 merges: **WP19, WP21, WP22, WP23** in parallel (Tier A, all
   independent of each other).
3. Then **WP24 → WP25** sequentially (shared file), **WP26** and **WP27**
   in parallel with those (different files).
4. **WP28, WP29** whenever, lowest priority.
