# MATLAB → Python Port Plan (Phases 2–6)

Frozen contracts (already in `main`, do not touch): `squeak_peek.config.*`,
`squeak_peek.labels.model.Label`, `squeak_peek.audio.io.{load_wav,save_wav}`,
`squeak_peek.audio.filters.{bandpass_filter,compute_stft,band_restrict}`,
`squeak_peek.detectors.base.AbstractDetector`.

Reference MATLAB source: `../DEV_Squeak-Peek-Studio/src/functions/`.
Golden data for validation: `../DEV_Squeak-Peek-Studio/data/`,
`tmp_calls_for_feature_compute.csv`, `usv_classification_results*.csv`.

Each WP below = one agent, one branch, touches only the files listed under
"Owns". No two WPs write the same file. Run in the dependency order shown;
WPs in the same tier can run fully in parallel.

---

## Tier 0 — fix first (blocking, ~5 min, do myself not an agent)
- `pip install -e ".[dev]"` in `.venv` — editable install currently broken,
  `pytest` can't `import squeak_peek`.
- Commit the pending GUI diff already on disk.

## Tier 1 — parallel, no interdependencies

### WP1 — PSD Detector
**Owns:** `src/squeak_peek/detectors/psd.py`, `tests/unit/test_psd.py`
**Port:** `PSDDetector.m` (uses `audio.filters.bandpass_filter` /
`compute_stft`, already implemented — don't reimplement).
**Interface:** `class PSDDetector(AbstractDetector)`, params from
`config.PSDParams`, `detect(signal, fs) -> list[Label]`.
**Decided: match the MATLAB technique exactly, not the existing stub.**
MATLAB uses `designfilt('bandpassiir', FilterOrder=12, ...)` +
`filtfilt` (zero-phase, order 12 IIR bandpass) and a **Hamming** window
STFT — our current `audio/filters.py` stub is Butterworth SOS order 5,
one-pass, Hann window, and is *not* what PSD (or BSCD/RBD, which use the
same `designfilt`+`filtfilt` bandpass) should call as-is. This WP owns
fixing that:
- Add `bandpass_filter_filtfilt(signal, fs, f_low, f_high, order=12)` to
  `audio/filters.py` using `scipy.signal.butter(..., output="sos")` +
  `scipy.signal.sosfiltfilt` (zero-phase, matches `filtfilt` behavior;
  `designfilt('bandpassiir')` is itself a Butterworth-type IIR design, so
  `butter`+`sosfiltfilt` is the correct scipy equivalent — order 12 is a
  *parameter*, not a hardcoded default, so WP2/WP3 can reuse it with their
  own bandpass order). Keep the existing `bandpass_filter` (order-5,
  one-pass) as-is for any caller still using it — this is an addition, not
  a replacement.
- Add a `window` param (default `"hann"`) to `compute_stft`, and have PSD
  pass `window="hamming"` — additive, backward compatible.
**WP2 and WP3 should import `bandpass_filter_filtfilt` from `audio/filters.py`
once WP1 lands it** — don't each reimplement the same bandpass. If WP1
hasn't landed yet when WP2/WP3 start, they may add it themselves; whoever
lands second rebases onto the other's version instead of duplicating it.

### WP2 — BSCD Detector
**Owns:** `src/squeak_peek/detectors/bscd.py` (core `bscd()` math +
`BSCDDetector` class), `tests/unit/test_bscd.py`
**Port:** `bscd.m` + `BSCDDetector.m`.
**Performance risk:** `bscd.m` is a per-sample recursive Bayesian update
(rank-1 matrix inverse updates) over the whole signal — at 250 kHz a
60 s clip is 15M samples. A naive Python transliteration will be orders of
magnitude too slow. **Decided: use `numba`** (`@njit` the inner per-sample
loop) — add `numba>=0.59` to `pyproject.toml` `dependencies` (whoever lands
first adds the line; second PR rebases). Still write the closed-form 2×2
update by hand rather than calling `np.linalg.inv` inside the jitted loop
(unsupported/slow in numba for tiny matrices — inline the 2×2 inverse
algebraically). Validate correctness on a **short** synthetic/real clip
(<2s) with plain numpy first, then jit, then confirm identical output.

### WP3 — RBD Detector
**Owns:** `src/squeak_peek/detectors/rbd.py` (core `rbd()` math +
`RBDDetector` class), `tests/unit/test_rbd.py`
**Port:** `RBD.m` + `RBDDetector.m`.
**Same performance risk as WP2**, worse: AR order 4 means 4×4+4×4 matrix
updates per sample instead of 2×2. **Decided: `numba` here too** — same
guidance as WP2 (closed-form/algebraic updates inside an `@njit` loop;
for order-4 this may mean using `numba`-supported `np.linalg.inv` on small
fixed-size arrays rather than hand algebra — try algebraic first, fall back
to numba's `np.linalg.inv` support if it's cleaner). Validate on short
clips first.

### WP4 — Feature extraction
**Owns:** `src/squeak_peek/features/extract.py`, `tests/unit/test_features.py`
**Port:** `extractUSVFrameFeatures.m` exactly — 12 columns, same order
(`BandPower, SpecCentroid, SpecSpread, SpecFlatness, SpecEntropy, ZCR,
SNR_est, SpecFlux, DomFreq, Delta_BandPower, Delta_Centroid, Delta_Entropy`).
**Interface:** `extract_frame_features(x, fs, frame_len, hop_len, nfft,
fmin, fmax) -> tuple[np.ndarray, np.ndarray]` (X, mid_times) — signature
mirrors MATLAB 1:1 so WP6/WP7 can call it without guessing.
**Validate against** `tmp_calls_for_feature_compute.csv` /
`usv_classification_results_with_features.csv` in the MATLAB repo if those
contain per-frame feature dumps — check column alignment first.

### WP5 — Label I/O, post-processing, metrics
**Owns:** `src/squeak_peek/labels/io.py`, `labels/postprocess.py`,
`labels/metrics.py`, `tests/unit/test_labels.py`
**Port:**
- `io.py` ← `importLabels.m` + `exportLabels.m` + `exportLabelsDetector.m`.
  **Decided: the MATLAB 2-line-per-label format (with frequency line) is
  the canonical one.** `load_labels`/`save_labels` in
  `gui/_label_io.py` (the current tab-separated 2/3-column stub) are
  superseded — this WP replaces `gui/_label_io.py`'s two functions with
  thin wrappers that import and re-export `labels.io.import_labels` /
  `labels.io.export_labels` (or delete `_label_io.py` and update its two
  call sites in the GUI directly — either is fine, but don't leave two
  competing label formats in the codebase). Match `exportLabelsDetector.m`
  as a second export function (`export_labels_detector`) for detector
  output specifically (fixed `"d"` label, zeroed frequencies).
- `postprocess.py` ← `mergeCloseLabels.m`, `removeShortLabels.m`,
  `padLabels.m`, `filterLowCentroidLabels.m`, `adaptiveStrongestFilter.m`
- `metrics.py` ← `compareLabels.m` (midpoint-matching TP/FP/FN/P/R/F1) —
  this replaces the ad-hoc IoU function currently inline in
  `gui/_tab_metrics.py`.
**No dependency on WP1–4.** Unblocks the CLI `evaluate` command and the
Metrics tab immediately.

---

## Tier 2 — depends on Tier 1

### WP6 — ML Detector (inference)
**Owns:** `src/squeak_peek/detectors/ml.py`, `tests/unit/test_ml_detector.py`
**Depends on:** WP4 (feature extractor), model artifact contract from WP7.
**Port:** `MLDetector.m`. MATLAB `TreeBagger` → **`sklearn.ensemble.
RandomForestClassifier`**; this is a new model format, not a `.mat` loader
— agree the on-disk contract with WP7 *before* starting:
```python
# joblib dump of a dict:
{"model": RandomForestClassifier, "feature_cols": list[str],
 "frame_params": {"frame_len_s": float, "hop_len_s": float,
                   "fcutMin": float, "fcutMax": float}}
```
Reproduce: sliding-window features → `predict_proba` → threshold at
`Sensitivity` → 3-frame median filter → binary→segments → drop events
under `minEventDuration`.

### WP7 — ML Training + Optimize
**Owns:** `src/squeak_peek/ml/train.py`, `ml/optimize.py`,
`tests/unit/test_ml_train.py`
**Depends on:** WP4 (feature extractor), WP5 (`labels/io.py` for ground
truth), same model contract as WP6.
**Port:** `MLDetectorTrain.m` (class balancing via undersampling,
`RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
oob_score=True)`, `random_state=42`) and `MLDetectorOptimize.m`
(sensitivity sweep + optional noise-ratio retraining sweep, using WP5's
`compare_labels`/metrics for F1, **not** a reimplementation of
`calculateStats`). No plotting required (GUI progress display is a
separate concern, not this WP's job) — return the numeric results only.

---

## Tier 3 — integration (small, sequential, after Tier 1+2 land)

### WP8 — CLI wiring
**Owns:** `cli/main.py`
Replace the `[Phase N pending]` stubs in `detect`, `batch`, `evaluate` with
real calls into WP1/2/3 detectors, WP5 post-processing + metrics.

### WP9 — GUI wiring
**Owns:** `src/squeak_peek/gui/_tab_detection.py`,
`src/squeak_peek/gui/_tab_metrics.py`
Wire `DetectionTab._run()` to the real detector classes (remove the "Not
yet implemented" dialog); wire `MetricsTab` to `labels.metrics.compare_labels`
instead of its inline IoU function.

---

**Current callers of the old format to update:** `gui/_tab_data_input.py`
and `gui/_tab_label_edit.py` both call `load_labels`/`save_labels` from
`gui/_label_io.py` — grep for them and update call sites if the file is
deleted rather than kept as a wrapper.

## Cross-cutting rules for every agent
1. Read the MATLAB source file(s) named above in full before writing code —
   don't work from this summary alone, it elides edge cases.
2. Match MATLAB's DC-removal / normalization order exactly
   (`x = x - mean(x); x = x / max(abs(x))`) — several detectors depend on
   this being done before filtering.
3. Every new detector/feature module needs at least one test that runs
   against a real or synthetic signal end-to-end, not just shape checks.
4. Don't touch `config.py`, `labels/model.py`, or another WP's files. If a
   shared function (e.g. `compute_stft`) needs a new parameter, add it
   backward-compatibly and say so in the PR description — don't change its
   existing default behavior.
5. Update `README.md`'s phase status table only for your own phase, in
   your own commit.
