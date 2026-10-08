# Methods

What each detector actually computes, which parameters matter, and when to
prefer one over another.

All five detectors share a frequency band — `fcutMin` / `fcutMax`, default
**40–120 kHz**. Everything outside it is discarded before analysis. This is
a *detection* setting, unlike the display range in
[Visualization](guide/visualization.md).

## Choosing a detector

| | Needs a model | Speed | Strength | Weakness |
|---|---|---|---|---|
| [PSD](#psd) | no | fastest | Clear, high-SNR recordings | Noise-sensitive; soft onsets |
| [BSCD](#bscd) | no | fast | Onsets/offsets in noisier audio | More false positives on transients |
| [RBD](#rbd) | no | slow | The most precise boundaries | Expensive; most parameters to tune |
| [ML](#ml-random-forest) | yes | fast | Learns your noise, not just energy | Only as good as its training data |
| [CNN](#cnn-faster-r-cnn) | yes | medium | Predicts frequency extent too | Needs real training data |
| [PITCH](#pitch-trace) | no | fast | Immune to broadband noise; reports a real frequency | Misses faint calls; splits calls whose contour drops out |

**Start with PSD.** If it misses soft calls or clips onsets in noisy
recordings, try BSCD. Reach for RBD when boundary accuracy is the point.
Train an ML or CNN detector when the classical detectors keep firing on a
noise source specific to your setup. Use PITCH as a second opinion when
cage noise is the problem: it keys on a coherent frequency contour rather
than on energy, so knocks and rustle never trigger it.

Whichever you pick, read [Tonality filtering](#tonality-filtering) — it is
the cheapest large accuracy gain available. PSD also needs
[Pre-detection denoising](#pre-detection-denoising), which is on for it by
default — on raw recordings its in-band noise floor costs most of its
precision.

---

## PSD

**Power Spectral Density** — thresholds band power against an adaptive
noise floor. Fast and robust on clear, high-SNR recordings.

The pipeline:

1. DC removal and amplitude normalisation.
2. Zero-phase bandpass filter (order-12 IIR, `filtfilt`).
3. Hamming-window STFT, matching MATLAB's
   `spectrogram(x, hamming(L), round(L*overlap), L, fs)`.
4. Power envelope: the power spectrum summed across the band and
   normalised to its own maximum.
5. Noise floor estimated by a **moving minimum**.
6. **Adaptive threshold** from local SNR statistics.
7. Binary thresholding and edge detection to get events.
8. Rejection of events below a minimum effective power.

The adaptive noise floor is what makes it work on real recordings: a fixed
threshold fails the moment the ventilation changes.

The envelope is summed straight from the power spectrum, with no round trip
through decibels. A dB detour needs an `+eps` floor to keep `log10(0)`
finite, and on a quiet recording that floor — not the signal — would set the
noise floor and therefore the SNR term of the threshold.

A detection that occupies a single STFT frame is kept here, as in
`PSDDetector.m`, and left for **Remove Short Labels** to drop if you want it
gone. That keeps the decision about what is too short in one place, with a
threshold in seconds, instead of hiding it in the detector's frame grid.

### Parameters

| Parameter | Default | What it does |
|---|---|---|
| `segmentLength` | `8192` | STFT window length for the power spectrum |
| `overlapFactor` | `0.59` | Fraction of each window overlapping the next |
| `maWindow` | `3` | Moving-average smoothing of the power envelope (frames) |
| `noiseWindow` | `240` | Moving-minimum window estimating the noise floor (frames) |
| `localWindow` | `194` | Window for local mean/std in the adaptive threshold (frames) |
| `k` | `0.023` | Scales the local-statistics term of the threshold |
| `w` | `0.994` | Weight on the local SNR term |
| `minEffectivePower` | `8.5e-05` | Minimum mean power to accept a candidate |
| `denoise` | `true` | Run on [denoised](#pre-detection-denoising) audio |
| `runWholeSignal` | `true` | Process the full signal; off restricts to an ROI |
| `ROIstart` / `ROIlength` | `60` / `10` s | The ROI, used only when `runWholeSignal` is off |

### Tuning

- **Missing quiet calls?** Lower `minEffectivePower`, then lower `w`.
- **Too many false positives?** Raise `w` first — it directly controls how
  far above the local noise statistics a frame must sit.
- **Noise floor drifting with a slow background change?** Shorten
  `noiseWindow` so the moving minimum tracks it.
- **ROI mode** is for development: restrict to a 10-second stretch to try
  parameters quickly, then turn `runWholeSignal` back on.

---

## BSCD

**Bayesian Sequential Change Detection** — flags points where the signal's
statistics shift abruptly, rather than points where it is loud. Good for
call onsets and offsets in noisier audio.

The pipeline:

1. DC removal and amplitude normalisation.
2. Zero-phase bandpass filter (order-12 IIR, `filtfilt`).
3. The change statistic, computed on the **squared** filtered signal.
4. Moving-average smoothing over `maWindow` samples.
5. Thresholding (see [Threshold mode](#threshold-mode)) and edge detection.

### The change statistic

At every sample, BSCD slides a window of `wlen` seconds and asks which of
two models explains it better:

- a **step**: one mean before the centre of the window, another after it;
- a **constant**: a single mean across the whole window.

The score is the log Bayesian evidence for the step over the constant, so a
peak means "the signal's statistics changed *here*" — which is why BSCD
finds the start of a call that rises gradually out of the noise, exactly the
case where an energy threshold clips the onset. The flip side is that any
abrupt transient is a change, which is why BSCD benefits most from
[tonality filtering](#tonality-filtering).

Both models are updated by rank-1 (Sherman–Morrison) steps as the window
slides, so the cost is linear in the number of samples rather than
quadratic, and the loop is JIT-compiled with numba. A recursive
sample-by-sample statistic therefore still runs at usable speed on 250 kHz
recordings.

This is a line-for-line port of the original `bscd.m` (Čmejla), down to
where the MATLAB loop leaves the first and last half-window untouched.

### Threshold mode

| Mode | What it does |
|---|---|
| `mean` (default) | One global threshold at the mean of the smoothed statistic — the rule `BSCDDetector.m` uses |
| `adaptive` | Local noise floor plus an SNR-weighted term, the same scheme [PSD](#psd) uses |

`mean` is the original detector and what the shipped settings use.
`adaptive` is an addition: the MATLAB version declared `noiseWindow`,
`localWindow`, `k` and `w` but never used them, and this mode is what those
four parameters drive. **In `mean` mode they do nothing.**

Try `adaptive` when the recording's background level drifts over the
session — a single global mean cannot follow that, while a local floor can.
Expect to retune `k` and `w` when you switch; their shipped values were not
fitted for this mode.

### Parameters

| Parameter | Default | What it does |
|---|---|---|
| `wlen` | `0.008` s | Sliding analysis window for detecting statistical change |
| `maWindow` | `7500` | Samples averaged to smooth the change statistic |
| `thresholdMode` | `mean` | `mean` or `adaptive` (above) |
| `noiseWindow` | `256` | Moving-minimum window for the noise floor — **`adaptive` only** |
| `localWindow` | `256` | Window for local mean/std — **`adaptive` only** |
| `k` | `0.023` | Scales the local-statistics term — **`adaptive` only** |
| `w` | `0.994` | Weight on the local SNR term — **`adaptive` only** |
| `denoise` | `false` | Run on [denoised](#pre-detection-denoising) audio — lowers BSCD's precision on the reference recordings |

### Tuning

- `wlen` sets the timescale of change BSCD is sensitive to. Shorter finds
  sharper onsets and more noise; longer is steadier but blurs fast calls.
- `maWindow` is the main dial in the default `mean` mode: it sets how much
  the statistic is smoothed before being compared against its own mean, and
  so how readily a short burst crosses.
- The windows are in **samples, not seconds**, and the defaults are large
  because the statistic is computed per sample. Scale them if you change
  the sample rate substantially.
- In `adaptive` mode, `w` is the main precision/recall dial, as in PSD.

Smoothing follows MATLAB's `movmean`/`smoothdata` convention, where the
window *shrinks* at the edges rather than padding — so the first and last
`maWindow/2` samples are not biased towards zero by an imaginary silent
run-up.

---

## RBD

**Relative Bayesian Difference** (Recursive Bayesian Detector) — at each
sample, fits one autoregressive (AR) model to the half-window on the left
and another to the half-window on the right, and compares that piecewise
model against a single AR model over the whole window. A point where two
separate models explain the data much better than one is a change point.

The pipeline:

1. DC removal and amplitude normalisation.
2. Zero-phase bandpass filter to `fcutMin`–`fcutMax` (order-12 IIR) —
   `bandpass`, on by default.
3. The change statistic: `4 · (ln res_single − ln(res_left + res_right))`,
   the log ratio of the least-squares residual energies, at every sample.
   Both systems are updated recursively as the window slides (a line-for-line
   port of `RBD.m`, JIT-compiled with numba).
4. Normalisation to its maximum and moving-average smoothing over
   `smoothingWindowRBD`.
5. Thresholding (`thresholdMode`) and edge detection.

### Two departures from `RBDDetector.m`

Both are switchable back to the MATLAB behaviour:

| Parameter | Default | MATLAB | Why |
|---|---|---|---|
| `bandpass` | `true` | `false` | Without it the AR models are fitted to the full-band signal, where they mostly describe noise below the USV band |
| `thresholdMode` | `median` | `original` | `original` thresholds relative to the statistic's **global maximum** (`dynamicScaling`, `amplitudeThreshold`), so one loud transient moves the threshold for the whole recording. `median` flags samples above `medianFactor` × the smoothed statistic's median, which the sparse calls barely move |

On the reference recordings the bandpass alone took RBD from F1 ≈ 0.25–0.4
to ≈ 0.7; the median threshold keeps that while being far less sensitive to
the recording's loudest event. The defaults below were also checked for
**label length**: the midpoint criterion used by
[Metrics](guide/metrics.md) does not penalise over-long detections, so
settings were chosen on both midpoint F1 and an overlap criterion
(IoU ≥ 0.3). Longer windows and heavier smoothing score well on midpoints
only because they stretch every detection.

### Parameters

| Parameter | Default | What it does |
|---|---|---|
| `wlen` | `0.02` s | Window compared on either side of each candidate boundary (MATLAB default `0.04`) |
| `AR_order_left` | `4` | AR model order left of the boundary |
| `AR_order_right` | `4` | AR model order right of the boundary |
| `Bayesian_Evidence_order` | `4` | AR order of the single-model baseline |
| `bandpass` | `true` | Bandpass to `fcutMin`–`fcutMax` before fitting |
| `thresholdMode` | `median` | `median` or `original` (above) |
| `medianFactor` | `4.0` | Threshold as a multiple of the median — **`median` only** |
| `smoothingWindowRBD` | `0.03` s | Smoothing of the statistic before thresholding (MATLAB default `0.02`) |
| `dynamicScaling` | `0.3` | Scales the moving-mean threshold — **`original` only** |
| `smoothingWindowThr` | `0.02` s | Smoothing of that threshold — **`original` only** |
| `amplitudeThreshold` | `0.02` | Minimum normalised statistic — **`original` only** |
| `denoise` | `false` | Run on [denoised](#pre-detection-denoising) audio — no gain once the bandpass is on |

### Tuning

- **`medianFactor`** is the main precision/recall dial: higher finds fewer,
  more confident calls. The optimum on the reference recordings was flat
  between 4 and 6.
- **`wlen`** also sets how sharp the boundaries are: the statistic responds
  within about `wlen`/2 of a change, so a longer window stretches every
  detection.
- **`smoothingWindowRBD`** merges a call's onset and offset peaks into one
  detection; too short splits calls, too long merges neighbours.
- **AR orders** are the model-complexity dial; 4 is a sensible default for
  USV-band audio, and going much above 8 rarely helps.
- RBD is the slowest detector (each sample updates several 8 × 8 systems).
  If it is too slow, lengthen `wlen` before lowering the AR orders.

Smoothing follows MATLAB's `movmean` convention, shrinking at the edges
instead of padding.

---

## ML (Random Forest)

A sliding-window **Random Forest** that classifies each frame as call or
noise from a 12-dimensional acoustic feature vector, then merges
consecutive positive frames into events.

The features, per frame:

| | | |
|---|---|---|
| BandPower | SpecCentroid | SpecSpread |
| SpecFlatness | SpecEntropy | ZCR |
| SNR_est | SpecFlux | DomFreq |
| Delta_BandPower | Delta_Centroid | Delta_Entropy |

The three delta features give the classifier a sense of change over time,
which is what distinguishes a frequency-modulated whistle from steady
broadband noise with the same instantaneous spectrum.

### Parameters

| Parameter | Default | What it does |
|---|---|---|
| `modelPath` | — | Path to the trained `.joblib` model. **Required** |
| `sensitivity` | `0.42` | Frame-probability cutoff for calling a frame a call |
| `minEventDuration` | `0.003` s | Shortest event kept after merging adjacent frames |

`sensitivity` is a direct precision/recall dial that needs no retraining:
lower it to find more calls, raise it to find fewer and cleaner ones.

→ [Training an ML detector](training.md#ml-detector-random-forest)

---

## CNN (Faster R-CNN)

A **Faster R-CNN** object detector run over spectrogram tiles. Unlike the
ML detector's frame classifier, it predicts a call's full box directly —
start time, end time, start frequency **and** end frequency — the way
DeepSqueak's own detector works.

Needs PyTorch, which `pip install -e .` already installs (see
[Install](install.md#neural-network-components)). The bundled desktop app
includes it too.

### Parameters

| Parameter | Default | What it does |
|---|---|---|
| `modelPath` | — | Path to the trained `.pt` checkpoint. **Required** |
| `sensitivity` | `0.5` | Box-score cutoff for accepting a predicted call |
| `minEventDuration` | `0.003` s | Shortest event kept after merging overlapping tile detections |

The detector tiles the recording, runs the network on each tile, and merges
boxes that overlap across tile boundaries.

→ [Training a CNN detector](training.md#cnn-detector-faster-r-cnn)

---

## Pitch trace

The detector behind the orange contour in
[Visualization](guide/visualization.md). It reports one event per stretch
of audio where a **coherent frequency contour** stands out from the
background — so what you see traced on screen is exactly what it detects.

Each STFT bin is referenced to its own median over the analysed block,
removing stationary noise lines; each frame is then referenced to its own
median across frequency. A frame counts as part of a call when its
strongest bin stands `prominenceDb` above that. Within each run of call
frames the contour is a **Viterbi path** that trades bin power against
frequency jumps — these calls often show several parallel bands of similar
power, and a plain per-frame argmax hops between them.

That last step is what makes the detector noise-immune in a way the
energy-based detectors are not: a knock or a rustle is loud across all
frequencies at once, so no single bin stands out from the frame's own
median, and no contour forms. The cost is symmetric — a call too faint for
the contour to lock on is simply not there, and a call whose contour drops
out mid-way is reported as two events.

It is also the **only detector that writes a real frequency per call**
(the others leave both frequency fields at 0), which is what makes the
call-band filter below possible.

### Parameters

| Parameter | Default | What it does |
|---|---|---|
| `fcutMin` / `fcutMax` | `40` / `120` kHz | The band searched for a contour |
| `segmentLength` | `1024` samples | STFT window. Matches Visualization's default, so events line up with the trace. Large windows slow the contour search considerably |
| `overlapFactor` | `0.5` | Finer boundaries at the cost of speed |
| `prominenceDb` | `12` dB | How far a frame's strongest bin must stand above the background |
| `minRunFrames` | `3` | Shortest run of call frames accepted as an event |
| `maxJumpKhz` | `6` kHz | Largest frequency change allowed between two frames |
| `jumpPenaltyDbPerKhz` | `2` dB/kHz | Cost per kHz of frequency change when choosing the contour |
| `maxContourFreqKhz` | `100` kHz | Bins above this are ignored when searching |
| `minCallFreqKhz` / `maxCallFreqKhz` | `0` / `250` kHz | Discard events whose contour sits outside this band |
| `minDurationMs` | `0` ms | Discard events shorter than this |
| `blockSeconds` | `20` s | Block length the recording is processed in |

### The call-band filter

`minCallFreqKhz` / `maxCallFreqKhz` throw away whole events whose **median
contour frequency** falls outside the band — "drop anything above 100 kHz
or below 50 kHz".

This is not the same as narrowing `fcutMin`/`fcutMax` to the same range.
The search band changes what the tracker *sees*: narrow it and the
background median is estimated from fewer bins, and a contour near the new
edge can be pulled onto it. The call-band filter runs **after** the contour
has been found over the full search band, so it only decides what to keep.
Prefer it when you want to exclude a frequency range; leave the search band
wide.

### Blocks

Long recordings are processed in `blockSeconds` blocks so the whole
spectrogram is never in memory at once. Blocks overlap by one window, so a
call sitting on a seam is seen intact in at least one of them; the
duplicate that produces is merged away afterwards. Each block's noise
background is estimated from that block, so shorter blocks adapt faster to
drifting noise.

---

## Pre-detection denoising

An optional stage that suppresses stationary background noise in the
**audio** before a classical detector runs. It is a **per-detector**
choice — each of PSD, BSCD and RBD has its own `denoise` parameter (also a
checkbox per detector under **Detection → Pre-processing**) — because it
helps one detector and hurts another:

| Detector | `denoise` default | Effect on the reference recordings |
|---|---|---|
| PSD | **on** | Pooled F1 0.47 → 0.77; precision 0.35 → 0.85 |
| BSCD | off | Pooled F1 0.75 → 0.62 (more false positives) |
| RBD | off | No gain once RBD's own bandpass is on |

### The idea

An ultrasonic recording is mostly background: fans, electronics, the
microphone's own hiss. That background is roughly constant over a session,
while the calls are sparse and brief — so each frequency bin's noise level
can be estimated **from the recording itself**, as the median of that bin's
magnitude over time, and subtracted. It is spectral subtraction, the same
idea as Audacity's Noise Reduction, with the noise profile taken
automatically instead of from a selection you make by hand.

PSD sums band power, so on raw audio that constant in-band noise is most of
its envelope, and its precision collapses. The PSD results in the original
MATLAB thesis were obtained on recordings denoised beforehand; with this
stage on, the Python PSD reaches the same F1 directly from raw recordings.

### How it works

1. **Noise profile.** The `noisePercentile`-th percentile of each rFFT bin's
   magnitude, over up to 20 000 Hann-windowed frames spread evenly across
   the recording — so the estimate costs the same for a 2-minute file as for
   an hour-long one.
2. **Soft subtraction.** Each bin is scaled by
   `(|X| − oversubtraction · N) / |X|`, clipped to a floor of
   `−maxReductionDb`. A floor rather than a hard gate: the residual stays a
   quiet hiss instead of the warbling "musical noise" a gate produces.
3. **Smoothing.** The gain is smoothed over 3 bins × 5 frames before it is
   applied, for the same reason.
4. **Resynthesis.** Weighted overlap-add (periodic Hann, hop = `nfft`/4),
   block by block, so memory stays bounded on hour-long recordings. The
   output is the same length and sample rate as the input, so any detector
   runs on it unchanged.

The denoised audio is computed once per recording and shared by every
selected detector that wants it.

### Parameters

[Settings → Pre-processing](guide/settings.md#pre-processing), stored under
`Detection.PRE` and shared by every detector with `denoise` on:

| Parameter | Default | What it does |
|---|---|---|
| `nfft` | `1024` samples | STFT window for estimation and suppression |
| `noisePercentile` | `50.0` | Percentile of each bin's magnitude taken as its noise level |
| `oversubtraction` | `1.25` | Multiple of that level subtracted |
| `maxReductionDb` | `30.0` dB | Largest attenuation any bin may receive |

The defaults were chosen on the reference recordings for PSD; a lower
percentile or a smaller `maxReductionDb` removes less noise.

### Scope

- **ML and CNN** ignore the per-detector checkboxes and apply the denoising
  recorded in their own model file at training time (**Train detector →
  Denoise the training audio**), so training and inference always see the
  same kind of audio.
- **Detection only.** Export, [Label Edit](guide/label-edit.md),
  sonification, the video soundtrack and
  [call-type classification](guide/classification.md) all use the original
  recording. Nothing you listen to, look at or ship is denoised.

!!! warning "Retune if you change it"

    A detector's thresholds describe a particular noise level. Switching a
    detector's `denoise` changes that level, so its other parameters are no
    longer fitted — re-run **Detection → Train detector → Tune detector
    parameters** (tuning applies the detector's `denoise` setting), or score
    the change against reference labels in [Metrics](guide/metrics.md).

---

## Tonality filtering

The single largest accuracy gain available, and it costs nothing but a
checkbox.

### The idea

A rodent USV is a **narrowband frequency-modulated whistle**: within any
short frame, nearly all of its energy sits close to one peak frequency. The
false positives that dominate energy- and change-point detectors — cage
knocks, bedding rustle, scratching — are **broadband**, with energy smeared
across the band.

The tonality score measures exactly that difference: the fraction of a
frame's energy falling within ±5 kHz of its peak frequency, averaged over
frames and weighted by frame energy. It runs from 0 to 1 — near 1 for a
pure tone, near the bandwidth's share of the band for broadband noise.

### Results

Measured on held-out USVSEG mouse recordings, with the threshold chosen on
separate training recordings:

| Detector | | Precision | Recall | F1 |
|---|---|---|---|---|
| PSD | off | 0.465 | 0.707 | 0.561 |
| PSD | `--min-tonality 0.5` | **0.934** | 0.675 | **0.784** |
| BSCD | off | 0.628 | 0.827 | 0.714 |
| BSCD | `--min-tonality 0.5` | **0.899** | 0.763 | **0.825** |

PSD's precision doubles. Recall falls by about three percentage points.

### Using it

=== "GUI"

    1. [Detection](guide/detection.md) tab → tick **Filter Broadband** in
       post-processing.
    2. [Settings](guide/settings.md#post-processing) → Post-processing →
       set **Min tonality** to `0.5`.

=== "CLI"

    ```bash
    squeak-peek-cli detect audio.wav --detector bscd --min-tonality 0.5
    ```

### Why it is off by default

It trades recall for precision, and which side of that trade you want
depends on your analysis. Two design choices limit the damage:

- It runs **before** merging, so a merged detection spanning the gap
  between two real calls is not penalised.
- Segments too short to score, or with no in-band energy, are **kept**. The
  filter only removes detections it can positively judge as broadband.

!!! warning "Not for every species or setup"

    The threshold of 0.5 was tuned on mouse recordings. Calls that are
    genuinely noisy or harmonically rich score lower, and a species whose
    repertoire includes broadband calls will lose them. Check the effect on
    a recording you have ground truth for — via [Metrics](guide/metrics.md)
    — before applying it to a whole dataset.

---

## The full chain

```
audio ─▶ [Denoise] ─▶ detector ─▶ Filter Broadband ─▶ Merge Close Labels ─▶ Remove Short Labels ─▶ export
```

Denoising is a per-detector choice (on for PSD, off for BSCD and RBD by
default; ML and CNN follow their model); the three post-processing steps run
in the order shown, on every detector's output, before export.

| Step | Setting | Default |
|---|---|---|
| Denoise | each detector's `denoise` (algorithm: `Detection.PRE`) | PSD on, BSCD/RBD off |
| Filter Broadband | `minTonality` | `0` (off) |
| Merge Close Labels | `maxGapToMerge` | `0.005` s |
| Remove Short Labels | `minLabelLength` | `0.001` s |

Merging before removing short labels matters: two fragments of one call are
merged into a single valid event rather than both being discarded as too
short.
