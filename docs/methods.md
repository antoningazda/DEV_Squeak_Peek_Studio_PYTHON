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

**Start with PSD.** If it misses soft calls or clips onsets in noisy
recordings, try BSCD. Reach for RBD when boundary accuracy is the point.
Train an ML or CNN detector when the classical detectors keep firing on a
noise source specific to your setup.

Whichever you pick, read [Tonality filtering](#tonality-filtering) — it is
the cheapest large accuracy gain available. If the recordings themselves
are noisy rather than the detections, see
[Pre-detection denoising](#pre-detection-denoising), which cleans the audio
the classical detectors see.

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
| `noiseWindow` | `120` | Moving-minimum window estimating the noise floor (frames) |
| `localWindow` | `10` | Window for local mean/std in the adaptive threshold (frames) |
| `k` | `0.023` | Scales the local-statistics term of the threshold |
| `w` | `3.0` | Weight on the local SNR term |
| `minEffectivePower` | `3e-05` | Minimum mean power to accept a candidate |
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
| `noiseWindow` | `800000` | Moving-minimum window for the noise floor — **`adaptive` only** |
| `localWindow` | `5000` | Window for local mean/std — **`adaptive` only** |
| `k` | `15.0` | Scales the local-statistics term — **`adaptive` only** |
| `w` | `300.0` | Weight on the local SNR term — **`adaptive` only** |

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
candidate boundary, fits an autoregressive model to the signal on the left
and another on the right, and compares that piecewise model against a
single-AR baseline using a Bayesian evidence ratio. A boundary where two
separate models explain the data much better than one is a real change
point.

This is the most principled of the three classical detectors and gives the
most precise boundaries. It is also the most expensive — it fits AR models
across a sliding window — and has the most parameters.

### Parameters

| Parameter | Default | What it does |
|---|---|---|
| `wlen` | `0.02` s | Window compared on either side of each candidate boundary |
| `AR_order_left` | `4` | AR model order left of the boundary |
| `AR_order_right` | `4` | AR model order right of the boundary |
| `Bayesian_Evidence_order` | `4` | AR order for the baseline evidence |
| `dynamicScaling` | `0.00015` | Scales the adaptive threshold relative to local statistics |
| `smoothingWindowRBD` | `0.04` s | Smoothing of the detection statistic before thresholding |
| `smoothingWindowThr` | `0.01` s | Smoothing of the adaptive threshold itself |
| `amplitudeThreshold` | `0.01` | Minimum normalised amplitude for a candidate to be kept |

### Tuning

- **AR orders** are the model-complexity dial: higher orders capture more
  complex spectral shape but need more data per window and more
  computation. 4 is a sensible default for USV-band audio; going much above
  8 rarely helps.
- **`dynamicScaling`** is the main sensitivity control.
- The two smoothing windows interact: smoothing the statistic more than the
  threshold makes detection conservative, and vice versa. Change one at a
  time.
- If RBD is too slow, lengthen `wlen` before lowering the AR orders.

Both smoothing windows use MATLAB's `movmean` convention, shrinking at the
edges instead of padding, so the start and end of a recording are not
systematically pushed below threshold.

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

## Pre-detection denoising

An optional stage **before** any classical detector runs, off by default.
Turn it on with **Detection → Pre-processing → Denoise**, or
`--denoise` on the [CLI](cli.md#detect).

### The idea

An ultrasonic recording is mostly background: fans, electronics, the
microphone's own hiss. That background is roughly constant over a session,
while the calls are sparse and brief — so each frequency bin's noise level
can be estimated **from the recording itself**, as a low percentile of that
bin's magnitude over time, and subtracted. It is spectral subtraction, the
same idea as Audacity's Noise Reduction, with the noise profile taken
automatically instead of from a selection you make by hand.

PSD and BSCD otherwise see that constant in-band noise as a large part of
their envelope, which costs precision on quiet recordings.

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

### Parameters

[Settings → Pre-processing](guide/settings.md#pre-processing), stored under
`Detection.PRE`:

| Parameter | Default | What it does |
|---|---|---|
| `enabled` | `false` | The **Denoise** checkbox |
| `nfft` | `1024` samples | STFT window for estimation and suppression |
| `noisePercentile` | `20.0` | Percentile of each bin's magnitude taken as its noise level |
| `oversubtraction` | `1.5` | Multiple of that level subtracted |
| `maxReductionDb` | `18.0` dB | Largest attenuation any bin may receive |

### Scope

- **Classical detectors only.** PSD, BSCD and RBD run on the denoised
  audio. **ML** and **CNN** instead apply the denoising recorded in their
  own model file at training time, so training and inference always see the
  same kind of audio — which is also why their results do not change when
  you tick the box.
- **Detection only.** Export, [Label Edit](guide/label-edit.md),
  sonification, the video soundtrack and
  [call-type classification](guide/classification.md) all use the original
  recording. Nothing you listen to, look at or ship is denoised.

!!! warning "Retune after you enable it"

    A detector's thresholds describe a particular noise level, and
    denoising changes that level. Parameters fitted on raw audio are not
    the right parameters for denoised audio — on paper it only removes
    noise, but in practice the envelope it feeds the detector is both
    quieter and flatter.

    Either re-run **Detection → Train detector → Tune detector
    parameters** with the checkbox in the state you intend to use (tuning
    denoises first, so it fits what you will actually run), or score the
    change against reference labels in [Metrics](guide/metrics.md) before
    applying it to a dataset.

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

Denoising is off by default and applies to the classical detectors only;
the three post-processing steps run in the order shown, on every detector's
output, before export.

| Step | Setting | Default |
|---|---|---|
| Denoise | `Detection.PRE.enabled` | `false` (off) |
| Filter Broadband | `minTonality` | `0` (off) |
| Merge Close Labels | `maxGapToMerge` | `0.005` s |
| Remove Short Labels | `minLabelLength` | `0.001` s |

Merging before removing short labels matters: two fragments of one call are
merged into a single valid event rather than both being discarded as too
short.
