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
| [CNN](#cnn-faster-r-cnn) | yes | medium | Predicts frequency extent too | Needs PyTorch and real training data |

**Start with PSD.** If it misses soft calls or clips onsets in noisy
recordings, try BSCD. Reach for RBD when boundary accuracy is the point.
Train an ML or CNN detector when the classical detectors keep firing on a
noise source specific to your setup.

Whichever you pick, read [Tonality filtering](#tonality-filtering) — it is
the cheapest large accuracy gain available.

---

## PSD

**Power Spectral Density** — thresholds band power against an adaptive
noise floor. Fast and robust on clear, high-SNR recordings.

The pipeline:

1. DC removal and amplitude normalisation.
2. Zero-phase bandpass filter (order-12 IIR, `filtfilt`).
3. Hamming-window STFT.
4. Power envelope within the frequency band.
5. Noise floor estimated by a **moving minimum**.
6. **Adaptive threshold** from local SNR statistics.
7. Binary thresholding and edge detection to get events.
8. Rejection of events below a minimum effective power.

The adaptive noise floor is what makes it work on real recordings: a fixed
threshold fails the moment the ventilation changes.

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

The inner per-sample loop is JIT-compiled with numba, so despite being a
recursive sample-by-sample statistic it runs at usable speed on 250 kHz
recordings.

Because it keys on *change* rather than *level*, BSCD finds the start of a
call that rises gradually out of the noise — exactly the case where an
energy threshold clips the onset. The flip side is that any abrupt
transient is a change, which is why BSCD benefits most from
[tonality filtering](#tonality-filtering).

### Parameters

| Parameter | Default | What it does |
|---|---|---|
| `wlen` | `0.008` s | Sliding analysis window for detecting statistical change |
| `maWindow` | `7500` | Frames averaged to smooth the change statistic |
| `noiseWindow` | `800000` | Frames used to estimate background noise level |
| `localWindow` | `5000` | Frames used to estimate the local signal level |
| `k` | `0.01` | Scales the local-statistics term of the adaptive threshold |
| `w` | `6.0` | Weight on the local SNR term |

### Tuning

- `wlen` sets the timescale of change BSCD is sensitive to. Shorter finds
  sharper onsets and more noise; longer is steadier but blurs fast calls.
- `w` is the main precision/recall dial, as in PSD.
- The windows are in **frames, not seconds**, and the defaults are large
  because the statistic is computed per sample. Scale them if you change
  the sample rate substantially.

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
| `dynamicScaling` | `0.1` | Scales the adaptive threshold relative to local statistics |
| `smoothingWindowRBD` | `0.04` s | Smoothing of the detection statistic before thresholding |
| `smoothingWindowThr` | `0.01` s | Smoothing of the adaptive threshold itself |
| `amplitudeThreshold` | `0.001` | Minimum normalised amplitude for a candidate to be kept |

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

Needs PyTorch — `pip install -e ".[cnn]"` covers it in every version (see
[Install](install.md#neural-network-components)). The bundled desktop app
already includes it.

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

## Post-processing order

Every detector's output passes through the same chain:

```
Filter Broadband  →  Merge Close Labels  →  Remove Short Labels
```

| Step | Threshold | Default |
|---|---|---|
| Filter Broadband | `minTonality` | `0` (off) |
| Merge Close Labels | `maxGapToMerge` | `0.005` s |
| Remove Short Labels | `minLabelLength` | `0.001` s |

Merging before removing short labels matters: two fragments of one call are
merged into a single valid event rather than both being discarded as too
short.
