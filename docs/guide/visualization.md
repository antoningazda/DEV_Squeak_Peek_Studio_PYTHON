# Visualization

A scrollable spectrogram of the loaded recording, with label overlays and
sonification. This is where you check that your data is what you think it
is, before and after detection.

## Moving around

The view shows one window of the recording at a time, defined by two fields:

| Control | Default | Meaning |
|---|---|---|
| **Start (s)** | `0.0` | Where the window begins, in seconds from the start of the recording |
| **Length (s)** | `1.0` | How much of the recording the window covers |

- **◀ Prev** / **Next ▶** step backwards and forwards by exactly one window
  length — so with the default settings, one second at a time.
- <kbd>←</kbd> and <kbd>→</kbd> do the same from the keyboard.
- Type a number into **Start (s)** to jump straight to a timestamp.

A one-second window is a good default for inspecting individual calls.
Widen it to 5–10 s to get a sense of a bout; much beyond that and
individual calls stop being visible at screen resolution.

## Overlays

Three checkboxes:

| Overlay | Shows |
|---|---|
| **Detected labels** | Boxes for the current detected-label set — whatever the last detector run or Label Edit session produced |
| **Reference labels** | Boxes for the ground-truth labels loaded in Data Input |
| **Pitch trace** | The estimated fundamental frequency of each call, drawn over the spectrogram |

Detected and reference labels are drawn in different colours (cyan and
white by default) so you can see at a glance where they agree and where they
do not — overlapping boxes are matches, a lone white box is a missed call,
a lone cyan box is a likely false positive.

!!! tip "Visual evaluation before numerical evaluation"

    Turning on both overlays and scrolling through a minute of recording
    tells you *how* a detector is failing — splitting calls in two, merging
    neighbours, firing on cage noise — in a way the
    [Metrics](metrics.md) numbers cannot.

The **pitch trace** is the same frequency contour the feature extractor uses
for classification. A clean, continuous trace through a call means the
spectrogram settings are resolving it properly; a broken or wandering trace
usually means the window is too long or too short for that call type.

## Sonification

Click **🔊 Sonify** to hear the current segment.

Rodent USVs sit between 40 and 120 kHz — roughly four octaves above the top
of human hearing. Sonification brings them down:

1. The ultrasonic band is kept and everything outside it discarded.
2. The samples are replayed at a lower rate. This is an exact pitch shift
   with no artefacts, and it stretches time by the same ratio — a 30 ms
   squeak becomes an audible 240 ms whistle. This is how a time-expansion
   bat detector works.
3. A soft spectral gate suppresses the broadband hiss that would otherwise
   be the loudest thing in the result.
4. Fades, a subsonic high-pass and a soft ceiling keep the output from
   clicking or clipping.

Two settings control it
([Settings → Visualization](settings.md#visualization)):

| Setting | Default | Effect |
|---|---|---|
| **Sonification ST** | `-35` | Pitch shift in semitones. More negative = lower and slower |
| **Sonification slowdown** | `4` | Time stretch factor |

When the slowdown matches the pitch ratio (the default case) the result is
produced by pure resampling and is mathematically pristine. Asking for a
slowdown that differs from the pitch ratio runs a phase-locked vocoder over
the difference only — still good, but no longer artefact-free. If
sonification ever sounds smeared or "phasey", bring the two settings back
into agreement.

## Spectrogram appearance

All of these live in [Settings → Visualization](settings.md#visualization)
and apply to this tab:

| Setting | Default | Notes |
|---|---|---|
| **Window** | `1024` | FFT window length in samples. Longer = finer frequency detail, coarser timing |
| **Overlap** | `512` | Samples of overlap between windows. Higher = smoother image, slower |
| **Min / Max frequency** | `40` / `120` kHz | The displayed band |
| **Colormap** | `invgray` | 15 options: parula, turbo, hsv, hot, cool, spring, summer, autumn, winter, gray, bone, copper, pink, jet, invgray |

!!! warning "Window length is a real trade-off"

    1024/512 at 250 kHz gives about 4 ms time resolution and 244 Hz
    frequency resolution — a reasonable balance for rat USVs. Raising the
    window to 4096 resolves frequency four times better but blurs short
    calls badly. If calls look like smeared blocks rather than sweeps, the
    window is too long.

[Label Edit](label-edit.md) keeps its *own* copy of these four settings, so
you can use a fine, zoomed-in rendering for review and a faster one for
scrolling.

---

**Next:** [Detection →](detection.md)
