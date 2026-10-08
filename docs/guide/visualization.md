# Visualization

A scrollable spectrogram of the loaded recording, with label overlays and
sonification. This is where you check that your data is what you think it
is, before and after detection.

<figure markdown>
  ![The Visualization tab: waveform, spectrogram, label overlays and pitch traces](../assets/screenshots/visualization.png#only-light){ .spk-shot }
  ![The Visualization tab: waveform, spectrogram, label overlays and pitch traces](../assets/screenshots/visualization-dark.png#only-dark){ .spk-shot }
  <figcaption>One second of the example recording: six calls, detected labels in
  cyan, reference labels in white, and the orange pitch trace over each call.</figcaption>
</figure>

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

### Panning and zooming with the mouse

You can also drive the view directly on the plot:

| Gesture | Effect |
|---|---|
| **Drag** left/right | Pan along time |
| **Scroll wheel** / trackpad pinch | Zoom in and out around the pointer |

Only the **time axis** moves — the frequency range stays exactly where your
settings put it, so a call never drifts out of the band you are inspecting,
and the waveform above stays aligned with the spectrogram below. Panning is
clamped to the recording, and the view is capped at 60 s, the same limit as
**Length (s)**.

The spectrogram is **re-rendered** for whatever range you land on, rather
than being stretched: zoom in and you get more detail, not bigger pixels. A
margin either side of the visible range is rendered too, so a drag reveals
real data rather than empty space. **Start (s)** and **Length (s)** follow
the new range, so the keyboard and the mouse never disagree about where you
are.

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

### The pitch trace

The **pitch trace** follows the dominant frequency of whatever is actually
sounding, independent of any label — an undetected call still gets a trace,
which makes the overlay a quick way to spot calls a detector missed.

How a frame qualifies, and how the contour is drawn:

- Each spectrogram bin is referenced to its own median over the rendered
  range, which removes stationary noise lines, and each frame to its own
  median across frequency. A frame counts as a call frame only when its
  strongest bin stands **12 dB** above that background, and runs shorter
  than three frames are dropped.
- Within each run the contour is a **Viterbi path** that trades bin power
  against frequency jumps (penalised per kHz, hard-capped at 6 kHz per
  frame). Rat calls often show several parallel bands of similar power, and
  a plain per-frame peak pick hops between them, drawing vertical zigzags;
  the path cost keeps the trace on one band.
- Bins above **100 kHz** are ignored, so stray high-frequency noise cannot
  drag the trace out of the USV range.

A clean, continuous trace through a call means the spectrogram settings are
resolving it properly; a broken or wandering trace usually means the window
is too long or too short for that call type.

!!! note "It is not the classifier's contour"

    The [call-type classifier](classification.md) extracts its own contour
    with its own tracker and settings. The trace drawn here is a display
    aid — reading it tells you whether the call is well resolved on screen,
    not what the model saw.

## Adding a label by hand

**Right-click** anywhere on the spectrogram (or the waveform) to create a
label centred on that point. It is added to the detected-label set straight
away, so it shows up in [Label Edit](label-edit.md) and is exported with
everything else.

Two settings shape it
([Settings → Visualization](settings.md#visualization)):

| Setting | Default | Effect |
|---|---|---|
| **Manual label length** | `0.075` s | Duration of the new label, centred on the click |
| **Manual label text** | `md` | The call type written into it |

`md` is short for "manual" and is deliberately a type no detector or
classifier produces, so a text search of the exported file finds exactly
the labels you added by hand. Change it if your taxonomy wants something
else.

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
