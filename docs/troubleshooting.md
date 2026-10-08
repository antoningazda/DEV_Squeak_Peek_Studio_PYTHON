# Troubleshooting

## Installation

### macOS: “Squeak Peek Studio is damaged and can't be opened”

The app is not damaged. Current builds are not signed with an Apple
Developer ID, and macOS shows this for any such app.

**Right-click (or Control-click) the app → Open → Open.** Once per
installed version.

If no **Open** button appears:

```bash
xattr -dr com.apple.quarantine "/Applications/Squeak Peek Studio.app"
```

### Windows: “Windows protected your PC”

SmartScreen, for the same reason. Click **More info** → **Run anyway**.

### Linux: the app will not start

Missing Qt runtime libraries. On Debian/Ubuntu:

```bash
sudo apt-get install -y libegl1 libopengl0 libxkbcommon0 \
  libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
  libxcb-xinerama0 libdbus-1-3
```

Run the binary from a terminal — the error naming the missing library is
printed there.

---

## Loading data

### Loading a recording is slow

Expected for high-sample-rate audio: a two-minute 250 kHz file is ~70 MB
and is read fully into memory. Turn on **Visualization → Show loading
dialog** in [Settings](guide/settings.md#visualization) to get a progress
indicator.

### Label files are not found in batch mode

Matching is **by filename stem**. `recordings/animal_07.wav` pairs with
`labels/animal_07*.txt`. Each folder field shows how many matches it found
— if that count is zero or wrong, the stems do not line up.

### A label file fails to load

The error appears under **Load errors**; the recording still loads. Check
the file against the [two-line format](file-formats.md#label-files) — the
commonest cause is a missing continuation line (the one starting with a
backslash).

---

## Detection

### No detections at all

1. **Check the frequency band.** `fcutMin` / `fcutMax` default to
   40–120 kHz. Mouse USVs often sit higher; some rat calls sit lower. Look
   at the spectrogram with a wide display range and see where the energy
   actually is.
2. **Check the sample rate.** A recording sampled at 44.1 kHz cannot
   contain anything above 22 kHz — there is nothing in the USV band to
   find.
3. **Is tonality filtering removing everything?** Set **Min tonality** back
   to `0` and re-run.
4. **ML/CNN:** is `modelPath` set, and does the file exist?

### Far too many detections

In order of effect:

1. Turn on **Filter Broadband** with `minTonality = 0.5`
   ([why](methods.md#tonality-filtering)).
2. Raise `w` (PSD, BSCD) — the weight on the local SNR term.
3. Raise `minEffectivePower` (PSD) or `amplitudeThreshold` (RBD).
4. Raise **Min label length** in post-processing.

### One call detected as several fragments

**Merge Close Labels** is on by default; raise **Max gap to merge** above
the default `0.005` s. Check the result visually — too large a value starts
merging genuinely separate calls.

### Call onsets are clipped

This is PSD's characteristic failure: a call that rises gradually out of
the noise crosses an energy threshold late. Try **BSCD**, which keys on
statistical change rather than level, or **RBD** for the most precise
boundaries.

### Detection is very slow

RBD fits autoregressive models across a sliding window and is inherently
expensive. Lengthen `wlen` before lowering the AR orders. For a parameter
hunt, turn off `runWholeSignal` on PSD and work on a 10-second ROI.

The first run after launch is also slower than later ones: BSCD and RBD
JIT-compile their inner loops on first use.

---

## Sonification and audio

### No sound

Check your system output device. On Linux you also need a working
PortAudio/ALSA setup (`libportaudio2`).

### Sonification is all hiss

Turn **Sonification denoising** on
([Settings → Visualization](guide/settings.md#visualization)). An
ultrasonic recording is mostly broadband noise; shifted down, that noise is
louder than the calls.

### Sonification sounds smeared or “phasey”

Set **Sonification playback speed** back to **Match the pitch shift**. That
makes it a pure tape-speed transform with no resynthesis and no artefacts.
An independent slowdown factor costs a phase-vocoder pass.

### The calls are still too high to hear comfortably

Make **Sonification semitones** more negative. −36 is exactly three
octaves.

---

## Video

### “No broadband transient (snap) found”

The automatic sync could not find the marker. Either:

- annotate the sync point manually as an **`sk`** label in
  [Label Edit](guide/label-edit.md) and click **Re-sync** — this is the
  reliable fix; or
- lower the **snap threshold factor** in Settings → Video.

### “Could not extract an audio track”

The video has no audio stream. Automatic sync is computed from the camera's
audio, so it cannot work without one. Add an `sk` label to the ultrasonic
recording and sync from that side.

### Video and audio drift apart

A single snap corrects a constant offset, not a clock-rate difference. If
the two devices' clocks run at measurably different rates over a long
recording, no single offset will hold — split the recording, or use a
device that can be hardware-synced.

---

## Classification

### “Output exists”

Results folders must not exist yet — results are never overwritten. Pick a
new folder name.

### “Not a usable model”

The folder must contain `manifest.json`. Point the picker at the folder
*containing* it, typically `<output>/run/model`.

### Almost everything comes back UNCERTAIN

The confidence thresholds were calibrated on recordings unlike yours. Train
a model on your own data ([Training models](training.md)), or accept the
review queue as the cost of a model trained elsewhere.

### A call type I care about never gets predicted

It probably had fewer than `--min-examples` (default 10) training examples
and was folded into plain USV. The training results panel lists which types
this happened to. Collect more examples of it — lowering the threshold
instead gives you a type learned from four examples, which is noise with a
name.

---

## Training

### Test scores look implausibly good

Check your **groups**. If two recordings of the same animal landed in
different splits, the model has memorised that individual and the test
score is meaningless. Recordings sharing animals must share a `GroupID`.

### “Add labeled recordings first” / fewer than 5 groups

Group-wise 60/20/20 splitting needs at least five groups. With fewer
animals you cannot get an honest held-out estimate.

### The model learned my real calls as noise

You supplied a detected-labels file for a recording whose call-type labels
are **incomplete**. NOISE examples are detections matching no labelled
call, so every unlabelled real call became a NOISE example. Either complete
the labels or leave the detected-labels column empty for that recording.

### Training is very slow

Use `--device mps` on Apple Silicon. For the CNN detector, stay on the
`mobilenet` backbone unless you have a GPU. Run **Check data** first so you
do not discover a problem an hour in.

---

## Still stuck?

- [Open an issue](https://github.com/antoningazda/DEV_Squeak_Peek_Studio_PYTHON/issues)
  — include your OS, the app version (Info tab), and what you did.
- Email the author: [antonin.gazda@gmail.com](mailto:antonin.gazda@gmail.com)
