# Quickstart

Your first detection run, start to finish, in about five minutes. The app
ships with a short example recording, so you can do all of this before
touching your own data.

!!! tip "Follow along with the example recording"

    A source checkout contains `data/example/single/USV_Example_Short.wav`
    together with matching detected and reference label files. The
    **Settings** tab points at these by default, so a fresh install already
    has somewhere to start.

## 1. Load a recording

Open the **Data Input** tab.

1. Leave the mode on **Single file**.
2. Set **WAV file** to your recording.
3. Optionally set **Detected labels** (an existing label file you want to
   review) and **Reference labels** (ground truth, used for
   [Metrics](guide/metrics.md)).
4. Click **Load files**.

The title bar shows the loaded recording, and the other tabs come alive.

→ [Data Input in detail](guide/data-input.md)

## 2. Check what you are looking at

Switch to **Visualization**. You get a spectrogram of a one-second window,
by default showing 40–120 kHz — the rodent USV band.

- Move through the recording with **Prev** / **Next**, or the
  <kbd>←</kbd> / <kbd>→</kbd> keys.
- Change **Start (s)** and **Length (s)** to jump somewhere specific or
  widen the window.
- Tick **Detected labels** and **Reference labels** to overlay the label
  files you loaded.
- Click **🔊 Sonify** to hear the current segment, pitch-shifted and slowed
  down into the audible range.

If the spectrogram looks empty, your calls may sit outside the default
frequency range — adjust it in [Settings](guide/settings.md#visualization).

→ [Visualization in detail](guide/visualization.md)

## 3. Run a detector

Go to the **Detection** tab.

1. Under **Detectors**, tick **PSD**. (You can tick several and they all run,
   each producing its own label file.)
2. Under **Post-processing**, **Merge Close Labels** and **Remove Short
   Labels** are on by default. Leave them.
3. Set the **Export folder**, or leave it empty to write next to the WAV.
4. Click **Run detectors**.

Each detector writes a file named

```
<recording>_<DETECTOR>_<timestamp>_detected.txt
```

and the result is loaded as the current detected labels, so it appears
immediately in Visualization and Label Edit.

→ [Detection in detail](guide/detection.md) ·
[How the detectors work](methods.md)

!!! question "Too many false positives?"

    Rodent USVs are narrowband whistles; most false positives (cage knocks,
    bedding rustle, scratching) are broadband. Tick **Filter Broadband** in
    post-processing and set **Min tonality** to `0.5` in Settings. On
    benchmark recordings this raised PSD's F1 from 0.56 to 0.78 and BSCD's
    from 0.71 to 0.83. See [Tonality filtering](methods.md#tonality-filtering).

## 4. Review the calls

Open **Label Edit**. It shows one detected call at a time, zoomed in, with
its spectrogram.

| Key | Action |
|---|---|
| <kbd>→</kbd> / <kbd>←</kbd> | Next / previous call |
| <kbd>D</kbd> | Accept this detection |
| <kbd>Shift</kbd>+<kbd>D</kbd> | Reject this detection |
| <kbd>C</kbd> | Accept the call-type classification |
| <kbd>Shift</kbd>+<kbd>C</kbd> | Reject the classification |
| <kbd>Space</kbd> | Accept both and move to the next call |

Most of a review pass is just holding <kbd>Space</kbd> and stopping when
something looks wrong. When you are done, click **Export labels…**.

→ [Label Edit in detail](guide/label-edit.md)

## 5. Score it

If you loaded reference labels, open **Metrics** and click **Compute
metrics** for true positives, false positives, false negatives, precision,
recall and F1.

→ [Metrics in detail](guide/metrics.md)

---

## The same thing from the command line

```bash
# detect
squeak-peek-cli detect recording.wav --detector psd --min-tonality 0.5

# score against reference labels
squeak-peek-cli evaluate recording_PSD_..._detected.txt reference.txt

# the whole folder at once
squeak-peek-cli batch recordings/ --detector bscd --output-dir out/
```

→ [Full CLI reference](cli.md)

## Where to go next

<div class="spk-grid" markdown>

<div class="spk-card" markdown>
### Pick the right detector
PSD, BSCD, RBD, ML and CNN each have different strengths and failure modes.
[Methods →](methods.md)
</div>

<div class="spk-card" markdown>
### Assign call types
Run a trained classifier over your detections and review only the calls the
model is unsure about. [Classification →](guide/classification.md)
</div>

<div class="spk-card" markdown>
### Train on your own colony
Recordings from your setup will beat any general-purpose model.
[Training models →](training.md)
</div>

<div class="spk-card" markdown>
### Process a whole cohort
Batch mode in the GUI, or `squeak-peek-cli batch` in a script.
[Data Input →](guide/data-input.md)
</div>

</div>
