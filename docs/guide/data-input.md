# Data Input

Everything starts here. Until you click **Load files**, the Visualization,
Detection and Label Edit tabs have nothing to show.

## Single file mode

The default. You pick up to three files:

| Field | Required | What it is |
|---|---|---|
| **WAV file** | Yes | The ultrasonic recording to analyse |
| **Detected labels** | No | A previously detected label file to load and continue editing in Label Edit |
| **Reference labels** | No | Ground-truth labels, used to score detections in [Metrics](metrics.md) |

Click **Load files**. The WAV is read into memory and the status line shows

```
Sample rate: 250,000 Hz · Duration: 144.106 s · Detected labels: 312 · Reference labels: 298
```

If a label file fails to parse, the error is reported under **Load errors**
and the recording still loads — a malformed reference file never blocks you
from working.

!!! note "Loading is not instant"

    High-sample-rate recordings are large: a two-minute file at 250 kHz is
    around 70 MB. Loading reads the whole thing into memory. Turn on
    **Visualization → Show loading dialog** in [Settings](settings.md) if
    you want a progress indicator for long files.

## Batch folder mode

Select **Batch folder** and you pick folders instead of files:

| Field | What it is |
|---|---|
| **WAV folder** | Folder of `.wav` recordings, processed one after another |
| **Detected labels folder** | Previously detected label files, matched to recordings **by filename** |
| **Reference labels folder** | Ground-truth label files, matched the same way |

Each field shows how many matching files were found, so you can see at a
glance whether the filename matching worked before you run anything.

**Filename matching** pairs `recordings/animal_07.wav` with
`labels/animal_07*.txt`. Keep the stem identical and the match is automatic.

What batch mode changes:

- **[Detection](detection.md)** runs every selected detector over every WAV
  in the folder and writes one label file per recording per detector.
- **[Classification](classification.md)** can pull its whole recording list
  from these folders in one click.
- **Visualization** and **Label Edit** still work on a single loaded
  recording — batch mode is about unattended processing, not about reviewing
  many files at once.

## Defaults

The file and folder paths that appear when the app starts come from the
settings file (`DataInput.*`). Set them once in
[Settings](settings.md#data-input) for the layout of your own data and you
will rarely touch this tab again.

---

**Next:** [Visualization →](visualization.md)
