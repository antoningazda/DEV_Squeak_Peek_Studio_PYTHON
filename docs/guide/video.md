# Video

Align a behaviour video with the ultrasonic recording, play them together,
and export a video whose soundtrack is the sonified ultrasound.

This answers the question a spectrogram cannot: *what was the animal doing
when it made that call?*

## The setup this assumes

A behaviour camera with its own ordinary microphone and a separate
ultrasonic microphone, started independently — so their clocks do not
agree. To tie them together you need a shared marker:

**Snap your fingers near the microphones at the start of the recording.**

A finger snap is broadband: it is audible as a sharp click in the camera's
audio track *and* visible as a transient at the top of the ultrasonic band.
That single event is enough to align the two timelines.

## Import and sync

Load a WAV in [Data Input](data-input.md) first — the video is aligned
*to* it.

Click **Import video…** and pick an `.mp4`, `.mov`, `.avi`, `.mkv` or
`.m4v`. The app then:

1. Extracts the video's audio track with the bundled ffmpeg (**the video
   must have an audio stream** — this is what the sync is computed from).
2. Finds the sync click on the ultrasonic side.
3. Finds the first sharp broadband transient in the video's audio.
4. Reports the offset:

```
Sync offset: +1.284 s  ('sk' reference label + auto-detected snap in video)
```

The offset is the number of seconds to add to a WAV time to get the
corresponding video time.

### How the ultrasonic snap is found

Two methods, tried in this order:

1. **An `sk` label.** If any loaded label file (reference first, then
   detected) contains a label named `sk`, its start time is used. This is
   the reliable option: annotate the snap once and the sync is exact.
2. **Automatic transient detection.** Failing that, the app bandpasses the
   recording to **100–120 kHz** and takes the first point where the energy
   envelope rises well above the noise floor. The band is deliberately at
   the *top* of the ultrasonic range, above where USV calls concentrate, so
   an ordinary call cannot be mistaken for the snap.

**Re-sync** re-runs detection for the current video — useful after you add
an `sk` label, or after changing the thresholds.

The video-side band and the threshold factor are configurable in
Settings → Video.

!!! failure "“No broadband transient (snap) found”"

    Either the snap was too soft or too soft-edged to stand out, or there
    isn't one. Annotate the sync point manually as an `sk` label in
    [Label Edit](label-edit.md) and hit **Re-sync**.

## Playback

**▶ Play** plays the video with the position readout
(`00:14 / 02:24`) and a scrub slider. The offset is applied, so the video
position and the audio timeline stay in agreement.

## Export

**Export video with spectrogram + sonified audio…** writes an MP4 that
combines:

- the original video frames,
- a spectrogram panel composited alongside them,
- a soundtrack of the **sonified** ultrasound, aligned by the sync offset.

Two things worth knowing:

- The **video stream is copied without re-encoding** — only the audio track
  is replaced — so export is fast and loses no image quality.
- The output is trimmed to the shorter of the two streams.

The soundtrack uses the full-track sonification, which preserves real-world
duration so the audio stays locked to the video's clock. (The segment
sonification in [Visualization](visualization.md) deliberately stretches
time instead, which is why it cannot be used here.)

This is the format to use for talks and supplementary material: the viewer
sees the behaviour, sees the call on the spectrogram, and hears it.

---

**Next:** [Metrics →](metrics.md)
