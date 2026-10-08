# Label Edit

Curate the detections, one call at a time. This is where a machine-produced
label file becomes a dataset you would put in a paper.

The tab shows a single detected call, zoomed in, with its own spectrogram
and a compact row of controls. It is designed to be driven almost entirely
from the keyboard.

<figure markdown>
  ![The Label Edit tab reviewing one call](../assets/screenshots/label-edit.png#only-light){ .spk-shot }
  ![The Label Edit tab reviewing one call](../assets/screenshots/label-edit-dark.png#only-dark){ .spk-shot }
  <figcaption>One call of 284, its boundaries drawn as draggable lines, with the
  running accept/reject tallies on the right.</figcaption>
</figure>

## The two independent questions

Every call carries **two** separate verdicts, because they are different
questions:

| | Question | Keys |
|---|---|---|
| **Detection State** | *Was a call correctly detected here?* | <kbd>D</kbd> accept · <kbd>Shift</kbd>+<kbd>D</kbd> reject |
| **Classification State** | *Is the assigned call type correct?* | <kbd>C</kbd> accept · <kbd>Shift</kbd>+<kbd>C</kbd> reject |

A detection can be perfectly real while its call type is wrong, and a
call type is meaningless if the detection is a false positive. Keeping them
apart means your exported file records what you actually decided — and it is
what lets [training](classification.md#train-model) use rejected detections
as NOISE examples without throwing away the correctly detected calls whose
*type* you merely disagreed with.

## Working through a file

| Key | Action |
|---|---|
| <kbd>→</kbd> | Next call |
| <kbd>←</kbd> | Previous call |
| <kbd>D</kbd> / <kbd>Shift</kbd>+<kbd>D</kbd> | Accept / reject the detection |
| <kbd>C</kbd> / <kbd>Shift</kbd>+<kbd>C</kbd> | Accept / reject the classification |
| <kbd>Space</kbd> | **Accept both and advance** |

In practice most of a review pass is <kbd>Space</kbd>, held down, stopping
whenever something looks wrong. All bindings are rebindable —
see [Keyboard shortcuts](shortcuts.md).

The header shows the current call's position, its duration in milliseconds
and its current label, e.g.

```
dur = 34.2 ms    label = '5t'
```

and running totals update live as you work:

```
Detections:      284 accepted / 28 rejected
Classifications: 261 accepted / 51 rejected
```

## Correcting a call type

- **Class** — a dropdown of your configured call types. The list comes from
  `LabelEdit.Classifications` in
  [Settings](settings.md#label-edit) (default:
  `d, sk, 5, 5t, 5w, c5`). Edit it there to match your own taxonomy.
- **Corrected class (rejected)** — a free-text field. After rejecting a
  classification, type the correct label here. Use this for a type that is
  not in your dropdown yet.

## Fixing boundaries

Drag a label's edges directly on the spectrogram to correct a start or end
time. This is the common fix when a detector clips a call's onset or runs
past its offset.

## Adding a missed call

**Right-click on the spectrogram** to create a new manual label at that
position — create it, then drag the edges to fit. Two
[Settings → Visualization](settings.md#visualization) values shape it:

| Setting | Default | Effect |
|---|---|---|
| `Visualization.ManualLabelLength` | `0.075` s | Length of the new label, centred on the click |
| `Visualization.ManualLabelMarker` | `md` | Call type written into it |

`md` is a type no detector or classifier produces, so searching an exported
file for it finds exactly the calls you added by hand. The same right-click
works on the [Visualization](visualization.md#adding-a-label-by-hand) tab.

## Undoing a mistake

Every edit to the labels is undoable from the **Edit** menu:

| Action | Default key |
|---|---|
| Undo | <kbd>Ctrl</kbd>+<kbd>Z</kbd> |
| Redo | <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> |

It covers accepting and rejecting, changing a call type, typing a
correction, dragging a boundary and creating a label by right-click — in
Label Edit and in Visualization alike. A typed correction is one undo step
for the whole field, not one per character.

The history holds the last **50** edits, and **loading a recording clears
it** — undo cannot reach back past a file change.

!!! warning "Undo is not a substitute for exporting"

    The history lives in memory, and running a detector again replaces the
    whole detected-label set. Export when you have finished a pass.

## Spectrogram settings

Label Edit keeps its **own** window, overlap, frequency range and colormap,
separate from the Visualization tab
([Settings → Label Edit](settings.md#label-edit)). The default colormap here
is `invgray` — dark calls on a light background, which many people find
easier to stare at for an hour of review.

Defaults are 1024/512 at 40–120 kHz, the same time/frequency trade-off as
Visualization.

## Export

**Export labels…** writes everything — the accept/reject decisions, any
boundary edits and any corrected call types — to a text file.

See [File formats](../file-formats.md#label-files) for what ends up in it.

!!! tip "Export before you move on"

    Review decisions live in memory until you export. Running a detector
    again replaces the current detected-label set.

---

**Next:** [Video →](video.md) · [Metrics →](metrics.md)
