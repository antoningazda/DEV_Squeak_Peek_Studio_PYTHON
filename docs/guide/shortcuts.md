# Keyboard shortcuts

Reviewing a recording means making the same few decisions a few hundred
times. Every one of them has a key.

## Defaults

| Action | Default | Where it works |
|---|---|---|
| Open WAV… | <kbd>Ctrl</kbd>+<kbd>O</kbd> | File menu |
| Quit | <kbd>Ctrl</kbd>+<kbd>Q</kbd> | File menu |
| Previous segment / label | <kbd>←</kbd> | Visualization / Label Edit |
| Next segment / label | <kbd>→</kbd> | Visualization / Label Edit |
| Accept detection | <kbd>D</kbd> | Label Edit |
| Reject detection | <kbd>Shift</kbd>+<kbd>D</kbd> | Label Edit |
| Accept classification | <kbd>C</kbd> | Label Edit |
| Reject classification | <kbd>Shift</kbd>+<kbd>C</kbd> | Label Edit |
| Accept both & advance | <kbd>Space</kbd> | Label Edit |
| Undo | <kbd>Ctrl</kbd>+<kbd>Z</kbd> | Edit menu — label edits in Label Edit / Visualization |
| Redo | <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> | Edit menu — label edits in Label Edit / Visualization |

On macOS, <kbd>Ctrl</kbd> means <kbd>⌘</kbd>.

<kbd>←</kbd> and <kbd>→</kbd> do the contextually sensible thing: in
[Visualization](visualization.md) they step by one segment, in
[Label Edit](label-edit.md) they step by one call.

## The review loop

Everything is reachable from the home row and the space bar:

```
Space  →  accept this call, show me the next one
D      →  the detection is real (but I'm not sure about the type)
Shift+D → this isn't a call
C      →  the call type is right
Shift+C → the call type is wrong (then type the correction)
←  →   →  move without deciding anything
```

Hold <kbd>Space</kbd> through the easy calls; stop when something looks
wrong.

## Customising

<figure markdown>
  ![Settings, Shortcuts section](../assets/screenshots/settings-shortcuts.png#only-light){ .spk-shot }
  ![Settings, Shortcuts section](../assets/screenshots/settings-shortcuts-dark.png#only-dark){ .spk-shot }
  <figcaption>Every binding, where it applies, and a reset button per row.</figcaption>
</figure>

**Settings → Shortcuts.** Click a shortcut field and press the new key
combination. It saves immediately and takes effect app-wide — no Apply, no
restart.

- A combination already bound to another action is **flagged**, and only one
  of the two will actually fire. Resolve it before relying on either.
- Each row has a reset button restoring that action's shipped default.
- **Reset all shortcuts to defaults** restores every binding.

Bindings are stored by the operating system (Qt's `QSettings`), not in the
[settings JSON](settings.md). They survive upgrades and are independent of
whichever settings file you have loaded — so a settings file you share with
a collaborator will not overwrite their key bindings.
