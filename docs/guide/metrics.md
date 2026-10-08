# Metrics

Score a detection run against ground truth. Requires **reference labels**
loaded in [Data Input](data-input.md).

<figure markdown>
  ![The Metrics tab](../assets/screenshots/metrics.png#only-light){ .spk-shot }
  ![The Metrics tab](../assets/screenshots/metrics-dark.png#only-dark){ .spk-shot }
  <figcaption>The example recording's bundled detections scored against its reference labels.</figcaption>
</figure>

## Label counts

Always shown, no reference needed: how many detected and reference labels
are currently loaded. A quick sanity check that you are scoring what you
think you are scoring.

## Detection metrics

Scored automatically whenever you open the tab, against whatever is loaded
at that moment — so the numbers always reflect the current detected and
reference labels, including edits made in [Label Edit](label-edit.md). While
the tab is open they update live. **Compute metrics** forces a rescore.

| Metric | Definition |
|---|---|
| **True positives (TP)** | Detected labels that correctly match a reference label |
| **False positives (FP)** | Detected labels with no matching reference label — likely false alarms |
| **False negatives (FN)** | Reference labels with no matching detection — missed calls |
| **Precision** | `TP / (TP + FP)` — of the detections made, the fraction that were correct |
| **Recall** | `TP / (TP + FN)` — of the real calls, the fraction that were found |
| **F1** | Harmonic mean of precision and recall — a single balanced score |

## How matching works

**Midpoint matching, one-to-one.**

A detected label matches a reference label when the **midpoint of the
reference label falls inside the detected label's time range**. Each
detected label can match at most one reference label.

Two consequences worth understanding before you read the numbers:

- **A detection that is too long still counts as a hit.** If a detector
  smears one call across 200 ms and the real call's midpoint is inside
  that span, it is a TP. Precision does not punish sloppy boundaries, only
  spurious events. If boundary accuracy matters for your analysis, inspect
  it visually in [Visualization](visualization.md) — these numbers will not
  show it.
- **Splitting one call in two costs you.** The first fragment containing
  the reference midpoint is the TP; the second fragment matches nothing and
  becomes an FP. This is why
  [Merge Close Labels](detection.md#post-processing) is on by default.

## Reading the result

Precision and recall move in opposite directions as you tune a detector,
and which one you want depends on what you are doing next:

- **Building a training set or counting calls?** Favour precision — a false
  positive taught to a model, or counted in a total, is worse than a missed
  marginal call. Turn on
  [Filter Broadband](../methods.md#tonality-filtering).
- **Reviewing everything by hand in [Label Edit](label-edit.md) anyway?**
  Favour recall — you will delete the false positives in seconds, but you
  can never review a call the detector never found.

F1 is the summary when you have no strong preference.

!!! tip "Compare detectors in one pass"

    Select several detectors in the [Detection tab](detection.md); each
    exports its own file. Load each result in turn as the detected labels
    and compute metrics to get a like-for-like comparison on your own
    recordings — which is far more informative than any published benchmark.

## From the command line

```bash
squeak-peek-cli evaluate detected.txt reference.txt
```

Same matching logic, same numbers, scriptable across a whole cohort.

→ [CLI reference](../cli.md#evaluate)

---

**Next:** [Settings →](settings.md)
