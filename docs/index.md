---
title: Detect and classify rodent ultrasonic vocalizations
hide:
  - navigation
---

<div class="spk-hero" markdown>

![Squeak Peek Studio](assets/logo.png){ .spk-logo }

# Squeak Peek Studio

Detect, classify, review and listen to ultrasonic vocalizations (USVs)
of laboratory rodents — a desktop application with a matching headless CLI.

[Download for macOS, Windows or Linux](install.md){ .md-button .md-button--primary }
[Quickstart](quickstart.md){ .md-button }

</div>

Rats and mice talk in the 40–120 kHz band, far above human hearing. Squeak
Peek Studio takes the raw high-sample-rate WAV recordings from your setup and
turns them into something you can actually work with: a scrollable
spectrogram, a list of detected calls, call types assigned by a trained
model, an interface for accepting or correcting each one, and audio you can
hear.

Everything the GUI does, the CLI does too, so a protocol you develop
interactively can be run unattended over a whole cohort.

## What it does

<div class="spk-grid" markdown>

<div class="spk-card" markdown>
### Detect
Five detection engines — PSD, BSCD, RBD, a Random Forest frame classifier
and a Faster R-CNN box detector. Run several at once and compare.
[Methods →](methods.md)
</div>

<div class="spk-card" markdown>
### Classify
A two-stage model (CNN for USV-vs-noise, Random Forest for call type) labels
each detected call and flags the ones it is unsure about for your review.
[Classification →](guide/classification.md)
</div>

<div class="spk-card" markdown>
### Review
Step through calls one key at a time, accept or reject each detection and
each call type, correct the type where the model got it wrong, and export.
[Label Edit →](guide/label-edit.md)
</div>

<div class="spk-card" markdown>
### Listen
Pitch-shifted, time-stretched sonification brings 70 kHz calls into your
headphones. Export a video with the spectrogram and sonified audio muxed in.
[Visualization →](guide/visualization.md)
</div>

<div class="spk-card" markdown>
### Measure
Compare any label set against your reference labels and get TP / FP / FN,
precision, recall and F1 — in the GUI or from the CLI.
[Metrics →](guide/metrics.md)
</div>

<div class="spk-card" markdown>
### Automate
`squeak-peek-cli` runs detection, evaluation, classification and model
training headlessly, over single files or whole folders.
[CLI reference →](cli.md)
</div>

</div>

## Typical workflow

1. **[Load](guide/data-input.md)** a recording (or a folder of them), plus
   any existing detected and reference label files.
2. **[Look and listen](guide/visualization.md)** — scroll the spectrogram,
   overlay labels, sonify a segment to check you are seeing real calls.
3. **[Detect](guide/detection.md)** — pick one or more detectors, add
   post-processing, run. Label files are written to your export folder.
4. **[Classify](guide/classification.md)** — assign call types with a trained
   model, and get a review queue of uncertain calls.
5. **[Review](guide/label-edit.md)** — accept, reject or correct each call
   with the keyboard; export the curated labels.
6. **[Evaluate](guide/metrics.md)** — score the result against reference
   labels.

## Requirements

| | |
|---|---|
| Operating system | macOS 12+, Windows 10/11, or a recent 64-bit Linux |
| Installation | Standalone installer — no Python needed |
| Audio input | WAV files; designed around 250 kHz sampling, works with any rate that covers your band |
| Optional | A GPU is never required; the CNN components run on CPU (and Apple Silicon MPS) |

## Credits and citation

Squeak Peek Studio is a Python reimplementation and extension of the MATLAB
application developed by **Ing. Antonín Gazda** as a Master's thesis at the
**Czech Technical University in Prague, Faculty of Electrical Engineering
(FEL)** (May 2025), in collaboration with the **National Institute of Mental
Health (NUDZ)**.

- Thesis: [CTU DSpace record](https://dspace.cvut.cz/entities/publication/dede5152-081b-41cf-a4ba-55dacad884d5)
- Source code: [github.com/antoningazda/DEV_Squeak_Peek_Studio_PYTHON](https://github.com/antoningazda/DEV_Squeak_Peek_Studio_PYTHON)
- Licence: MIT

If you use Squeak Peek Studio in published work, please cite the thesis and
link the repository.
