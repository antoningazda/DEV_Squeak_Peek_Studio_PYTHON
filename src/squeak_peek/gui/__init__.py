"""
PyQt6 desktop application.

    app.py                 — QApplication entry point + MainWindow (7-tab layout)
    _tab_data_input.py     — WAV / detected / reference label loading
    _tab_visualization.py  — Spectrogram + waveform viewer
    _tab_detection.py      — Runs PSD/BSCD/RBD detectors (ML pending Tier 2)
    _tab_label_edit.py     — Manual label editing
    _tab_metrics.py        — Detection accuracy metrics (labels.metrics.compare_labels)
    _tab_settings.py       — AppSettings editor
    _tab_info.py           — About/info panel
"""
