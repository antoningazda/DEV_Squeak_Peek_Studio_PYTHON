"""
USV call classification (CNN USV/NOISE stage + Random Forest call types).

    core/  — the standalone USV_Klasifikace pipeline, vendored verbatim
    api.py — app bridge: label files / Label lists ↔ pipeline, progress, cancel

Heavy dependencies (torch, sklearn) load only when api functions run.
"""
