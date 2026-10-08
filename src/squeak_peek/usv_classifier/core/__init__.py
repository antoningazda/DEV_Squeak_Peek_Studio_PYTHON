"""Rat USV pipeline. No MATLAB runtime or pretrained weights required.

Vendored from the standalone USV_Klasifikace tool (usv/ package, v0.2.0).
The numerics are kept verbatim so models and results stay interchangeable
with that tool. Changes, each marked "Squeak Peek addition": the optional
``audio`` argument of pipeline.classify_calls (classify an in-memory signal)
rf.fit_forest (works around a scikit-learn 1.9 bug, same trees) and the
pandas-3-safe train-table hash in pipeline.train_pipeline. App integration
lives one level up, in squeak_peek.usv_classifier.api.
"""
__version__ = "0.2.0"
FEATURE_VERSION = "USV_PY_FEATURES_1"
