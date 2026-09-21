"""
Machine Learning pipeline for the ML (Random Forest) detector.

    train.py    — train_model(): sliding-window Random Forest training
                  (ports MLDetectorTrain.m)
    optimize.py — sweep_sensitivity()/sweep_noise_ratio(): threshold and
                  class-balance calibration against a validation set
                  (ports MLDetectorOptimize.m)

See squeak_peek.detectors.ml.MLDetector for inference.
"""
