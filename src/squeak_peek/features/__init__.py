"""
Acoustic feature extraction.

This module implements:
    extract_frame_features()  — 12-D feature vector per audio frame
                                (BandPower, SpecCentroid, SpecSpread,
                                 SpecFlatness, SpecEntropy, ZCR, SNR_est,
                                 SpecFlux, DomFreq, Delta_BandPower,
                                 Delta_Centroid, Delta_Entropy)
"""

from .extract import extract_frame_features

__all__ = ["extract_frame_features"]
