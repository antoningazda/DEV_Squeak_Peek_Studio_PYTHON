from copy import deepcopy
import numpy as np
from .common import require_columns


def feature_schema():
    schema = deepcopy(_SCHEMA)
    schema["FullMinusContour"] = [c for c in schema["AllUnique"] if c not in schema["ContourOnly"]]
    schema["DurationAmplitude"] = ["Duration_ms", "RMS"]
    return schema


def feature_sets():
    s = feature_schema()
    return {k:s[k] for k in ("Compact", "CompactPlusEntropy", "MorphologyRich", "AllUnique", "ContourOnly", "NoContour", "NonFrequencyBaseline", "FullMinusContour", "DurationAmplitude")}


def feature_group_weights(columns):
    groups = []
    for f in columns:
        if f in ("Duration_ms", "ICI_ms"):
            g = "Temporal"
        elif f in ("RMS", "SNR_dB", "ZCR"):
            g = "SignalQuality"
        elif f.startswith("Spec") or f == "Bandwidth_Hz":
            g = "Spectrum"
        elif f in ("DomFreqMedian_Hz", "DomFreqMean_Hz", "F0Mean_Hz", "PeakFreq_Hz"):
            g = "FrequencyLocation"
        elif f.startswith(("DomFreq", "F0", "FM")) or f in ("DirectionChanges", "DirectionChangeRate_per100ms", "LinearityResidual_Hz", "MaxJump_Hz"):
            g = "ContourShape"
        else:
            g = "Other"
        groups.append(g)
    return np.array([1/np.sqrt(groups.count(g)) for g in groups]), groups


def matrix(table, columns):
    require_columns(table, columns)
    return table.loc[:, columns].to_numpy(dtype=float)


def requires_contour(columns):
    return bool(set(columns) & set(feature_schema()["ContourOnly"] + feature_schema()["LegacyAliases"][3:]))


_SCHEMA = {'Version': ['USV_FEATURES_V03'],
 'Compact': ['Duration_ms',
             'DomFreqMedian_Hz',
             'DomFreqRobustRange_Hz',
             'DomFreqLinearSlope_HzPerS',
             'DirectionChangeRate_per100ms',
             'LinearityResidual_Hz',
             'MaxJump_Hz',
             'SpecSpread_Hz'],
 'CompactPlusEntropy': ['Duration_ms',
                        'DomFreqMedian_Hz',
                        'DomFreqRobustRange_Hz',
                        'DomFreqLinearSlope_HzPerS',
                        'DirectionChangeRate_per100ms',
                        'LinearityResidual_Hz',
                        'MaxJump_Hz',
                        'SpecSpread_Hz',
                        'SpecEntropy'],
 'MorphologyRich': ['Duration_ms',
                    'DomFreqMedian_Hz',
                    'DomFreqRobustRange_Hz',
                    'DomFreqLinearSlope_HzPerS',
                    'DomFreqTotalVariation_HzPerS',
                    'DirectionChangeRate_per100ms',
                    'LinearityResidual_Hz',
                    'MaxJump_Hz',
                    'SpecSpread_Hz',
                    'SpecEntropy'],
 'AllUnique': ['Duration_ms',
               'RMS',
               'ZCR',
               'SpecCentroid_Hz',
               'SpecSpread_Hz',
               'SpecFlatness',
               'SpecEntropy',
               'DomFreqMedian_Hz',
               'DomFreqStd_Hz',
               'DomFreqRobustRange_Hz',
               'DomFreqLinearSlope_HzPerS',
               'DomFreqTotalVariation_HzPerS',
               'DirectionChangeRate_per100ms',
               'LinearityResidual_Hz',
               'MaxJump_Hz',
               'SNR_dB'],
 'ContourOnly': ['DomFreqMedian_Hz',
                 'DomFreqStd_Hz',
                 'DomFreqRobustRange_Hz',
                 'DomFreqLinearSlope_HzPerS',
                 'DomFreqTotalVariation_HzPerS',
                 'DirectionChangeRate_per100ms',
                 'LinearityResidual_Hz',
                 'MaxJump_Hz'],
 'NoContour': ['Duration_ms',
               'RMS',
               'ZCR',
               'SpecCentroid_Hz',
               'SpecSpread_Hz',
               'SpecFlatness',
               'SpecEntropy',
               'SNR_dB'],
 'NonFrequencyBaseline': ['Duration_ms', 'RMS', 'ZCR', 'SpecFlatness', 'SpecEntropy', 'SNR_dB'],
 'Quality': ['ContourValid',
             'TrackCoverage',
             'MedianPeakProminence_dB',
             'TrackJumpP95_Hz',
             'FractionInterpolated',
             'NumTrackGaps',
             'LongestTrackGap_ms'],
 'LegacyAliases': ['PeakFreq_Hz',
                   'PeakFreqStd_Hz',
                   'Bandwidth_Hz',
                   'F0Mean_Hz',
                   'F0Std_Hz',
                   'F0Range_Hz',
                   'F0MeanAbsDiff_Hz',
                   'F0DirectionChanges',
                   'F0LineFitResidual_Hz',
                   'FMRate_HzPerS',
                   'FMSlopeMean',
                   'FMSlopeStd',
                   'FMInflections'],
 'OutputNumeric': ['Duration_ms',
                   'ICI_ms',
                   'RMS',
                   'ZCR',
                   'SpecCentroid_Hz',
                   'SpecSpread_Hz',
                   'SpecFlatness',
                   'SpecEntropy',
                   'SNR_dB',
                   'DomFreqMean_Hz',
                   'DomFreqMedian_Hz',
                   'DomFreqStd_Hz',
                   'DomFreqRobustRange_Hz',
                   'DomFreqLinearSlope_HzPerS',
                   'DomFreqTotalVariation_HzPerS',
                   'DirectionChanges',
                   'DirectionChangeRate_per100ms',
                   'LinearityResidual_Hz',
                   'MaxJump_Hz',
                   'TrackCoverage',
                   'MedianPeakProminence_dB',
                   'TrackJumpP95_Hz',
                   'FractionInterpolated',
                   'NumTrackGaps',
                   'LongestTrackGap_ms',
                   'PeakFreq_Hz',
                   'PeakFreqStd_Hz',
                   'Bandwidth_Hz',
                   'F0Mean_Hz',
                   'F0Std_Hz',
                   'F0Range_Hz',
                   'F0MeanAbsDiff_Hz',
                   'F0DirectionChanges',
                   'F0LineFitResidual_Hz',
                   'FMRate_HzPerS',
                   'FMSlopeMean',
                   'FMSlopeStd',
                   'FMInflections'],
 'OutputLogical': ['ContourValid'],
 'OutputString': ['FeatureVersion', 'ContourStatus']}
