"""
Unit tests for squeak_peek.config.

Covers:
- Default model construction (no file needed)
- Loading from the MATLAB default.json
- Correct types and values for all sections
- Round-trip serialisation (save → load)
"""

from __future__ import annotations

import pytest

from squeak_peek.config import AppSettings, LabelEditSettings, PostProcessParams
from squeak_peek.detectors.bscd import BSCDParams
from squeak_peek.detectors.psd import PSDParams
from squeak_peek.detectors.rbd import RBDParams

# ── Default construction ────────────────────────────────────────────────────

class TestDefaults:
    def test_can_construct_with_no_args(self):
        s = AppSettings()
        assert s is not None

    def test_psd_defaults(self):
        p = PSDParams()
        assert p.fcutMin == 40_000
        assert p.fcutMax == 120_000
        assert p.segmentLength == 8192

    def test_bscd_defaults(self):
        b = BSCDParams()
        assert b.wlen == pytest.approx(0.01)

    def test_rbd_defaults(self):
        r = RBDParams()
        assert r.AR_order_left == 4
        assert r.AR_order_right == 4

    def test_post_defaults(self):
        pp = PostProcessParams()
        assert pp.maxGapToMerge == pytest.approx(0.005)
        assert pp.minLabelLength == pytest.approx(0.001)

    def test_label_edit_classification_list(self):
        le = LabelEditSettings()
        assert le.classification_list == ["d", "sk", "5", "5t", "5w", "c5"]


# ── Loading from MATLAB default.json ───────────────────────────────────────

class TestFromJson:
    def test_loads_without_error(self, settings):
        assert isinstance(settings, AppSettings)

    def test_psd_fcut_min(self, settings):
        assert settings.detection.params_for("PSD").fcutMin == 40_000

    def test_psd_fcut_max(self, settings):
        assert settings.detection.params_for("PSD").fcutMax == 120_000

    def test_psd_segment_length(self, settings):
        assert settings.detection.params_for("PSD").segmentLength == 8192

    def test_psd_overlap_factor(self, settings):
        assert settings.detection.params_for("PSD").overlapFactor == pytest.approx(0.59)

    def test_psd_k_and_w(self, settings):
        assert settings.detection.params_for("PSD").k == pytest.approx(0.023)
        assert settings.detection.params_for("PSD").w == pytest.approx(0.994)

    def test_psd_min_effective_power(self, settings):
        assert settings.detection.params_for("PSD").minEffectivePower == pytest.approx(8.5e-5)

    def test_bscd_wlen(self, settings):
        assert settings.detection.params_for("BSCD").wlen == pytest.approx(0.01)

    def test_bscd_ma_window(self, settings):
        assert settings.detection.params_for("BSCD").maWindow == 5_000

    def test_rbd_wlen(self, settings):
        assert settings.detection.params_for("RBD").wlen == pytest.approx(0.04)

    def test_rbd_dynamic_scaling(self, settings):
        assert settings.detection.params_for("RBD").dynamicScaling == pytest.approx(0.3)

    def test_post_max_gap(self, settings):
        assert settings.detection.post.maxGapToMerge == pytest.approx(0.005)

    def test_post_min_label_length(self, settings):
        assert settings.detection.post.minLabelLength == pytest.approx(0.001)

    def test_visualization_colormap(self, settings):
        assert settings.visualization.colormap == "parula"

    def test_visualization_segment_length(self, settings):
        assert settings.visualization.segment_length_seconds == pytest.approx(1.0)

    def test_visualization_spectrogram_window(self, settings):
        assert settings.visualization.spectrogram_window == 1024

    def test_visualization_hz_properties(self, settings):
        assert settings.visualization.spectrogram_min_freq_hz == pytest.approx(40_000)
        assert settings.visualization.spectrogram_max_freq_hz == pytest.approx(120_000)

    def test_label_edit_classifications(self, settings):
        assert "d" in settings.label_edit.classification_list
        assert "c5" in settings.label_edit.classification_list
        assert len(settings.label_edit.classification_list) == 6

    def test_batch_mode_default_false(self, settings):
        assert settings.data_input.batch_mode is False

    def test_ml_min_event_duration(self, settings):
        assert settings.detection.params_for("ML").minEventDuration == pytest.approx(0.003)


# ── Round-trip serialisation ────────────────────────────────────────────────

class TestRoundTrip:
    def test_save_and_reload(self, settings, tmp_path):
        out = tmp_path / "settings_out.json"
        settings.save_json(out)
        reloaded = AppSettings.from_json(out)

        # Spot-check a handful of values
        assert reloaded.detection.params_for("PSD").fcutMin == settings.detection.params_for("PSD").fcutMin
        assert reloaded.detection.params_for("RBD").wlen == pytest.approx(settings.detection.params_for("RBD").wlen)
        assert reloaded.visualization.colormap == settings.visualization.colormap

    def test_defaults_round_trip(self, tmp_path):
        s = AppSettings.defaults()
        out = tmp_path / "defaults.json"
        s.save_json(out)
        s2 = AppSettings.from_json(out)
        assert s2.detection.params_for("PSD").fcutMin == s.detection.params_for("PSD").fcutMin
