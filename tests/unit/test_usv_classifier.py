"""
Tests for squeak_peek.usv_classifier (vendored USV_Klasifikace pipeline +
app bridge) and the USV_MODEL classifier plugin.

Label → training-example mapping runs without torch. The end-to-end test
(train on synthetic recordings, then classify from files and from an
in-memory signal) needs torch and is skipped without the 'cnn' extra; it
checks plumbing and agreement between code paths, not model accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from squeak_peek.audio.io import save_wav
from squeak_peek.labels.io import export_labels, import_labels
from squeak_peek.labels.model import Label
from squeak_peek.usv_classifier import api

FS = 250_000


def _sweep(f0: float, f1: float, dur: float, rng) -> np.ndarray:
    t = np.arange(int(dur * FS)) / FS
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * dur))
    return 0.3 * np.sin(phase) * np.hanning(len(t)) + 0.001 * rng.standard_normal(len(t))


def _make_recording(folder: Path, name: str, seed: int, n: int = 24):
    """WAV with flat ('5'), upsweep ('5t') calls and broadband clicks.

    Returns (wav, call-type labels file, detected-labels file); detected
    labels cover every call plus the clicks (which only the detector saw).
    """
    rng = np.random.default_rng(seed)
    x = 0.002 * rng.standard_normal(int((n * 0.2 + 0.5) * FS))
    calls, clicks = [], []
    for i in range(n):
        start = 0.1 + i * 0.2
        dur = 0.03 + 0.01 * rng.random()
        kind = "5" if i % 2 == 0 else "5t"
        f0, f1 = (55e3, 56e3) if kind == "5" else (40e3, 80e3)
        seg = _sweep(f0, f1, dur, rng)
        a = int(start * FS)
        x[a:a + len(seg)] += seg
        calls.append(Label(start, start + dur, kind))
        c = start + 0.13
        b = int(c * FS)
        x[b:b + int(0.01 * FS)] += 0.2 * rng.standard_normal(int(0.01 * FS))
        clicks.append(Label(c, c + 0.01, "d"))
    wav = folder / f"{name}.wav"
    save_wav(wav, x.astype(np.float32), FS)
    ref = folder / f"{name}_ref.txt"
    export_labels(ref, calls)
    det = folder / f"{name}_det.txt"
    export_labels(det, sorted([Label(c.start_time, c.end_time, "d") for c in calls] + clicks,
                              key=lambda lbl: lbl.start_time))
    return wav, ref, det


# ── Label mapping (no torch) ─────────────────────────────────────────────

class TestTrainingSet:
    def test_label_sources(self, tmp_path):
        wavs = []
        for i in range(5):
            wavs.append(_make_recording(tmp_path, f"rec{i}", i, n=4))
        # Rec 0: one generic 'd', one explicit noise, one rejected detection.
        labels = import_labels(wavs[0][1])
        labels[0].label = "d"
        labels[1].label = "Noise"
        labels[2].detection_state = "Rejected"
        export_labels(wavs[0][1], labels)

        inputs = [api.TrainingInput(w, r, d) for w, r, d in wavs]
        ts = api.build_training_set(inputs, tmp_path / "ts", api.TrainingOptions())
        rec0 = ts.review[ts.review.RecordingID == "rec0"].sort_values("Start_s")
        assert list(rec0.ExpertFinalLabel[:3]) == ["USV", "NOISE", "NOISE"]
        # 4 calls + 4 clicks per recording; clicks overlap no call → NOISE.
        assert len(rec0) == 8
        assert (rec0.ExpertFinalLabel == "NOISE").sum() == 2 + 4
        assert set(ts.review.ExpertBinaryLabel) == {"USV", "NOISE"}
        assert set(ts.manifest.Split) == {"train", "calibration", "test"}
        assert ts.review.CallID.is_unique

    def test_no_detected_file_means_no_unmatched_noise(self, tmp_path):
        recs = [_make_recording(tmp_path, f"r{i}", i, n=4) for i in range(5)]
        ts = api.build_training_set([api.TrainingInput(w, r) for w, r, _ in recs],
                                    tmp_path / "ts", api.TrainingOptions())
        assert set(ts.review.ExpertFinalLabel) == {"5", "5t"}

    def test_mixed_auto_and_manual_split_rejected(self, tmp_path):
        recs = [_make_recording(tmp_path, f"r{i}", i, n=2) for i in range(5)]
        inputs = [api.TrainingInput(w, r, split="train" if i else "auto") for i, (w, r, _) in enumerate(recs)]
        with pytest.raises(ValueError, match="Split"):
            api.build_training_set(inputs, tmp_path / "ts", api.TrainingOptions())

    def test_animal_members_join_groups(self, tmp_path):
        recs = [_make_recording(tmp_path, f"r{i}", i, n=2) for i in range(6)]
        groups = ["A;B", "B;C", "D", "E", "F", "G"]
        inputs = [api.TrainingInput(w, r, group=g) for (w, r, _), g in zip(recs, groups)]
        ts = api.build_training_set(inputs, tmp_path / "ts", api.TrainingOptions())
        m = ts.manifest.set_index("RecordingID")
        assert m.loc["r0", "GroupID"] == m.loc["r1", "GroupID"]
        assert m.loc["r0", "Split"] == m.loc["r1", "Split"]


def test_predictions_to_labels():
    import pandas as pd

    src = [Label(0.0, 0.1, "d", 1.0, 2.0), Label(0.2, 0.3, "d"), Label(0.4, 0.5, "d")]
    pred = pd.DataFrame({
        "SourceRow": [1, 2, 3],
        "ClusterPred": ["5t", "NOISE", "UNCERTAIN"],
        "RawClusterPred": ["5t", "", "c5"],
    })
    out = api.predictions_to_labels(pred, src)
    assert [lbl.label for lbl in out] == ["5t", "NOISE", "c5?"]
    assert out[0].start_frequency == 1.0
    assert [lbl.classification_state for lbl in out] == ["Accepted", "Rejected", "Rejected"]
    out = api.predictions_to_labels(pred, src, drop_noise=True, guess_uncertain=False)
    assert [lbl.label for lbl in out] == ["5t", "UNCERTAIN"]


def test_predictions_to_labels_state_suffix_round_trip(tmp_path):
    import pandas as pd

    src = [Label(0.0, 0.1, "d", detection_state="Accepted"), Label(0.2, 0.3, "d")]
    pred = pd.DataFrame({"SourceRow": [1, 2], "ClusterPred": ["5t", "UNCERTAIN"],
                         "RawClusterPred": ["5t", "c5"]})
    out = api.predictions_to_labels(pred, src)
    export_labels(tmp_path / "x.txt", out)
    text = (tmp_path / "x.txt").read_text()
    assert "5t_DC" in text and "c5?_xc" in text
    back = import_labels(tmp_path / "x.txt")
    assert [(lbl.detection_state, lbl.classification_state) for lbl in back] == [
        ("Accepted", "Accepted"), ("None", "Rejected")]


def test_write_back_target(tmp_path):
    wav, txt = tmp_path / "rec.wav", tmp_path / "det" / "rec_PSD.txt"
    with_file = api.ClassifyInput(wav, [], labels_file=txt)
    loaded = api.ClassifyInput(wav, [])
    assert api.write_back_target(with_file, "none") is None
    assert api.write_back_target(with_file, "beside") == tmp_path / "det" / "rec_PSD_classified.txt"
    assert api.write_back_target(with_file, "overwrite") == txt
    assert api.write_back_target(loaded, "beside") == tmp_path / "rec_classified.txt"
    assert api.write_back_target(loaded, "overwrite") == tmp_path / "rec_classified.txt"
    with pytest.raises(ValueError):
        api.write_back_target(with_file, "bogus")


def test_find_matching_file(tmp_path):
    (tmp_path / "rec_PSD_1_detected.txt").write_text("")
    (tmp_path / "other.txt").write_text("")
    assert api.find_matching_file(tmp_path, "x/rec.wav").name == "rec_PSD_1_detected.txt"
    assert api.find_matching_file(tmp_path, "x/none.wav") is None


def test_vendored_core_matches_layout():
    """The pipeline modules the bridge relies on are importable without torch."""
    from squeak_peek.usv_classifier.core import FEATURE_VERSION, schema
    assert FEATURE_VERSION == "USV_PY_FEATURES_1"
    assert "Compact" in schema.feature_sets()


# ── End to end (torch) ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("usv")
    recs = [_make_recording(root, f"rec{i}", 100 + i) for i in range(5)]
    inputs = [api.TrainingInput(w, r, d) for w, r, d in recs]
    opts = api.TrainingOptions(epochs=2, trees=20, min_examples=2)
    messages = []
    result = api.train_model(inputs, root / "out", opts, progress=lambda f, m: messages.append(m))
    return root, recs, result, messages


class TestEndToEnd:
    def test_training_outputs(self, trained):
        _root, _recs, result, messages = trained
        assert result.status == "COMPLETED"
        assert (result.model_dir / "manifest.json").is_file()
        assert set(result.info.classes) == {"5", "5t"}
        assert result.metrics is not None and result.metrics["N"] > 0
        assert any("Extracting" in m for m in messages)

    def test_global_torch_state_restored(self, trained):
        import torch
        assert not torch.are_deterministic_algorithms_enabled()

    def test_classify_files_and_signal_agree(self, trained):
        root, recs, result, _ = trained
        wav, _ref, det = recs[0]
        labels = import_labels(det)
        res = api.classify_recordings([api.ClassifyInput(wav, labels)], result.model_dir, root / "cls")
        for name in ("predictions.csv", "expert_review_queue.csv", "calls_features.csv",
                     "inference_manifest.json"):
            assert (res.out_dir / name).is_file()
        assert len(res.predictions) == len(labels)
        out_labels = import_labels(res.label_files["rec0"])
        assert len(out_labels) == len(labels)
        assert all(lbl.classification_state != "None" for lbl in out_labels)
        assert res.written_back == {}

        from squeak_peek.audio.io import load_wav
        samples, fs = load_wav(wav)
        sig = api.classify_signal(labels, samples, fs, result.model_dir)
        np.testing.assert_array_equal(sig.ClusterPred.to_numpy(str),
                                      res.predictions.ClusterPred.to_numpy(str))
        np.testing.assert_allclose(sig.pUSV.to_numpy(float), res.predictions.pUSV.to_numpy(float))

    def test_matches_standalone_master_inference(self, trained):
        """Bridge output == the tool's own master_inference on the same inputs."""
        import pandas as pd

        from squeak_peek.usv_classifier.core.entry import classify
        root, recs, result, _ = trained
        wav, _ref, det = recs[1]
        labels = import_labels(det)
        res = api.classify_recordings([api.ClassifyInput(wav, labels, "rec1")], result.model_dir, root / "a")
        manifest = pd.DataFrame([{"RecordingID": "rec1", "WavFile": str(wav),
                                  "TxtFile": str(res.out_dir / "inputs" / "rec1.txt")}])
        classify(manifest, result.model_dir, root / "b")
        assert (root / "a" / "predictions.csv").read_text() == (root / "b" / "predictions.csv").read_text()

    def test_noise_free_labels(self, trained, tmp_path):
        from squeak_peek.labels.io import import_labels

        root, recs, result, _ = trained
        wav, _ref, det = recs[2]
        labels = import_labels(det)
        res = api.classify_recordings([api.ClassifyInput(wav, labels)],
                                      result.model_dir, tmp_path / "keep")
        full = import_labels(res.label_files["rec2"])
        clean = import_labels(res.noise_free_files["rec2"])
        # Guards against the assertions below going vacuous if the fixture
        # ever stops producing NOISE predictions.
        assert any(lbl.label == "NOISE" for lbl in full)
        assert [lbl for lbl in clean if lbl.label == "NOISE"] == []
        assert len(clean) == len([lbl for lbl in full if lbl.label != "NOISE"])
        assert [(lbl.start_time, lbl.label) for lbl in clean] == [
            (lbl.start_time, lbl.label) for lbl in full if lbl.label != "NOISE"]

        # drop_noise already yields that file, so no second one is written.
        res = api.classify_recordings([api.ClassifyInput(wav, labels)],
                                      result.model_dir, tmp_path / "drop", drop_noise=True)
        assert res.noise_free_files == {}

    def test_write_back(self, trained, tmp_path):
        import shutil
        root, recs, result, _ = trained
        wav, _ref, det = recs[3]
        src = tmp_path / det.name
        shutil.copy(det, src)
        original = src.read_text()
        labels = import_labels(src)
        res = api.classify_recordings([api.ClassifyInput(wav, labels, labels_file=src)],
                                      result.model_dir, tmp_path / "beside", write_back="beside")
        beside = src.with_name(f"{src.stem}_classified.txt")
        assert res.written_back == {"rec3": beside}
        assert beside.read_text() == res.label_files["rec3"].read_text()
        assert src.read_text() == original
        res = api.classify_recordings([api.ClassifyInput(wav, labels, labels_file=src)],
                                      result.model_dir, tmp_path / "over", write_back="overwrite")
        assert src.read_text() == res.label_files["rec3"].read_text()

    def test_plugin(self, trained):
        from squeak_peek.audio.io import load_wav
        from squeak_peek.classifiers.usv_model import USVModelClassifier, USVModelParams
        _root, recs, result, _ = trained
        wav, _ref, det = recs[2]
        samples, fs = load_wav(wav)
        labels = import_labels(det)
        params = USVModelParams(modelPath=str(result.model_dir / "manifest.json"), dropNoise=True)
        out = USVModelClassifier(params).classify(labels, samples, fs)
        pred = api.classify_signal(labels, samples, fs, result.model_dir)
        assert len(out) == int((pred.ClusterPred != "NOISE").sum())
        assert all(lbl.label != "NOISE" for lbl in out)

    def test_plugin_without_model_raises(self):
        from squeak_peek.classifiers.usv_model import USVModelClassifier, USVModelParams
        with pytest.raises(RuntimeError, match="No USV model"):
            USVModelClassifier(USVModelParams()).classify([Label(0, 0.1)], np.zeros(FS), FS)
