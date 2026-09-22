"""
Unit tests for the squeak-peek-cli commands (WP8: cli/main.py wiring).
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from cli.main import cli


class TestDetectCommand:
    def test_detect_psd_on_real_audio(self, example_wav_path: Path, tmp_path: Path, settings) -> None:
        settings_path = tmp_path / "settings.json"
        settings.save_json(settings_path)
        out_path = tmp_path / "detected.txt"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "detect",
                str(example_wav_path),
                "--detector", "psd",
                "--settings", str(settings_path),
                "--output", str(out_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert out_path.exists()
        assert "events detected" in result.output

        lines = out_path.read_text().splitlines()
        assert len(lines) % 2 == 0  # 2-line-per-label format

    def test_min_tonality_filters_broadband_detections(
        self, example_wav_path: Path, tmp_path: Path, settings
    ) -> None:
        """--min-tonality is opt-in and must only ever remove detections."""
        settings_path = tmp_path / "settings.json"
        settings.save_json(settings_path)
        runner = CliRunner()

        def run(extra: list[str], out_name: str) -> int:
            out_path = tmp_path / out_name
            result = runner.invoke(
                cli,
                ["detect", str(example_wav_path), "--detector", "psd",
                 "--settings", str(settings_path), "--output", str(out_path)] + extra,
            )
            assert result.exit_code == 0, result.output
            return len(out_path.read_text().splitlines()) // 2

        n_default = run([], "default.txt")
        n_filtered = run(["--min-tonality", "0.5"], "filtered.txt")
        n_zero = run(["--min-tonality", "0"], "zero.txt")

        assert n_filtered < n_default, "tonality filter removed nothing"
        assert n_zero == n_default, "0 must disable the filter, matching the default"

    def test_detect_ml_without_model_configured_fails_clearly(
        self, example_wav_path: Path, tmp_path: Path, settings
    ) -> None:
        settings_path = tmp_path / "settings.json"
        settings.save_json(settings_path)  # default settings: Detection.ML.modelPath == ""

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["detect", str(example_wav_path), "--detector", "ml", "--settings", str(settings_path)],
        )

        assert result.exit_code != 0
        assert "model" in result.output.lower()

    def test_detect_default_output_path(self, example_wav_path: Path, tmp_path: Path, settings) -> None:
        # Copy the wav into a writable tmp dir so the default output path lands there
        import shutil

        wav_copy = tmp_path / "clip.wav"
        shutil.copy(example_wav_path, wav_copy)
        settings_path = tmp_path / "settings.json"
        settings.save_json(settings_path)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["detect", str(wav_copy), "--detector", "psd", "--settings", str(settings_path)]
        )

        assert result.exit_code == 0, result.output
        expected = tmp_path / "clip_psd_detected.txt"
        assert expected.exists()


class TestEvaluateCommand:
    def test_evaluate_against_real_reference(
        self, example_labels_path: Path, example_ref_labels_path: Path
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", str(example_labels_path), str(example_ref_labels_path)])

        assert result.exit_code == 0, result.output
        assert "Precision" in result.output
        assert "Recall" in result.output
        assert "F1 Score" in result.output

    def test_evaluate_perfect_match(self, tmp_path: Path) -> None:
        labels_text = "1.000000\t2.000000\td\n\\\t0.000000\t0.000000\n"
        det_path = tmp_path / "det.txt"
        ref_path = tmp_path / "ref.txt"
        det_path.write_text(labels_text)
        ref_path.write_text(labels_text)

        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", str(det_path), str(ref_path)])

        assert result.exit_code == 0, result.output
        assert "1.000" in result.output  # perfect precision/recall/F1


class TestTrainCommand:
    def test_train_then_detect_ml_end_to_end(
        self, example_wav_path: Path, example_ref_labels_path: Path, tmp_path: Path, settings
    ) -> None:
        model_path = tmp_path / "model.joblib"
        runner = CliRunner()

        train_result = runner.invoke(
            cli,
            [
                "train", str(example_wav_path),
                "--labels", str(example_ref_labels_path),
                "--output", str(model_path),
                "--n-trees", "50",
            ],
        )
        assert train_result.exit_code == 0, train_result.output
        assert model_path.exists()
        assert "OOB accuracy" in train_result.output

        settings.detection.ml.modelPath = str(model_path)
        settings_path = tmp_path / "settings.json"
        settings.save_json(settings_path)
        out_path = tmp_path / "ml_detected.txt"

        detect_result = runner.invoke(
            cli,
            [
                "detect", str(example_wav_path),
                "--detector", "ml",
                "--settings", str(settings_path),
                "--output", str(out_path),
            ],
        )
        assert detect_result.exit_code == 0, detect_result.output
        assert out_path.exists()

    def test_train_mismatched_wav_and_label_counts_fails(
        self, example_wav_path: Path, example_ref_labels_path: Path, tmp_path: Path
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "train", str(example_wav_path), str(example_wav_path),
                "--labels", str(example_ref_labels_path),
                "--output", str(tmp_path / "model.joblib"),
            ],
        )
        assert result.exit_code != 0
        assert "WAV_PATHS" in result.output or "labels" in result.output.lower()


class TestBatchCommand:
    def test_batch_no_wav_files(self, tmp_path: Path, settings) -> None:
        settings_path = tmp_path / "settings.json"
        settings.save_json(settings_path)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli, ["batch", str(empty_dir), "--detector", "psd", "--settings", str(settings_path)]
        )

        assert result.exit_code == 0
        assert "No .wav files found" in result.output
