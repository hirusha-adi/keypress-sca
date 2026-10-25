"""Tests for frozen-model evaluation, ground-truth ordering, and report integrity."""

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location("numpad_evaluation", Path(__file__).with_name("4_evaluate.py"))
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


def make_sequence(root: Path, sequence: str) -> Path:
    directory = root / sequence
    directory.mkdir(parents=True)
    for press, key in reversed(list(enumerate(sequence, 1))):
        (directory / f"{press}-{key}.mp3").write_bytes(f"fixture press {press}".encode())
    (root / f"{sequence}.m4a").write_bytes(b"fixture source recording")
    return directory


def ranked_prediction(path, artifact, top_k):
    # An explicit independent prediction fixture has two correct decisions, one
    # wrong first choice, and the actual key among the top three in every case.
    rankings = {"1-0.mp3": ["0", "1", "2"], "2-1.mp3": ["2", "1", "0"],
                "3-2.mp3": ["2", "3", "4"]}
    keys = rankings[path.name]
    return {"predicted_key": keys[0], "ranked_keys": [
        {"key": key, "score": float(3 - index)} for index, key in enumerate(keys)]}


class EvaluationInputTests(unittest.TestCase):
    def test_discovery_orders_press_ten_after_press_nine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_sequence(root, "0123456789")
            result = evaluation.discover_recordings(root)
        self.assertEqual([clip.press_number for clip in result[0].clips], list(range(1, 11)))
        self.assertEqual([clip.actual_key for clip in result[0].clips], list("0123456789"))

    def test_malformed_duplicate_missing_and_wrong_labels_are_rejected(self):
        for problem in ("malformed", "duplicate", "missing", "wrong_label"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                directory = make_sequence(root, "012")
                if problem == "malformed":
                    (directory / "old-name.mp3").write_bytes(b"bad name")
                elif problem == "duplicate":
                    (directory / "1-1.mp3").write_bytes(b"duplicate press")
                elif problem == "missing":
                    (directory / "2-1.mp3").unlink()
                else:
                    (directory / "2-1.mp3").rename(directory / "2-7.mp3")
                with self.assertRaises(ValueError):
                    evaluation.discover_recordings(root)

    def test_manifest_source_and_label_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_sequence(root, "012")
            item = {"source": "012.m4a", "key_sequence": "012", "detected_presses": 3,
                    "source_sha256": evaluation.sha256(root / "012.m4a"), "clips": [
                        {"file": f"012/{index}-{key}.mp3", "press_number": index, "key": key}
                        for index, key in enumerate("012", 1)]}
            manifest_path = root / "split_manifest.json"
            manifest_path.write_text(json.dumps({"recordings": [item]}))
            discovered = evaluation.discover_recordings(root)
            provenance = evaluation.validate_provenance(root, discovered)
            self.assertTrue(provenance["recordings"]["012"]["source_hash_verified"])
            item["clips"][1]["key"] = "7"
            manifest_path.write_text(json.dumps({"recordings": [item]}))
            with self.assertRaisesRegex(ValueError, "labels/order"):
                evaluation.validate_provenance(root, discovered)
            item["clips"][1]["key"] = "1"
            manifest_path.write_text(json.dumps({"recordings": [item]}))
            (root / "012.m4a").write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "source hash"):
                evaluation.validate_provenance(root, discovered)


class FrozenEvaluationTests(unittest.TestCase):
    def test_overall_metrics_pool_clips_across_recordings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_sequence(root / "tests", "012")
            make_sequence(root / "tests", "7")
            model_path = root / "model.joblib"
            model_path.write_bytes(b"frozen model")

            def predict(path, artifact, top_k):
                if path.name == "1-7.mp3":
                    return {"predicted_key": "0", "ranked_keys": [
                        {"key": key, "score": float(3 - index)}
                        for index, key in enumerate(("0", "1", "2"))]}
                return ranked_prediction(path, artifact, top_k)

            with patch.object(evaluation, "load_model", return_value={}), \
                    patch.object(evaluation, "predict_key", side_effect=predict), \
                    contextlib.redirect_stdout(io.StringIO()):
                report = evaluation.evaluate(root / "tests", model_path, root / "reports")
            self.assertEqual((report["correct"], report["sample_count"]), (2, 4))
            self.assertEqual(report["accuracy"], 0.5)
            self.assertEqual(report["top3_accuracy"], 0.75)
            self.assertEqual([item["test_sequence"] for item in report["recordings"]], ["012", "7"])

    def test_metrics_reports_and_model_fingerprint_are_consistent_without_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tests_dir = root / "tests"
            make_sequence(tests_dir, "012")
            model_path = root / "model.joblib"
            model_path.write_bytes(b"frozen model")
            pipeline = Mock()
            pipeline.fit.side_effect = AssertionError("Evaluation must never fit the model")
            artifact = {"pipeline": pipeline, "training_metadata": {"training_sample_count": 251}}
            expected_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
            with patch.object(evaluation, "load_model", return_value=artifact), \
                    patch.object(evaluation, "predict_key", side_effect=ranked_prediction) as predict, \
                    contextlib.redirect_stdout(io.StringIO()):
                report = evaluation.evaluate(tests_dir, model_path, root / "reports")
            pipeline.fit.assert_not_called()
            self.assertEqual(predict.call_count, 3)
            self.assertEqual(report["correct"], 2)
            self.assertAlmostEqual(report["accuracy"], 2 / 3)
            self.assertEqual(report["top3_correct"], 3)
            self.assertEqual(report["top3_accuracy"], 1)
            self.assertEqual(report["model_sha256"], expected_hash)
            self.assertEqual(report["model_training_sample_count"], 251)
            self.assertEqual(report["score_type"], "Uncalibrated classifier decision score; not a probability")
            self.assertEqual(evaluation.sha256(model_path), expected_hash)
            run_dir = Path(report["report_directory"])
            self.assertEqual(json.loads((run_dir / "evaluation.json").read_text()), report)
            self.assertEqual(json.loads((root / "reports/latest.json").read_text()), report)
            per_recording = json.loads((run_dir / "012.json").read_text())
            self.assertEqual(per_recording["expected_sequence"], "012")
            self.assertEqual(per_recording["predicted_sequence"], "022")
            self.assertEqual(per_recording["correct"], 2)
            with (run_dir / "predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["press_number"] for row in rows], ["1", "2", "3"])
            self.assertEqual([row["predicted_key"] for row in rows], ["0", "2", "2"])

    def test_different_models_preserve_previous_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_sequence(root / "tests", "012")
            model_path = root / "model.joblib"
            model_path.write_bytes(b"model one")
            with patch.object(evaluation, "load_model", return_value={}), \
                    patch.object(evaluation, "predict_key", side_effect=ranked_prediction), \
                    contextlib.redirect_stdout(io.StringIO()):
                first = evaluation.evaluate(root / "tests", model_path, root / "reports")
                old_report = Path(first["report_directory"]) / "evaluation.json"
                original_bytes = old_report.read_bytes()
                model_path.write_bytes(b"model two")
                second = evaluation.evaluate(root / "tests", model_path, root / "reports")
            self.assertNotEqual(first["report_directory"], second["report_directory"])
            self.assertEqual(old_report.read_bytes(), original_bytes)

    def test_model_changed_during_prediction_fails_before_writing_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_sequence(root / "tests", "012")
            model_path = root / "model.joblib"
            model_path.write_bytes(b"model before")

            def mutate(path, artifact, top_k):
                model_path.write_bytes(b"model after")
                return ranked_prediction(path, artifact, top_k)

            with patch.object(evaluation, "load_model", return_value={}), \
                    patch.object(evaluation, "predict_key", side_effect=mutate), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(ValueError, "Model file changed"):
                evaluation.evaluate(root / "tests", model_path, root / "reports")
            self.assertFalse((root / "reports").exists())


if __name__ == "__main__":
    unittest.main()
