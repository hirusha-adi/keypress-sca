"""Tests for full-data neural/classic fitting and optional validation."""

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from sklearn.pipeline import Pipeline

from keypress_model import load_model


SPEC = importlib.util.spec_from_file_location("numpad_training", Path(__file__).with_name("2_train.py"))
training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = training
SPEC.loader.exec_module(training)


class FullDataTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        self.output = self.root / "work"
        # This intentionally has too little data for the old validation split.
        for label, count in zip(training.CLASSES, (1, 2, 1, 2)):
            directory = self.dataset / label
            directory.mkdir(parents=True)
            for ordinal in range(1, count + 1):
                (directory / f"{ordinal}.mp3").write_bytes(f"{label}:{ordinal}".encode())
        self.samples = training.discover_samples(self.dataset)
        self.expected_labels = np.asarray([sample.label for sample in self.samples])
        self.expected_features = np.stack([
            self.features(int(sample.label), sample.ordinal) for sample in self.samples
        ])
        self.decode = self.enterContext(mock.patch.object(
            training, "decode_audio", side_effect=lambda path, sample_rate: np.asarray(
                [int(path.parent.name), int(path.stem)], dtype=np.float32
            )
        ))
        self.extract = self.enterContext(mock.patch.object(
            training, "feature_variants_from_waveform",
            side_effect=lambda waveform, config: {
                config.mode: self.features(int(waveform[0]), int(waveform[1]))
            },
        ))
        for name in ("holdout_split", "repeated_development_folds", "feature_profiles",
                     "classifier_candidates", "_train_with_validation"):
            self.enterContext(mock.patch.object(
                training, name, side_effect=AssertionError(f"classic full-data training called {name}")
            ))
        self.make_pipeline = self.enterContext(mock.patch.object(
            training, "candidate_pipeline", wraps=training.candidate_pipeline
        ))
        self.fit_calls = []
        actual_fit = Pipeline.fit

        def recording_fit(pipeline, features, labels, *args, **kwargs):
            self.fit_calls.append((np.array(features, copy=True), np.array(labels, copy=True)))
            return actual_fit(pipeline, features, labels, *args, **kwargs)

        self.enterContext(mock.patch.object(Pipeline, "fit", new=recording_fit))
        self.enterContext(redirect_stdout(io.StringIO()))

    @staticmethod
    def features(label, ordinal):
        row = np.zeros(80, dtype=np.float64)
        row[label] = 1.0
        row[4] = ordinal / 10.0
        return row

    def previous_validation(self):
        self.output.mkdir()
        report = json.dumps({
            "split_protocol": {"development_count": 4, "holdout_count": 2},
            "heldout_metrics": {"accuracy": 0.5},
        }, indent=2).encode() + b"\n"
        artifacts = {
            "evaluation.json": report,
            "evaluation_model.joblib": b"prior development model",
            "heldout_predictions.csv": b"actual_key,predicted_key\n0,1\n",
            "confusion_matrix.png": b"prior held-out plot",
        }
        for name, content in artifacts.items():
            (self.output / name).write_bytes(content)
        return artifacts

    def snapshot(self):
        return {path.relative_to(self.output).as_posix(): path.read_bytes()
                for path in self.output.rglob("*") if path.is_file()}

    def test_classic_fits_every_clip_once_without_internal_validation(self):
        report = training.train(self.dataset, self.output, classic=True)

        self.assertEqual(len(self.fit_calls), 1)
        np.testing.assert_array_equal(self.fit_calls[0][0], self.expected_features)
        np.testing.assert_array_equal(self.fit_calls[0][1], self.expected_labels)
        self.assertEqual(self.make_pipeline.call_count, 1)
        self.assertEqual(self.decode.call_count, len(self.samples))
        self.assertEqual(self.extract.call_count, len(self.samples))

        artifact = load_model(self.output / "keypress_model.joblib")
        pipeline = artifact["pipeline"]
        self.assertEqual(pipeline.named_steps["standardscaler"].n_samples_seen_, len(self.samples))
        self.assertEqual(pipeline.named_steps["linearsvc"].C, 0.1)
        self.assertEqual(artifact["classes"], list(training.CLASSES))
        self.assertEqual(artifact["feature_config"]["mode"], "mfcc_weighted")
        self.assertEqual(artifact["feature_config"]["n_mfcc"], 40)
        self.assertEqual(artifact["feature_config"]["energy_weight_power"], 0.5)
        np.testing.assert_array_equal(pipeline.predict(self.expected_features), self.expected_labels)
        metadata = artifact["training_metadata"]
        self.assertEqual(metadata["fit_split"], "all_clips")
        self.assertEqual(metadata["training_sample_count"], len(self.samples))
        self.assertIs(metadata["validation_performed"], False)
        for metrics in (metadata, report):
            self.assertNotIn("heldout_accuracy", metrics)
            self.assertNotIn("heldout_macro_f1", metrics)
            self.assertNotIn("heldout_metrics", metrics)
        self.assertEqual(report["training_mode"], "full_data")
        self.assertEqual(report["training_sample_count"], len(self.samples))
        self.assertEqual(report["withheld_sample_count"], 0)
        self.assertIs(report["validation_performed"], False)
        self.assertIs(report["external_test_data_used"], False)
        self.assertEqual([sample["file"] for sample in report["samples"]], [
            sample.path.relative_to(self.dataset).as_posix() for sample in self.samples
        ])
        self.assertTrue(all(sample["split"] == "training" for sample in report["samples"]))
        self.assertEqual(json.loads((self.output / "training.json").read_text()), report)
        self.assertFalse((self.output / "evaluation_model.joblib").exists())
        self.assertFalse((self.output / "heldout_predictions.csv").exists())

    def test_previous_validation_is_preserved_in_a_separate_historical_run(self):
        artifacts = self.previous_validation()
        unrelated = self.output / "rms_peaks.png"
        unrelated.write_bytes(b"segmentation figure")
        latest = self.output / "test_results" / "latest.json"
        latest.parent.mkdir()
        latest.write_bytes(b'{"separate_test": true}\n')

        training.train(self.dataset, self.output, classic=True)

        fingerprint = hashlib.sha256(artifacts["evaluation.json"]).hexdigest()[:16]
        archive = self.output / "runs" / f"validation-{fingerprint}"
        for name, content in artifacts.items():
            with self.subTest(name=name):
                self.assertEqual((archive / name).read_bytes(), content)
                self.assertFalse((self.output / name).exists())
        self.assertEqual(unrelated.read_bytes(), b"segmentation figure")
        self.assertEqual(latest.read_bytes(), b'{"separate_test": true}\n')
        self.assertTrue((self.output / "training.json").is_file())
        load_model(self.output / "keypress_model.joblib")

    def test_failed_feature_extraction_preserves_previous_models_and_reports(self):
        self.previous_validation()
        (self.output / "keypress_model.joblib").write_bytes(b"prior full-data model")
        before = self.snapshot()
        self.extract.side_effect = ValueError("feature extraction failed")

        with self.assertRaisesRegex(ValueError, "feature extraction failed"):
            training.train(self.dataset, self.output, classic=True)

        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.fit_calls)

    def test_failed_fit_preserves_previous_models_and_reports(self):
        self.previous_validation()
        (self.output / "keypress_model.joblib").write_bytes(b"prior full-data model")
        before = self.snapshot()

        with mock.patch.object(Pipeline, "fit", side_effect=ValueError("model fit failed")):
            with self.assertRaisesRegex(ValueError, "model fit failed"):
                training.train(self.dataset, self.output, classic=True)

        self.assertEqual(self.snapshot(), before)


class TrainingCommandTests(unittest.TestCase):
    def test_default_hybrid_training_passes_every_clip_without_splitting_or_reading_test_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, output = root / "2_numpad", root / "work"
            expected_paths = []
            for label, count in zip(training.CLASSES, (1, 2, 1, 2)):
                directory = dataset / label
                directory.mkdir(parents=True)
                for ordinal in range(1, count + 1):
                    path = directory / f"{ordinal}.mp3"
                    path.write_bytes(f"training:{label}:{ordinal}".encode())
                    expected_paths.append(path)
            test_directory = root / "2_tests"
            test_directory.mkdir()
            (test_directory / "0123.mp4").write_bytes(b"separate test recording must remain unread")
            provenance = {"source_recordings": ["training recordings only"]}
            expected_report = {"training_sample_count": len(expected_paths), "validation_performed": False}
            train_hybrid = mock.Mock(return_value=expected_report)
            backend = SimpleNamespace(train_hybrid=train_hybrid)
            original_open = Path.open

            def training_files_only(path, *args, **kwargs):
                if test_directory == path or test_directory in path.parents:
                    raise AssertionError("Training read the separate test data")
                return original_open(path, *args, **kwargs)

            with mock.patch.dict(sys.modules, {"hybrid_training": backend}), \
                    mock.patch.object(training, "source_provenance", return_value=provenance) as sources, \
                    mock.patch.object(Path, "open", new=training_files_only), \
                    mock.patch.object(training, "holdout_split", side_effect=AssertionError("Hybrid training split data")), \
                    mock.patch.object(training, "repeated_development_folds", side_effect=AssertionError("Hybrid training ran CV")), \
                    mock.patch.object(training, "_train_full_data", side_effect=AssertionError("Hybrid training used classic fitting")):
                actual = training.train(dataset, output)
            self.assertIs(actual, expected_report)
            sources.assert_called_once_with(dataset)
            train_hybrid.assert_called_once()
            self.assertEqual(train_hybrid.call_args.args, (dataset, output))
            options = train_hybrid.call_args.kwargs
            self.assertEqual([sample.path for sample in options["samples"]], expected_paths)
            self.assertEqual([sample.label for sample in options["samples"]], ["0", "1", "1", "2", "3", "3"])
            self.assertIs(options["provenance"], provenance)
            self.assertEqual({key: options[key] for key in ("epochs", "batch_size", "device")},
                             {"epochs": 80, "batch_size": 8, "device": "auto"})

    def test_neural_only_dispatches_the_original_deep_trainer_with_all_samples(self):
        dataset, output = Path("2_numpad"), Path("work")
        samples = [training.Sample(dataset / label / "1.mp3", label, 1)
                   for label in training.CLASSES]
        provenance = {"source_recordings": []}
        expected_report = {"training_sample_count": len(samples), "validation_performed": False}
        train_deep = mock.Mock(return_value=expected_report)
        backend = SimpleNamespace(train_deep=train_deep)
        forbidden_hybrid = SimpleNamespace(train_hybrid=mock.Mock(side_effect=AssertionError("Wrong trainer")))
        with mock.patch.dict(sys.modules, {"deep_training": backend, "hybrid_training": forbidden_hybrid}), \
                mock.patch.object(training, "discover_samples", return_value=samples) as discover, \
                mock.patch.object(training, "source_provenance", return_value=provenance), \
                mock.patch.object(training, "holdout_split", side_effect=AssertionError("Unexpected split")), \
                mock.patch.object(training, "repeated_development_folds", side_effect=AssertionError("Unexpected CV")):
            result = training.train(dataset, output, neural_only=True)
        self.assertIs(result, expected_report)
        discover.assert_called_once_with(dataset)
        train_deep.assert_called_once_with(dataset, output, samples=samples, provenance=provenance,
                                           epochs=80, batch_size=8, device="auto")

    def test_validation_is_an_explicit_dispatch(self):
        dataset, output = Path("dataset"), Path("output")
        expected = {"validation_performed": True}
        with mock.patch.object(training, "_train_with_validation", return_value=expected) as validate:
            self.assertIs(training.train(dataset, output, validate=True), expected)
        validate.assert_called_once_with(dataset, output)

    def test_cli_passes_hybrid_neural_classic_validation_and_custom_training_options(self):
        cases = (
            ([], {"validate": False, "classic": False, "neural_only": False, "epochs": 80, "batch_size": 8, "device": "auto"}),
            (["--neural-only"], {"validate": False, "classic": False, "neural_only": True, "epochs": 80, "batch_size": 8, "device": "auto"}),
            (["--classic"], {"validate": False, "classic": True, "neural_only": False, "epochs": 80, "batch_size": 8, "device": "auto"}),
            (["--validate"], {"validate": True, "classic": False, "neural_only": False, "epochs": 80, "batch_size": 8, "device": "auto"}),
            (["--epochs", "75", "--batch-size", "2", "--device", "cpu"],
             {"validate": False, "classic": False, "neural_only": False, "epochs": 75, "batch_size": 2, "device": "cpu"}),
        )
        for flags, expected in cases:
            with self.subTest(flags=flags):
                dataset, output = Path("tiny-dataset"), Path("test-output")
                with mock.patch.object(training.shutil, "which", return_value="/bin/tool"):
                    with mock.patch.object(training, "train", return_value={}) as train:
                        result = training.main([
                            "--dataset-dir", str(dataset), "--output-dir", str(output), *flags
                        ])
                self.assertEqual(result, 0)
                train.assert_called_once_with(dataset.resolve(), output.resolve(), **expected)


class TrainingSplitTests(unittest.TestCase):
    def make_samples(self):
        # Unequal counts exercise rounding and class-specific chronological cuts.
        counts = dict(zip(training.CLASSES, (25, 26, 37, 32)))
        return [training.Sample(Path(f"{key}/{index}.mp3"), str(key), index)
                for key in training.CLASSES for index in range(1, counts[key] + 1)]

    def test_holdout_is_latest_clips_with_an_unused_boundary(self):
        samples = self.make_samples()
        development, purged, holdout = training.holdout_split(samples)
        self.assertEqual(training.CLASSES, ("0", "1", "2", "3"))
        self.assertEqual(len(purged), len(training.CLASSES) * training.PURGE_NEIGHBORS)
        sets = [set(indices) for indices in (development, purged, holdout)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(range(len(samples))))
        for key in training.CLASSES:
            count = sum(sample.label == key for sample in samples)
            held_count = math.ceil(count * training.HOLDOUT_FRACTION)
            boundary = count - held_count - training.PURGE_NEIGHBORS
            self.assertEqual([samples[index].ordinal for index in development if samples[index].label == key],
                             list(range(1, boundary + 1)))
            self.assertEqual([samples[index].ordinal for index in purged if samples[index].label == key],
                             list(range(boundary + 1, count - held_count + 1)))
            self.assertEqual([samples[index].ordinal for index in holdout if samples[index].label == key],
                             list(range(count - held_count + 1, count + 1)))

    def test_development_folds_never_use_holdout_or_neighboring_presses(self):
        samples = self.make_samples()
        development, boundary, holdout = training.holdout_split(samples)
        seen_validation = []
        for train, validation, purged in training.development_folds(samples, development):
            self.assertEqual(set(train) | set(validation) | set(purged), set(development))
            self.assertFalse(set(train) & set(validation))
            self.assertFalse(set(train) & set(purged))
            self.assertFalse((set(train) | set(validation) | set(purged)) & (set(holdout) | set(boundary)))
            seen_validation.extend(validation.tolist())
            for label in training.CLASSES:
                train_order = [samples[index].ordinal for index in train if samples[index].label == label]
                validation_order = [samples[index].ordinal for index in validation if samples[index].label == label]
                self.assertTrue(train_order)
                self.assertEqual(validation_order, list(range(min(validation_order), max(validation_order) + 1)))
                self.assertTrue(all(abs(left - right) > 1 for left in train_order for right in validation_order))
        self.assertEqual(sorted(seen_validation), sorted(development.tolist()))

    def test_discovery_uses_numeric_order_and_rejects_ambiguous_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in training.CLASSES:
                directory = root / label
                directory.mkdir()
                for number in (10, 1, 2):
                    (directory / f"{number}.mp3").touch()
            samples = training.discover_samples(root)
            self.assertEqual([sample.ordinal for sample in samples[:3]], [1, 2, 10])
            (root / "0" / "01.mp3").touch()
            with self.assertRaisesRegex(ValueError, "duplicate numeric"):
                training.discover_samples(root)

    def test_too_little_data_is_rejected_before_model_selection(self):
        samples = [training.Sample(Path(f"{key}/{index}.mp3"), str(key), index)
                   for key in training.CLASSES for index in range(1, 8)]
        with self.assertRaisesRegex(ValueError, "too few clips"):
            training.holdout_split(samples)

    def test_repeated_partitions_cover_only_development_once_per_partition(self):
        samples = self.make_samples()
        development, boundary, holdout = training.holdout_split(samples)
        folds = training.repeated_development_folds(samples, development)
        self.assertEqual(len(folds), sum(training.CV_PARTITIONS))
        self.assertEqual(sum(len(fold["validation"]) for fold in folds),
                         len(development) * len(training.CV_PARTITIONS))
        for n_splits in training.CV_PARTITIONS:
            partition = [fold for fold in folds if fold["n_splits"] == n_splits]
            self.assertEqual(len(partition), n_splits)
            observed = []
            for fold in partition:
                train, validation, purged = fold["training"], fold["validation"], fold["purged"]
                self.assertFalse(set(train) & set(validation))
                self.assertFalse(set(train) & set(purged))
                self.assertFalse(set(validation) & set(purged))
                self.assertEqual(set(train) | set(validation) | set(purged), set(development))
                self.assertFalse(set(train) & (set(boundary) | set(holdout)))
                observed.extend(validation.tolist())
                for label in training.CLASSES:
                    train_ordinals = [samples[index].ordinal for index in train if samples[index].label == label]
                    validation_ordinals = [samples[index].ordinal for index in validation if samples[index].label == label]
                    self.assertTrue(all(abs(left - right) > 1
                                        for left in train_ordinals for right in validation_ordinals))
            self.assertEqual(sorted(observed), sorted(development.tolist()))

    def test_candidate_grid_is_fixed_and_features_are_explicit(self):
        profiles = training.feature_profiles()
        self.assertEqual(len(profiles), 6)
        self.assertEqual(sum(len(training.classifier_candidates(config.n_mfcc * 2))
                             for _, config in profiles), 72)
        configs = dict(profiles)
        self.assertEqual(configs["legacy_mfcc20"].mode, "mfcc_stats")
        self.assertEqual(configs["weighted_mfcc30"].mode, "mfcc_weighted")
        self.assertEqual(configs["weighted_mfcc30"].n_mfcc, 30)
        self.assertEqual(configs["weighted_mfcc30_sqrt"].energy_weight_power, 0.5)
        self.assertEqual(configs["weighted_mfcc40_sqrt"].n_mfcc, 40)

    def test_selection_uses_accuracy_then_worst_fold_then_simplicity(self):
        def candidate(accuracy=0.98, worst=0.95, dimensions=60, kind="linear", c=0.01):
            settings = {"kind": kind}
            if kind != "lda":
                settings["C"] = c
            return {
                "pooled_cv_accuracy": accuracy, "worst_fold_accuracy": worst,
                "feature_count": dimensions, "classifier": settings, "profile_order": 0,
            }

        preferred = candidate()
        alternatives = [
            candidate(accuracy=0.97, worst=1.0),
            candidate(worst=0.90, dimensions=40),
            candidate(dimensions=80),
            candidate(kind="lda"),
            candidate(kind="rbf"),
            candidate(c=0.1),
        ]
        for alternative in alternatives:
            with self.subTest(alternative=alternative):
                self.assertIs(min([alternative, preferred], key=training.candidate_selection_key), preferred)

    def test_invalid_partition_count_is_rejected(self):
        samples = self.make_samples()
        development, _, _ = training.holdout_split(samples)
        for count in (0, 1, 2.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                training.development_folds(samples, development, count)


if __name__ == "__main__":
    unittest.main()
