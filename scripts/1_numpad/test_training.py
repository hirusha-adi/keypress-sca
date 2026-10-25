"""Tests for chronological model selection and evaluation split isolation."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("numpad_training", Path(__file__).with_name("2_train.py"))
training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = training
SPEC.loader.exec_module(training)


class TrainingSplitTests(unittest.TestCase):
    def make_samples(self):
        return [training.Sample(Path(f"{key}/{index}.mp3"), str(key), index)
                for key in range(10) for index in range(1, (27 if key == 6 else 26))]

    def test_holdout_is_latest_clips_with_an_unused_boundary(self):
        samples = self.make_samples()
        development, purged, holdout = training.holdout_split(samples)
        self.assertEqual((len(development), len(purged), len(holdout)), (190, 10, 51))
        sets = [set(indices) for indices in (development, purged, holdout)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(range(len(samples))))
        for key in training.CLASSES:
            self.assertEqual([samples[index].ordinal for index in development if samples[index].label == key], list(range(1, 20)))
            self.assertEqual([samples[index].ordinal for index in purged if samples[index].label == key], [20])
            self.assertEqual(min(samples[index].ordinal for index in holdout if samples[index].label == key), 21)

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
                   for key in range(10) for index in range(1, 8)]
        with self.assertRaisesRegex(ValueError, "too few clips"):
            training.holdout_split(samples)

    def test_repeated_partitions_cover_only_development_once_per_partition(self):
        samples = self.make_samples()
        development, boundary, holdout = training.holdout_split(samples)
        folds = training.repeated_development_folds(samples, development)
        self.assertEqual(len(folds), 12)
        self.assertEqual(sum(len(fold["validation"]) for fold in folds), 570)
        for n_splits in (3, 4, 5):
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
