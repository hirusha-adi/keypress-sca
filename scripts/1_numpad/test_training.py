"""Tests for chronological model selection and an untouched evaluation holdout."""

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


if __name__ == "__main__":
    unittest.main()
