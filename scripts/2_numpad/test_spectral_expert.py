"""Spectral training, persistence, and inference checks without audio hardware."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np

import spectral_expert as spectral


def fixtures():
    rate = spectral.CONFIG["sample_rate"]
    times = np.arange(round(rate * spectral.CONFIG["duration_seconds"])) / rate - .08
    waves, labels = [], []
    for label, frequency in enumerate((1700, 3400, 6100, 9800)):
        for repetition in range(2):
            local = np.maximum(times, 0)
            wave = np.sin(2 * np.pi * (frequency + repetition * 11) * local)
            wave += .2 * np.sin(2 * np.pi * (frequency + 1000) * local)
            wave *= np.exp(-local / .025) * (times >= 0)
            wave /= np.max(np.abs(wave))
            waves.append(wave.astype(np.float32))
            labels.append(str(label))
    return np.stack(waves), np.asarray(labels)


class SpectralExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.waves, cls.labels = fixtures()
        # Training accepts arrays only and must never reach for evaluation files.
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("Training read a file")):
            cls.expert = spectral.fit_expert(cls.waves, cls.labels, seed=7)

    def test_all_original_clips_and_three_variants_reach_all_six_fits(self):
        metadata = self.expert["training_metadata"]
        self.assertEqual(metadata["original_training_sample_count"], 8)
        self.assertEqual(metadata["augmented_training_sample_count"], 32)
        self.assertEqual(metadata["augmentation_factor"], 4)
        self.assertEqual(metadata["withheld_sample_count"], 0)
        self.assertFalse(metadata["validation_performed"])
        self.assertFalse(metadata["cross_validation_performed"])
        self.assertFalse(metadata["test_audio_used_for_fitting"])
        self.assertTrue(metadata["benchmark_used_for_development"])
        self.assertEqual(metadata["class_counts"], {key: 2 for key in spectral.CLASSES})
        self.assertEqual(len(self.expert["members"]), 6)
        self.assertEqual({(member["view"], member["kind"]) for member in self.expert["members"]},
                         {(view, kind) for view in spectral.VIEWS for kind in ("linear", "lda")})
        for member in self.expert["members"]:
            self.assertEqual(member["pipeline"].named_steps["standardscaler"].n_samples_seen_, 32)
            self.assertEqual(member["pipeline"].n_features_in_, spectral.DIMENSIONS[member["view"]])

    def test_features_and_predictions_preserve_batch_order_and_separate_distinct_transients(self):
        features = spectral.extract_features(self.waves)
        for view, dimensions in spectral.DIMENSIONS.items():
            self.assertEqual(features[view].shape, (8, dimensions))
            self.assertTrue(np.isfinite(features[view]).all())
            reverse = spectral.extract_features(self.waves[::-1])[view]
            np.testing.assert_allclose(reverse, features[view][::-1], rtol=1e-12, atol=1e-12)
        for waveform, label in zip(self.waves, self.labels):
            probabilities = spectral.predict_probabilities(waveform, self.expert)
            self.assertEqual(probabilities.shape, (4,))
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue((probabilities >= 0).all())
            self.assertAlmostEqual(float(probabilities.sum()), 1)
            self.assertEqual(self.expert["classes"][np.argmax(probabilities)], label)

    def test_augmentation_is_reproducible_and_retains_original_rows(self):
        first = spectral._augment_training(self.waves, np.random.default_rng(5))
        repeated = spectral._augment_training(self.waves, np.random.default_rng(5))
        changed = spectral._augment_training(self.waves, np.random.default_rng(6))
        np.testing.assert_array_equal(first[:8], self.waves)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first[8:], changed[8:]))
        self.assertTrue(np.isfinite(first).all())

    def test_joblib_round_trip_preserves_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spectral.joblib"
            joblib.dump(self.expert, path, compress=3)
            restored = joblib.load(path)
        spectral.validate_expert(restored)
        np.testing.assert_array_equal(
            spectral.predict_probabilities(self.waves[0], self.expert),
            spectral.predict_probabilities(self.waves[0], restored),
        )

    def test_missing_heads_wrong_classes_and_invalid_inputs_are_rejected(self):
        for alteration in ("members", "classes", "config"):
            malformed = copy.copy(self.expert)
            if alteration == "members":
                malformed["members"] = malformed["members"][:-1]
            elif alteration == "classes":
                malformed["classes"] = ["1", "0", "2", "3"]
            else:
                malformed["config"] = {**malformed["config"], "n_mels": 64}
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                spectral.validate_expert(malformed)
        with self.assertRaisesRegex(ValueError, "all keys"):
            spectral.fit_expert(self.waves[:2], self.labels[:2])
        with self.assertRaisesRegex(ValueError, "prepared"):
            spectral.extract_features(self.waves[:, :-1])
        invalid = self.waves[0].copy()
        invalid[100] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            spectral.predict_probabilities(invalid, self.expert)


if __name__ == "__main__":
    unittest.main()
