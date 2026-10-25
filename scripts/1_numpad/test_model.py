"""Feature and inference checks independent of classifier training accuracy."""

from dataclasses import asdict
import io
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from scipy.io import wavfile

import keypress_model as model


def synthetic_press(sample_rate: int = 24000) -> np.ndarray:
    """A broadband press and quieter release, with decay and quiet around them."""
    rng = np.random.default_rng(47)
    waveform = np.zeros(round(0.75 * sample_rate), dtype=np.float64)
    for onset, amplitude in ((0.20, 0.65), (0.29, 0.35)):
        t = np.arange(round(0.14 * sample_rate)) / sample_rate
        carrier = (
            0.6 * rng.normal(size=len(t))
            + np.sin(2 * np.pi * 1850 * t)
            + 0.4 * np.sin(2 * np.pi * 4100 * t)
        )
        sound = amplitude * carrier * np.exp(-t / 0.018)
        start = round(onset * sample_rate)
        waveform[start:start + len(sound)] += sound
    waveform *= 0.8 / np.max(np.abs(waveform))
    return waveform


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.config = model.FeatureConfig()
        self.waveform = synthetic_press(self.config.sample_rate)

    def test_all_feature_variants_have_finite_expected_dimensions(self):
        features = model.feature_variants_from_waveform(self.waveform, self.config)
        expected = {
            "mfcc_stats": 40,
            "logmel_stats": 128,
            "logmel_patch": 2048,
            "logmel_temporal": 320,
            "mfcc_weighted": 40,
            "logmel_weighted": 128,
        }
        self.assertEqual(set(features), set(expected))
        for mode, size in expected.items():
            with self.subTest(mode=mode):
                self.assertEqual(features[mode].shape, (size,))
                self.assertTrue(np.isfinite(features[mode]).all())

    def test_energy_weighting_reduces_background_influence(self):
        original = model.feature_variants_from_waveform(self.waveform, self.config)
        rng = np.random.default_rng(149)
        with_background = self.waveform + rng.normal(0, 0.0001, len(self.waveform))
        noisy = model.feature_variants_from_waveform(with_background, self.config)
        plain_change = np.linalg.norm(noisy["mfcc_stats"] - original["mfcc_stats"])
        weighted_change = np.linalg.norm(noisy["mfcc_weighted"] - original["mfcc_weighted"])
        self.assertLess(weighted_change, plain_change * 0.2)

    def test_weighted_thirty_mfcc_configuration_and_legacy_defaults(self):
        config = model.FeatureConfig(mode="mfcc_weighted", n_mfcc=30)
        variants = model.feature_variants_from_waveform(self.waveform, config)
        self.assertEqual(variants[config.mode].shape, (60,))
        legacy = asdict(self.config)
        legacy.pop("energy_weight_power")
        restored = model.FeatureConfig(**legacy)
        np.testing.assert_array_equal(
            model.feature_variants_from_waveform(self.waveform, restored)["mfcc_stats"],
            model.feature_variants_from_waveform(self.waveform, self.config)["mfcc_stats"],
        )

    def test_leading_quiet_and_recording_gain_do_not_change_features(self):
        original = model.feature_variants_from_waveform(self.waveform, self.config)
        shifted = np.pad(self.waveform * 0.2, (self.config.sample_rate // 2, 0))
        transformed = model.feature_variants_from_waveform(shifted, self.config)
        for mode in original:
            with self.subTest(mode=mode):
                np.testing.assert_allclose(original[mode], transformed[mode], rtol=1e-7, atol=1e-7)

    def test_silence_stationary_noise_and_non_finite_audio_are_rejected(self):
        rng = np.random.default_rng(71)
        invalid = {
            "silence": np.zeros_like(self.waveform),
            "stationary_noise": rng.normal(0, 0.005, size=len(self.waveform)),
            "nan": self.waveform.copy(),
            "infinity": self.waveform.copy(),
        }
        invalid["nan"][100] = np.nan
        invalid["infinity"][100] = np.inf
        for name, waveform in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                model.feature_variants_from_waveform(waveform, self.config)

    def test_multiple_separated_presses_are_rejected(self):
        waveform = np.concatenate((self.waveform, self.waveform))
        with self.assertRaises(ValueError):
            model.feature_variants_from_waveform(waveform, self.config)

    def test_pcm_wav_and_direct_waveform_use_the_same_preprocessing(self):
        pcm = np.round(self.waveform * 30000).astype(np.int16)
        reference = pcm.astype(np.float64) / 32768
        expected = model.feature_variants_from_waveform(reference, self.config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "press.wav"
            wavfile.write(path, self.config.sample_rate, pcm)
            actual = model.feature_variants(path, self.config)
        for mode in expected:
            with self.subTest(mode=mode):
                np.testing.assert_allclose(expected[mode], actual[mode], rtol=1e-6, atol=1e-6)


class ModelPersistenceTests(unittest.TestCase):
    def test_waveform_prediction_reports_native_probabilities_separately_from_accuracy(self):
        config = model.FeatureConfig()
        rng = np.random.default_rng(59)
        features = rng.normal(size=(60, 40))
        labels = np.repeat([str(key) for key in range(10)], 6)
        waveform = synthetic_press()
        probe = model.feature_variants_from_waveform(waveform, config)[config.mode].reshape(1, -1)
        for kind in ("lda", "rbf", "linear"):
            with self.subTest(kind=kind):
                pipeline = model.build_classifier(0.01, kind=kind)
                pipeline.fit(features, labels)
                artifact = {"pipeline": pipeline, "feature_config": asdict(config)}
                result = model.predict_waveform(waveform, artifact)
                self.assertNotIn("accuracy", result)
                self.assertEqual(result["predicted_key"], pipeline.predict(probe)[0])
                if kind == "lda":
                    index = list(pipeline.classes_).index(result["predicted_key"])
                    self.assertAlmostEqual(result["confidence"], pipeline.predict_proba(probe)[0, index])
                    self.assertEqual(result["confidence_type"], "uncalibrated_model_probability")
                else:
                    self.assertIsNone(result["confidence"])
                    self.assertEqual(result["confidence_type"], "unavailable")
                    self.assertTrue(all(row["probability"] is None for row in result["ranked_keys"]))

    def test_file_and_native_rate_waveform_prediction_share_the_same_model(self):
        config = model.FeatureConfig()
        rng = np.random.default_rng(61)
        pipeline = model.build_classifier(0.01, kind="lda")
        pipeline.fit(rng.normal(size=(60, 40)), np.repeat([str(key) for key in range(10)], 6))
        artifact = {"pipeline": pipeline, "feature_config": asdict(config)}
        waveform = synthetic_press().astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "press.wav"
            wavfile.write(path, config.sample_rate, waveform)
            from_file = model.predict_key(path, artifact)
        from_memory = model.predict_waveform(waveform, artifact, sample_rate=config.sample_rate)
        self.assertEqual(from_file["predicted_key"], from_memory["predicted_key"])
        self.assertEqual(from_file["ranked_keys"], from_memory["ranked_keys"])
        self.assertEqual(from_file["confidence"], from_memory["confidence"])

    def test_classifier_families_preserve_decision_scores_after_reload(self):
        rng = np.random.default_rng(29)
        features = rng.normal(size=(60, 24))
        labels = np.repeat([str(key) for key in range(10)], 6)
        probes = rng.normal(size=(4, 24))
        for kind in ("rbf", "linear", "lda"):
            with self.subTest(kind=kind):
                pipeline = model.build_classifier(0.01, kind=kind)
                pipeline.fit(features, labels)
                scores = pipeline.decision_function(probes)
                self.assertEqual(scores.shape, (4, 10))
                self.assertTrue(np.isfinite(scores).all())
                np.testing.assert_array_equal(
                    pipeline.predict(probes), pipeline.classes_[np.argmax(scores, axis=1)]
                )
                stream = io.BytesIO()
                joblib.dump(pipeline, stream)
                stream.seek(0)
                restored = joblib.load(stream)
                np.testing.assert_allclose(restored.decision_function(probes), scores)
                np.testing.assert_array_equal(restored.predict(probes), pipeline.predict(probes))

    def test_saved_configuration_and_prediction_survive_serialization(self):
        # Arbitrary feature vectors test persistence and inference plumbing only;
        # they make no claim about acoustic classification accuracy.
        config = model.FeatureConfig(n_mfcc=12, duration_seconds=0.30, pre_peak_seconds=0.03)
        rng = np.random.default_rng(19)
        features = rng.normal(size=(30, 24))
        labels = np.repeat([str(key) for key in range(10)], 3)
        pipeline = model.build_classifier(1.0)
        pipeline.fit(features, labels)
        artifact = {
            "schema_version": model.MODEL_SCHEMA_VERSION,
            "pipeline": pipeline,
            "feature_config": asdict(config),
            "classes": [str(key) for key in range(10)],
            "training_metadata": {"test_fixture": True},
        }
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "press.wav"
            wavfile.write(audio, config.sample_rate, synthetic_press().astype(np.float32))
            path = Path(temporary) / "model.joblib"
            joblib.dump(artifact, path)
            restored = model.load_model(path)
            self.assertEqual(restored["feature_config"], asdict(config))
            before = model.predict_key(audio, artifact, top_k=3)
            after = model.predict_key(audio, restored, top_k=3)
        self.assertEqual(before["predicted_key"], after["predicted_key"])
        self.assertIn(after["predicted_key"], artifact["classes"])
        self.assertEqual(len(after["ranked_keys"]), 3)
        self.assertEqual(after["predicted_key"], after["ranked_keys"][0]["key"])
        self.assertEqual(
            [entry["key"] for entry in before["ranked_keys"]],
            [entry["key"] for entry in after["ranked_keys"]],
        )
        scores = [entry["score"] for entry in after["ranked_keys"]]
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(scores, sorted(scores, reverse=True))
        np.testing.assert_allclose(scores, [entry["score"] for entry in before["ranked_keys"]])


if __name__ == "__main__":
    unittest.main()
