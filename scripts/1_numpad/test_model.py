"""Feature and inference checks independent of classifier training accuracy."""

from dataclasses import asdict
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
        }
        self.assertEqual(set(features), set(expected))
        for mode, size in expected.items():
            with self.subTest(mode=mode):
                self.assertEqual(features[mode].shape, (size,))
                self.assertTrue(np.isfinite(features[mode]).all())

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
