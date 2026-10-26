"""Neural frontend and artifact tests; no pretrained downloads or microphone."""

from dataclasses import asdict
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    import deep_keypress as deep
    from keypress_model import load_model, predict_waveform

    class TinyNetwork(nn.Module):
        def __init__(self, classes=4):
            super().__init__()
            self.head = nn.Linear(3, classes)

        def forward(self, image):
            return self.head(image.mean(dim=(2, 3)))


@unittest.skipUnless(HAS_TORCH, "Install requirements-deep.txt to test the neural backend")
class DeepModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = deep.DeepConfig()
        sample_rate = self.config.sample_rate
        self.waveform = np.random.default_rng(2).normal(0, 1e-5, sample_rate).astype(np.float32)
        time = np.arange(round(0.1 * sample_rate)) / sample_rate
        pulse = 0.7 * np.sin(2 * np.pi * 9500 * time) * np.exp(-time / 0.012)
        self.waveform[sample_rate // 3:sample_rate // 3 + len(pulse)] += pulse.astype(np.float32)

    def artifact(self):
        torch.manual_seed(42)
        network = TinyNetwork()
        return {
            "schema_version": 2, "backend": "torch_spectrogram_ensemble",
            "classes": ["0", "1", "2", "3"],
            "deep_config": asdict(self.config),
            "feature_config": asdict(deep.shared_feature_config(self.config)),
            "members": [{"architecture": "swin_t", "seed": 42,
                         "state_dict": deep.state_to_numpy(network)}],
        }

    def test_frontend_is_deterministic_finite_and_retains_high_frequencies(self):
        prepared = deep.prepare_waveform(self.waveform, self.config)
        frontend = deep.SpectrogramFrontend(self.config)
        signal = torch.from_numpy(prepared)[None]
        image = frontend(signal)
        self.assertEqual(tuple(image.shape), (1, 3, 224, 224))
        self.assertTrue(torch.isfinite(image).all())
        self.assertTrue(torch.equal(image, frontend(signal)))
        # The 9.5 kHz transient must survive preprocessing, rather than being
        # discarded by an 8 kHz Nyquist limit as in many speech models.
        power = np.abs(np.fft.rfft(prepared)) ** 2
        frequencies = np.fft.rfftfreq(len(prepared), 1 / self.config.sample_rate)
        self.assertGreater(float(power[frequencies > 8000].sum() / power.sum()), 0.95)

    def test_augmentation_is_seeded_and_never_mutates_cached_audio(self):
        original = torch.from_numpy(deep.prepare_waveform(self.waveform, self.config))[None]
        before = original.clone()
        frontend = deep.SpectrogramFrontend(self.config)
        torch.manual_seed(42)
        first = frontend(original, augment=True)
        torch.manual_seed(42)
        second = frontend(original, augment=True)
        torch.manual_seed(43)
        third = frontend(original, augment=True)
        self.assertTrue(torch.equal(original, before))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))
        self.assertTrue(torch.isfinite(first).all())

    def test_persisted_numpy_weights_load_and_predict_without_pretrained_download(self):
        artifact = self.artifact()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.joblib"
            joblib.dump(artifact, path)
            restored = load_model(path)
            self.assertNotIn("_deep_predictor", restored)
            with patch.object(deep, "create_network", side_effect=lambda architecture, classes: TinyNetwork(classes)) as factory, \
                    patch.object(torch.cuda, "is_available", return_value=False):
                result = predict_waveform(self.waveform, restored, top_k=4, sample_rate=32000)
                repeated = predict_waveform(self.waveform, restored, top_k=4, sample_rate=32000)
            factory.assert_called_once_with("swin_t", 4)
            self.assertEqual(result, repeated)
            self.assertEqual(len(result["ranked_keys"]), 4)
            self.assertAlmostEqual(sum(item["probability"] for item in result["ranked_keys"]), 1, places=6)
            self.assertEqual(result["predicted_key"], result["ranked_keys"][0]["key"])
            self.assertEqual(result["confidence"], result["ranked_keys"][0]["probability"])
            self.assertIn("not measured accuracy", result["confidence_type"])

    def test_invalid_weights_and_shared_configuration_are_rejected(self):
        artifact = self.artifact()
        artifact["members"][0]["state_dict"]["head.weight"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            deep.validate_artifact(artifact)
        artifact = self.artifact()
        artifact["feature_config"]["sample_rate"] = 24000
        with self.assertRaisesRegex(ValueError, "configurations differ"):
            deep.validate_artifact(artifact)

    def test_silent_or_nonfinite_audio_does_not_receive_a_key_prediction(self):
        artifact = self.artifact()
        for waveform in (np.zeros(32000), np.full(32000, np.nan)):
            with self.subTest(finite=np.isfinite(waveform).all()), self.assertRaises(ValueError):
                predict_waveform(waveform, artifact)


if __name__ == "__main__":
    unittest.main()
