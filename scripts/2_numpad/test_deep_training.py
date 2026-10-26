"""Small real-optimizer tests for full-data deep training and safe publication."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import joblib
import numpy as np


HAS_DEEP_DEPENDENCIES = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("torchvision") is not None
)
if HAS_DEEP_DEPENDENCIES:
    import torch
    from torch import nn
    import deep_training as training

    class TinyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(8, 8)
            self.head = nn.Linear(8, 4)

        def forward(self, features):
            return self.head(torch.tanh(self.backbone(features)))

    class TinyFrontend(nn.Module):
        observed = []

        def __init__(self, config):
            super().__init__()

        def forward(self, waveform, augment=False):
            # Distinct fixture amplitudes identify every example reaching fitting.
            self.observed.extend(waveform[:, 0].detach().cpu().tolist())
            return waveform[:, :8]


@unittest.skipUnless(HAS_DEEP_DEPENDENCIES, "optional torch/torchvision dependencies are not installed")
class DeepTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset, self.output = self.root / "dataset", self.root / "work"
        self.samples = []
        # The uneven six-row dataset also exercises a partial final batch.
        for label, count in zip(training.CLASSES, (1, 2, 1, 2)):
            directory = self.dataset / label
            directory.mkdir(parents=True)
            for ordinal in range(1, count + 1):
                path = directory / f"{ordinal}.mp3"
                path.write_bytes(f"{label}:{ordinal}".encode())
                self.samples.append(SimpleNamespace(path=path, label=label, ordinal=ordinal))
        self.config = training.DeepConfig()
        self.expected_values = [float(index + 1) for index in range(len(self.samples))]
        self.values = {sample.path: self.expected_values[index] for index, sample in enumerate(self.samples)}
        self.decode = self.enterContext(mock.patch.object(
            training, "decode_audio", side_effect=lambda path, rate: np.full(
                round(self.config.sample_rate * self.config.duration_seconds), self.values[path], np.float32
            ),
        ))
        self.enterContext(mock.patch.object(training, "prepare_waveform", side_effect=lambda wave, config: wave))
        self.create = self.enterContext(mock.patch.object(
            training, "create_network", side_effect=lambda architecture, classes, pretrained: TinyNetwork()
        ))
        self.enterContext(mock.patch.object(training, "SpectrogramFrontend", TinyFrontend))
        self.enterContext(redirect_stdout(io.StringIO()))
        TinyFrontend.observed = []

    def run_training(self, **kwargs):
        options = dict(samples=self.samples, provenance={"source_recordings": []},
                       epochs=2, batch_size=4, device="cpu", architectures=("swin_t",), config=self.config)
        options.update(kwargs)
        return training.train_deep(self.dataset, self.output, **options)

    def test_every_clip_reaches_every_epoch_without_reading_test_data(self):
        original_read = Path.read_bytes

        def check_read(path):
            self.assertNotIn("tests", path.parts)
            self.assertNotIn("2_tests", path.parts)
            return original_read(path)

        with mock.patch.object(Path, "read_bytes", check_read):
            report = self.run_training()
        self.assertEqual(sorted(TinyFrontend.observed), sorted(self.expected_values * 2))
        self.assertEqual(self.decode.call_count, 6)
        self.create.assert_called_once_with("swin_t", 4, pretrained=True)
        self.assertEqual(report["training_sample_count"], 6)
        self.assertEqual(report["withheld_sample_count"], 0)
        self.assertFalse(report["validation_performed"])
        self.assertFalse(report["cross_validation_performed"])
        self.assertFalse(report["external_test_data_used"])
        self.assertNotIn("heldout_accuracy", report)
        self.assertEqual({sample["split"] for sample in report["samples"]}, {"training"})
        history = json.loads((self.output / "training_history.json").read_text())
        self.assertEqual([row["training_examples_seen"] for row in history], [6, 6])
        self.assertEqual([row["epoch"] for row in history], [1, 2])
        artifact = joblib.load(self.output / "keypress_model.joblib")
        training.validate_artifact(artifact)
        self.assertEqual(artifact["schema_version"], 2)
        self.assertEqual(artifact["members"][0]["architecture"], "swin_t")
        self.assertEqual(artifact["training_metadata"]["fit_split"], "all_clips")
        self.assertEqual(json.loads((self.output / "training.json").read_text()), report)

    def test_completed_matching_members_and_prepared_audio_are_reused(self):
        first = self.run_training()
        self.decode.reset_mock()
        with mock.patch.object(training, "_train_member", side_effect=AssertionError("Matching member retrained")):
            second = self.run_training()
        self.decode.assert_not_called()
        self.assertEqual(first["run_fingerprint"], second["run_fingerprint"])
        self.assertEqual(first["final_augmented_training_metrics"], second["final_augmented_training_metrics"])

    def test_failed_member_leaves_deployed_model_and_reports_unchanged(self):
        self.output.mkdir(parents=True)
        previous = {
            "keypress_model.joblib": b"old deployed model",
            "training.json": b'{"old": true}',
            "summary.json": b'{"accuracy": 0.9}',
            "training.log": b"old training log\n",
        }
        for name, data in previous.items():
            (self.output / name).write_bytes(data)
        with mock.patch.object(training, "_train_member", side_effect=RuntimeError("fixture failed")):
            with self.assertRaisesRegex(RuntimeError, "fixture failed"):
                self.run_training()
        for name, data in previous.items():
            self.assertEqual((self.output / name).read_bytes(), data)
        self.assertFalse((self.output / "runs").exists())

    def test_success_archives_exact_previous_bytes_before_publishing(self):
        self.output.mkdir(parents=True)
        previous = {
            "keypress_model.joblib": b"old deployed model",
            "training.json": b'{"old": true}',
            "summary.json": b'{"accuracy": 0.9}',
            "training.log": b"old training log\n",
        }
        for name, data in previous.items():
            (self.output / name).write_bytes(data)
        report = self.run_training()
        archive = self.output / report["previous_model_archive"]
        for name, data in previous.items():
            self.assertEqual((archive / name).read_bytes(), data)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertIsNone(summary["separate_test"])
        self.assertEqual(summary["training_sample_count"], 6)
        self.assertNotEqual((self.output / "keypress_model.joblib").read_bytes(), previous["keypress_model.joblib"])


if __name__ == "__main__":
    unittest.main()
