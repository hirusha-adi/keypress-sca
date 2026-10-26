"""Hybrid fusion and training publication checks without neural inference."""

import contextlib
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np
from scipy.signal import resample_poly

import hybrid_keypress as hybrid
import hybrid_training as training
from keypress_model import FeatureConfig


@dataclass(frozen=True)
class FixtureDeepConfig:
    sample_rate: int = 32000
    duration_seconds: float = 0.32
    pre_peak_seconds: float = 0.08


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_artifact():
    features = asdict(FeatureConfig(
        sample_rate=32000, duration_seconds=0.32, pre_peak_seconds=0.08,
        n_fft=1024, hop_length=128, n_mels=224, n_mfcc=40, fmax=15500,
    ))
    neural = {
        "schema_version": 2, "backend": "torch_spectrogram_ensemble",
        "classes": list("0123"), "feature_config": features,
        "deep_config": asdict(FixtureDeepConfig()),
        "members": [{"architecture": "swin_t", "state_dict": {
            "head.weight": np.ones((4, 3), dtype=np.float16),
        }}],
        "training_metadata": {
            "training_sample_count": 4, "fit_split": "all_clips",
            "epochs_per_member": 80, "parameter_count": 141792184,
            "implementation_sha256": {"original_neural_implementation": "preserved"},
        },
    }
    expert = {"classes": list("0123"), "training_metadata": {"original_training_sample_count": 4}}
    return {
        "schema_version": 2, "backend": "neural_spectral_hybrid",
        "classes": list("0123"), "feature_config": features,
        "neural_model": neural, "spectral_expert": expert,
        "neural_weight": 0.2,
    }


class HybridPredictionTests(unittest.TestCase):
    def setUp(self):
        self.artifact = make_artifact()
        self.prepared = np.ones(10240, dtype=np.float32)
        # Deliberately shuffled class order: neural probability positions are
        # unrelated to the fixed acoustic class-vector order.
        self.neural_result = {
            "predicted_key": "1", "ranked_keys": [
                {"key": "1", "probability": 0.85},
                {"key": "3", "probability": 0.05},
                {"key": "0", "probability": 0.05},
                {"key": "2", "probability": 0.05},
            ],
        }
        self.deep = SimpleNamespace(
            DeepConfig=FixtureDeepConfig,
            prepare_waveform=Mock(return_value=self.prepared),
            predict_waveform=Mock(return_value=self.neural_result),
            validate_artifact=Mock(), warmup=Mock(),
        )
        self.spectral = SimpleNamespace(
            predict_probabilities=Mock(return_value=np.array([0.15, 0.05, 0.70, 0.10])),
            validate_expert=Mock(),
        )
        self.enterContext(patch.dict(sys.modules, {"deep_keypress": self.deep, "spectral_expert": self.spectral}))

    def test_fusion_aligns_classes_and_retains_normalized_probabilities_and_log_margin(self):
        waveform = np.linspace(-0.5, 0.5, 16000)
        result = hybrid.predict_waveform(waveform, self.artifact, top_k=4, sample_rate=32000)
        expected = {"0": 0.13, "1": 0.21, "2": 0.57, "3": 0.09}
        self.assertEqual(result["predicted_key"], "2")
        self.assertEqual([item["key"] for item in result["ranked_keys"]], ["2", "1", "0", "3"])
        self.assertAlmostEqual(sum(item["probability"] for item in result["ranked_keys"]), 1)
        for item in result["ranked_keys"]:
            self.assertAlmostEqual(item["probability"], expected[item["key"]])
            self.assertAlmostEqual(item["score"], np.log(expected[item["key"]]))
        self.assertAlmostEqual(result["confidence"], 0.57)
        self.assertAlmostEqual(result["margin"], np.log(0.57 / 0.21))
        self.assertIn("Uncalibrated", result["confidence_type"])
        self.assertNotIn("accuracy", result)
        self.assertIs(self.spectral.predict_probabilities.call_args.args[0], self.prepared)
        self.assertIs(self.spectral.predict_probabilities.call_args.args[1], self.artifact["spectral_expert"])
        self.assertEqual(self.deep.predict_waveform.call_args.kwargs, {"top_k": 4, "sample_rate": 32000})

    def test_native_48k_input_is_resampled_in_full_before_either_expert(self):
        waveform = np.linspace(-0.4, 0.6, 24000)
        expected = resample_poly(waveform, 2, 3)
        result = hybrid.predict_waveform(waveform, self.artifact, top_k=2, sample_rate=48000)
        self.assertEqual(len(result["ranked_keys"]), 2)
        prepared_input = self.deep.prepare_waveform.call_args.args[0]
        neural_input = self.deep.predict_waveform.call_args.args[0]
        np.testing.assert_array_equal(prepared_input, expected)
        np.testing.assert_array_equal(neural_input, expected)
        self.assertEqual(len(prepared_input), 16000)
        self.assertIs(prepared_input, neural_input)
        self.assertIs(self.spectral.predict_probabilities.call_args.args[0], self.prepared)
        self.assertEqual(self.deep.predict_waveform.call_args.kwargs["sample_rate"], 32000)

    def test_low_rate_capture_and_invalid_top_k_fail_before_any_prediction(self):
        cases = [({"sample_rate": 24000}, "32 kHz or higher")]
        cases += [({"top_k": value, "sample_rate": 32000}, "between 1 and 4") for value in (0, 5, 1.5)]
        for options, message in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                hybrid.predict_waveform(np.ones(16000), self.artifact, **options)
        self.deep.prepare_waveform.assert_not_called()
        self.deep.predict_waveform.assert_not_called()
        self.spectral.predict_probabilities.assert_not_called()

    def test_non_normalized_or_non_finite_expert_outputs_are_rejected(self):
        for probability in (np.full(4, 0.1), np.array([0.1, 0.1, np.nan, 0.7])):
            with self.subTest(probability=probability):
                self.spectral.predict_probabilities.return_value = probability
                with self.assertRaisesRegex(ValueError, "invalid probabilities"):
                    hybrid.predict_waveform(np.ones(16000), self.artifact, sample_rate=32000)

    def test_validation_checks_child_experts_class_alignment_format_and_fixed_weight(self):
        hybrid.validate_artifact(self.artifact)
        self.deep.validate_artifact.assert_called_once_with(self.artifact["neural_model"])
        self.spectral.validate_expert.assert_called_once_with(self.artifact["spectral_expert"])
        for invalid in ("format", "classes", "expert_classes", "features", "weight"):
            artifact = make_artifact()
            if invalid == "format":
                artifact["backend"] = "other"
            elif invalid == "classes":
                artifact["classes"] = list("0132")
            elif invalid == "expert_classes":
                artifact["spectral_expert"]["classes"] = list("0132")
            elif invalid == "features":
                artifact["feature_config"] = {**artifact["feature_config"], "pre_peak_seconds": 0.04}
            else:
                artifact["neural_weight"] = 0.3
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                hybrid.validate_artifact(artifact)
        self.deep.validate_artifact.side_effect = ValueError("Corrupt neural child")
        with self.assertRaisesRegex(ValueError, "Corrupt neural child"):
            hybrid.validate_artifact(self.artifact)

    def test_live_warms_embedded_neural_model_before_microphone_setup(self):
        spec = importlib.util.spec_from_file_location("hybrid_live_fixture", Path(__file__).with_name("5_live_predict.py"))
        live = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(live)
        order = []
        self.deep.warmup.side_effect = lambda child: order.append(("warmup", child))

        def input_settings(*args):
            order.append(("input_setup", args))
            return None, {"name": "Fixture LCS microphone"}, 48000

        with patch.object(live, "load_model", return_value=self.artifact), \
                patch.object(live, "input_settings", side_effect=input_settings), \
                patch.object(live, "microphone_blocks", return_value=(item for item in ())), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = live.main(["--duration", "1"])
        self.assertEqual(result, 0)
        self.assertEqual([item[0] for item in order], ["warmup", "input_setup"])
        self.assertIs(order[0][1], self.artifact["neural_model"])


class HybridTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset, self.output = self.root / "dataset", self.root / "work"
        self.output.mkdir()
        self.samples, self.metadata = [], []
        for label in "0123":
            path = self.dataset / label / "1.mp3"
            path.parent.mkdir(parents=True)
            path.write_bytes(f"training:{label}".encode())
            self.samples.append(SimpleNamespace(path=path, label=label, ordinal=1))
            self.metadata.append({"file": f"{label}/1.mp3", "key": label, "ordinal": 1,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "split": "training"})
        self.neural = make_artifact()["neural_model"]
        self.neural["training_metadata"]["dataset_fingerprint"] = fingerprint(self.metadata)
        self.provenance = {"source_recordings": ["training sources"]}
        self.expert = make_artifact()["spectral_expert"]
        self.deep = SimpleNamespace(
            DeepConfig=FixtureDeepConfig,
            prepare_waveform=Mock(side_effect=lambda waveform, config: np.full(10240, waveform[0], np.float32)),
            validate_artifact=Mock(),
        )
        self.deep_training = SimpleNamespace(
            train_deep=Mock(), _sample_metadata=Mock(return_value=self.metadata),
            fingerprint=fingerprint, _archive_previous_run=Mock(return_value=None),
        )
        self.spectral = SimpleNamespace(fit_expert=Mock(return_value=self.expert), validate_expert=Mock())
        self.enterContext(patch.dict(sys.modules, {
            "deep_keypress": self.deep, "deep_training": self.deep_training, "spectral_expert": self.spectral,
        }))
        self.enterContext(patch("keypress_model.decode_audio", side_effect=lambda path, rate: np.array([int(path.parent.name)], np.float32)))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_matching_completed_neural_weights_are_reused_and_every_clip_fits_spectral_expert(self):
        joblib.dump(self.neural, self.output / "keypress_model.joblib")
        report = training.train_hybrid(self.dataset, self.output, samples=self.samples, provenance=self.provenance)
        self.deep_training.train_deep.assert_not_called()
        self.spectral.fit_expert.assert_called_once()
        waveforms, labels = self.spectral.fit_expert.call_args.args
        self.assertEqual(waveforms.shape, (4, 10240))
        np.testing.assert_array_equal(waveforms[:, 0], [0, 1, 2, 3])
        np.testing.assert_array_equal(labels, list("0123"))
        self.assertEqual(self.deep.prepare_waveform.call_count, len(self.samples))
        self.assertEqual(report["training_sample_count"], 4)
        self.assertEqual(report["withheld_sample_count"], 0)
        self.assertTrue(report["benchmark_used_for_development"])
        self.assertIs(report["provenance"], self.provenance)
        restored = joblib.load(self.output / "keypress_model.joblib")
        self.assertEqual(restored["backend"], "neural_spectral_hybrid")
        self.assertEqual(restored["neural_model"]["training_metadata"]["implementation_sha256"],
                         self.neural["training_metadata"]["implementation_sha256"])
        np.testing.assert_array_equal(restored["neural_model"]["members"][0]["state_dict"]["head.weight"],
                                      self.neural["members"][0]["state_dict"]["head.weight"])

    def test_missing_neural_model_trains_with_all_samples_and_requested_options(self):
        def train_neural(dataset, output, **options):
            output.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.neural, output / "keypress_model.joblib")

        self.deep_training.train_deep.side_effect = train_neural
        training.train_hybrid(self.dataset, self.output, samples=self.samples, provenance=self.provenance,
                              epochs=90, batch_size=2, device="cpu")
        self.deep_training.train_deep.assert_called_once_with(
            self.dataset, self.output / "neural_training", samples=self.samples,
            provenance=self.provenance, epochs=90, batch_size=2, device="cpu",
        )
        self.spectral.fit_expert.assert_called_once()

    def test_failed_spectral_fit_preserves_previously_deployed_artifact_and_reports(self):
        joblib.dump(self.neural, self.output / "keypress_model.joblib")
        (self.output / "training.json").write_bytes(b"previous training report")
        (self.output / "summary.json").write_bytes(b"previous summary")
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.spectral.fit_expert.side_effect = ValueError("Spectral fit failed")
        with self.assertRaisesRegex(ValueError, "Spectral fit failed"):
            training.train_hybrid(self.dataset, self.output, samples=self.samples, provenance=self.provenance)
        after = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.assertEqual(after, before)
        self.deep_training.train_deep.assert_not_called()
        self.deep_training._archive_previous_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
