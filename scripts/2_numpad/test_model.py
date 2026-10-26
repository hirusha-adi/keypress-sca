"""Feature and inference checks independent of classifier training accuracy."""

from dataclasses import asdict
import builtins
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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

    def test_background_bumps_do_not_become_additional_keypresses(self):
        clean = np.pad(self.waveform, (0, self.config.sample_rate))
        times = np.arange(len(clean)) / self.config.sample_rate
        carrier = np.sin(2 * np.pi * 1800 * times)
        broad = .12 * np.exp(-.5 * ((times - 1.15) / .08) ** 2) * carrier
        plateau = .04 * ((times >= .95) & (times <= 1.3))
        low_contrast = (plateau + .06 * np.exp(-.5 * ((times - 1.1) / .008) ** 2)) * carrier
        reference = model.aligned_segment(clean, self.config)
        for name, bump in (("broad", broad), ("low_contrast", low_contrast)):
            with self.subTest(background=name):
                waveform = clean + bump
                # The added background is far from the true press and leaves
                # both the selected event and its feature window unchanged.
                np.testing.assert_allclose(model.aligned_segment(waveform, self.config), reference,
                                           rtol=1e-7, atol=1e-7)

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


class NeuralModelIntegrationTests(unittest.TestCase):
    def make_artifact(self):
        config = model.FeatureConfig(
            mode="logmel_patch", sample_rate=32000, duration_seconds=0.32,
            pre_peak_seconds=0.04, n_fft=1024, hop_length=128,
            n_mels=224, n_mfcc=40, fmax=15500,
        )
        return {
            "schema_version": model.NEURAL_MODEL_SCHEMA_VERSION,
            "backend": model.NEURAL_BACKEND,
            "classes": list("0123"), "feature_config": asdict(config),
            "members": [{"architecture": "swin_t", "seed": 42,
                         "state_dict": {"head.weight": np.ones((4, 3), dtype=np.float16)}}],
            "training_metadata": {"training_sample_count": 706, "fit_split": "all_clips"},
        }

    def test_neural_artifact_round_trip_validates_numpy_weights_without_a_pipeline(self):
        artifact = self.make_artifact()
        validate = Mock()
        backend = SimpleNamespace(validate_artifact=validate)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "neural.joblib"
            joblib.dump(artifact, path)
            with patch.dict(sys.modules, {"deep_keypress": backend}):
                restored = model.load_model(path)
        validate.assert_called_once()
        self.assertIs(validate.call_args.args[0], restored)
        self.assertNotIn("pipeline", restored)
        self.assertEqual(restored["classes"], list("0123"))
        self.assertEqual(restored["training_metadata"]["training_sample_count"], 706)
        np.testing.assert_array_equal(restored["members"][0]["state_dict"]["head.weight"],
                                      artifact["members"][0]["state_dict"]["head.weight"])
        self.assertEqual(restored["members"][0]["state_dict"]["head.weight"].dtype, np.float16)

    def test_neural_waveform_dispatch_preserves_native_sample_rate_and_result(self):
        artifact = self.make_artifact()
        waveform = synthetic_press(48000)
        prediction = {"predicted_key": "2", "confidence": 0.7,
                      "ranked_keys": [{"key": "2", "score": 0.7, "probability": 0.7}]}
        predict = Mock(return_value=prediction)
        backend = SimpleNamespace(predict_waveform=predict)
        with patch.dict(sys.modules, {"deep_keypress": backend}):
            result = model.predict_waveform(waveform, artifact, top_k=2, sample_rate=48000)
        self.assertIs(result, prediction)
        self.assertIs(predict.call_args.args[0], waveform)
        self.assertIs(predict.call_args.args[1], artifact)
        self.assertEqual(predict.call_args.kwargs, {"top_k": 2, "sample_rate": 48000})

    def make_hybrid_artifact(self):
        neural = self.make_artifact()
        neural["feature_config"]["pre_peak_seconds"] = 0.08
        return {
            "schema_version": model.NEURAL_MODEL_SCHEMA_VERSION,
            "backend": model.HYBRID_BACKEND,
            "classes": list("0123"), "feature_config": dict(neural["feature_config"]),
            "neural_model": neural,
            "spectral_expert": SimpleNamespace(classes_=np.array(list("0123"))),
            "neural_weight": 0.2, "spectral_weight": 0.8,
            "training_metadata": {"training_sample_count": 706, "fit_split": "all_clips"},
        }

    def test_hybrid_artifact_loads_with_its_own_validator_and_embedded_neural_model(self):
        artifact = self.make_hybrid_artifact()
        validate = Mock()
        backend = SimpleNamespace(validate_artifact=validate)
        forbidden_deep = SimpleNamespace(validate_artifact=Mock(side_effect=AssertionError("Wrong backend validator")))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hybrid.joblib"
            joblib.dump(artifact, path)
            with patch.dict(sys.modules, {"hybrid_keypress": backend, "deep_keypress": forbidden_deep}):
                restored = model.load_model(path)
        validate.assert_called_once()
        self.assertIs(validate.call_args.args[0], restored)
        self.assertEqual(restored["backend"], model.HYBRID_BACKEND)
        self.assertEqual(restored["neural_model"]["backend"], model.NEURAL_BACKEND)
        self.assertEqual(restored["training_metadata"]["training_sample_count"], 706)
        np.testing.assert_array_equal(restored["neural_model"]["members"][0]["state_dict"]["head.weight"],
                                      artifact["neural_model"]["members"][0]["state_dict"]["head.weight"])

    def test_hybrid_prediction_dispatches_native_audio_without_reading_a_pipeline(self):
        artifact = self.make_hybrid_artifact()
        waveform = synthetic_press(48000)
        prediction = {"predicted_key": "3", "confidence": 0.85,
                      "ranked_keys": [{"key": "3", "score": -0.1625, "probability": 0.85}]}
        predict = Mock(return_value=prediction)
        backend = SimpleNamespace(predict_waveform=predict)
        forbidden_deep = SimpleNamespace(predict_waveform=Mock(side_effect=AssertionError("Wrong prediction backend")))
        with patch.dict(sys.modules, {"hybrid_keypress": backend, "deep_keypress": forbidden_deep}):
            result = model.predict_waveform(waveform, artifact, top_k=3, sample_rate=48000)
        self.assertIs(result, prediction)
        self.assertIs(predict.call_args.args[0], waveform)
        self.assertIs(predict.call_args.args[1], artifact)
        self.assertEqual(predict.call_args.kwargs, {"top_k": 3, "sample_rate": 48000})

    def test_neural_file_prediction_decodes_at_saved_analysis_rate(self):
        artifact = self.make_artifact()
        waveform = synthetic_press(32000)
        path = Path("fixture/press.wav")
        prediction = {"predicted_key": "1", "confidence": 0.8, "ranked_keys": []}
        backend = SimpleNamespace(predict_waveform=Mock(return_value=prediction))
        with patch.dict(sys.modules, {"deep_keypress": backend}), \
                patch.object(model, "decode_audio", return_value=waveform) as decode:
            result = model.predict_key(path, artifact, top_k=3)
        decode.assert_called_once_with(path, 32000)
        self.assertEqual(result, {"audio_file": str(path.resolve()), **prediction})
        self.assertEqual(backend.predict_waveform.call_args.kwargs, {"top_k": 3, "sample_rate": None})

    def test_neural_validator_errors_and_unsupported_backends_are_not_silenced(self):
        for unsupported in (False, True):
            with self.subTest(unsupported=unsupported), tempfile.TemporaryDirectory() as temporary:
                artifact = self.make_artifact()
                if unsupported:
                    artifact["backend"] = "unknown"
                path = Path(temporary) / "invalid.joblib"
                joblib.dump(artifact, path)
                backend = SimpleNamespace(validate_artifact=Mock(side_effect=ValueError("Invalid neural weights")))
                message = "Unsupported neural" if unsupported else "Invalid neural weights"
                with patch.dict(sys.modules, {"deep_keypress": backend}), \
                        self.assertRaisesRegex(ValueError, message):
                    model.load_model(path)
                self.assertEqual(backend.validate_artifact.call_count, 0 if unsupported else 1)

    def test_legacy_artifact_loading_and_prediction_do_not_import_neural_dependencies(self):
        config = model.FeatureConfig()
        rng = np.random.default_rng(343)
        pipeline = model.build_classifier(kind="linear")
        pipeline.fit(rng.normal(size=(16, 40)), np.repeat(list("0123"), 4))
        artifact = {"schema_version": model.MODEL_SCHEMA_VERSION, "pipeline": pipeline,
                    "classes": list("0123"), "feature_config": asdict(config)}
        original_import = builtins.__import__

        def forbid_neural_dependencies(name, *args, **kwargs):
            if name.split(".")[0] in {"deep_keypress", "hybrid_keypress", "torch", "torchvision"}:
                raise AssertionError("Legacy model imported neural dependencies")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.joblib"
            joblib.dump(artifact, path)
            with patch("builtins.__import__", side_effect=forbid_neural_dependencies):
                restored = model.load_model(path)
                result = model.predict_waveform(synthetic_press(), restored)
        self.assertIn(result["predicted_key"], list("0123"))


class ModelPersistenceTests(unittest.TestCase):
    def test_four_class_models_preserve_predictions_and_validate_top_k(self):
        config = model.FeatureConfig()
        rng = np.random.default_rng(79)
        features = rng.normal(size=(32, 40))
        labels = np.repeat(list("0123"), 8)
        waveform = synthetic_press()
        for kind in ("rbf", "linear", "lda"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                pipeline = model.build_classifier(0.01, kind=kind)
                pipeline.fit(features, labels)
                artifact = {"schema_version": model.MODEL_SCHEMA_VERSION, "pipeline": pipeline,
                            "feature_config": asdict(config), "classes": list("0123")}
                path = Path(temporary) / "four_keys.joblib"
                joblib.dump(artifact, path)
                restored = model.load_model(path)
                before = model.predict_waveform(waveform, artifact, top_k=4)
                after = model.predict_waveform(waveform, restored, top_k=4)
                self.assertEqual(before, after)
                self.assertEqual(set(row["key"] for row in after["ranked_keys"]), set("0123"))
                self.assertEqual(after["predicted_key"], after["ranked_keys"][0]["key"])
                self.assertNotIn("accuracy", after)
                for top_k in (0, 5, 1.5):
                    with self.subTest(top_k=top_k), self.assertRaisesRegex(ValueError, "between 1 and 4"):
                        model.predict_waveform(waveform, restored, top_k=top_k)
                artifact["classes"] = list("0132")
                joblib.dump(artifact, path)
                with self.assertRaisesRegex(ValueError, "declared classes"):
                    model.load_model(path)

    def test_binary_models_expand_the_second_class_decision_score(self):
        config = model.FeatureConfig()
        rng = np.random.default_rng(83)
        features = rng.normal(size=(20, 40))
        labels = np.repeat(list("03"), 10)
        waveform = synthetic_press()
        for kind in ("rbf", "linear", "lda"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                pipeline = model.build_classifier(0.01, kind=kind)
                pipeline.fit(features, labels)
                artifact = {"schema_version": model.MODEL_SCHEMA_VERSION, "pipeline": pipeline,
                            "feature_config": asdict(config), "classes": list("03")}
                path = Path(temporary) / "two_keys.joblib"
                joblib.dump(artifact, path)
                result = model.predict_waveform(waveform, model.load_model(path), top_k=2)
                self.assertEqual(result["predicted_key"], result["ranked_keys"][0]["key"])
                self.assertEqual(set(row["key"] for row in result["ranked_keys"]), set("03"))
                self.assertAlmostEqual(sum(row["score"] for row in result["ranked_keys"]), 0)

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
