"""Live CLI accounting and replay checks that never access a microphone."""

import builtins
import contextlib
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.io import wavfile


SPEC = importlib.util.spec_from_file_location(
    "numpad_live_predict", Path(__file__).with_name("5_live_predict.py")
)
live = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = live
SPEC.loader.exec_module(live)


def prediction_record(**changes):
    """A wrong, highly confident prediction with independently measured accuracy."""
    record = {
        "event_index": 4,
        "predicted_key": "7",
        "elapsed_seconds": 1.25,
        "timestamp": 0.0,
        "accuracy": 0.75,
        "correct_count": 3,
        "compared_count": 4,
        "actual_key": "8",
        "confidence": 0.992,
        "ranked_keys": [
            {"key": "7", "score": 3.0, "probability": 0.992},
            {"key": "1", "score": 2.0, "probability": 0.006},
            {"key": "8", "score": 1.0, "probability": 0.002},
        ],
        "snr_db": 32.5,
        "peak_dbfs": -6.2,
        "latency_ms": 501.0,
        "inference_ms": 2.4,
        "quality": [],
        "status": "predicted",
    }
    record.update(changes)
    return record


class AccuracyTrackingTests(unittest.TestCase):
    def test_unlabelled_predictions_do_not_claim_measured_accuracy(self):
        tracker = live.PredictionTracker()
        for index, key in enumerate(("7", "2", None), 1):
            result = tracker.record(key)
            self.assertEqual(result["index"], index)
            self.assertIsNone(result["actual_key"])
            self.assertIsNone(result["correct"])
            self.assertIsNone(result["accuracy"])
            self.assertEqual(result["compared_count"], 0)
        self.assertIsNone(tracker.summary()["accuracy"])

    def test_accuracy_counts_wrong_rejected_and_extra_detections(self):
        tracker = live.PredictionTracker("078")
        first = tracker.record("0")
        self.assertEqual(first["actual_key"], "0")
        self.assertTrue(first["correct"])
        self.assertEqual(first["accuracy"], 1.0)
        second = tracker.record("2")
        self.assertEqual(second["actual_key"], "7")
        self.assertFalse(second["correct"])
        self.assertEqual(second["accuracy"], 0.5)
        rejected = tracker.record(None)
        self.assertEqual(rejected["actual_key"], "8")
        self.assertFalse(rejected["correct"])
        self.assertAlmostEqual(rejected["accuracy"], 1 / 3)
        extra = tracker.record("8")
        self.assertIsNone(extra["actual_key"])
        self.assertFalse(extra["correct"])
        self.assertEqual(extra["correct_count"], 1)
        self.assertEqual(extra["compared_count"], 4)
        self.assertEqual(extra["accuracy"], 0.25)
        self.assertEqual(tracker.summary()["accuracy"], 0.25)

    def test_final_accuracy_penalizes_missed_expected_presses(self):
        tracker = live.PredictionTracker("0123")
        self.assertEqual(tracker.record("0")["accuracy"], 1.0)
        self.assertEqual(tracker.record("1")["accuracy"], 1.0)
        summary = tracker.summary()
        self.assertEqual(summary["correct_count"], 2)
        self.assertEqual(summary["compared_count"], 4)
        self.assertEqual(summary["accuracy"], 0.5)

    def test_known_sequence_with_no_detections_has_zero_final_accuracy(self):
        summary = live.PredictionTracker("07").summary()
        self.assertEqual(summary["correct_count"], 0)
        self.assertEqual(summary["compared_count"], 2)
        self.assertEqual(summary["accuracy"], 0.0)

    def test_expected_labels_must_be_nonempty_ascii_digits(self):
        for invalid in ("", "07a", "0 7", "07\n", "１２", "٠٧"):
            with self.subTest(sequence=invalid), self.assertRaises(ValueError):
                live.PredictionTracker(invalid)


class PredictionFormattingTests(unittest.TestCase):
    def test_line_has_event_time_key_true_accuracy_and_separate_confidence(self):
        line = live.format_prediction(prediction_record())
        self.assertRegex(line, r"^00:00:01\.250: 7 \(")
        self.assertRegex(line, r"\bacc=75(?:\.0)?%")
        self.assertRegex(line, r"\bconf=99\.2%")
        self.assertIn("top3=", line)
        self.assertLess(line.index("7:", line.index("top3=")), line.index("1:", line.index("top3=")))
        self.assertLess(line.index("1:", line.index("top3=")), line.index("8:", line.index("top3=")))
        self.assertRegex(line, r"\bsnr=32\.5\s*dB")
        self.assertRegex(line, r"\bpeak=-6\.2\s*dBFS")
        self.assertRegex(line, r"\blatency=501(?:\.0)?\s*ms")
        self.assertRegex(line, r"\binfer=2\.4\s*ms")
        self.assertIn("n=4", line)

    def test_accuracy_stays_unavailable_even_when_confidence_is_high(self):
        line = live.format_prediction(prediction_record(
            accuracy=None, actual_key=None, correct_count=0, compared_count=0
        ))
        self.assertIn("acc=n/a", line)
        self.assertIn("conf=99.2%", line)

    def test_decision_scores_do_not_become_probabilities(self):
        line = live.format_prediction(prediction_record(
            confidence=None,
            ranked_keys=[{"key": "7", "score": 42.5, "probability": None}],
        ))
        self.assertIn("conf=n/a", line)
        self.assertNotIn("42.5%", line)

    def test_wall_time_uses_requested_timezone(self):
        line = live.format_prediction(prediction_record(), time_format="wall", timezone_name="UTC")
        self.assertRegex(line, r"^00:00:00\.000: 7 \(")

    def test_rejected_event_remains_visible_and_keeps_its_event_number(self):
        line = live.format_prediction(prediction_record(
            predicted_key=None, confidence=None, ranked_keys=[],
            status="rejected", quality=["insufficient_context"],
        ))
        self.assertIn("n=4", line)
        self.assertIn("acc=75", line)
        self.assertIn("insufficient_context", line)


class MicrophoneIsolationTests(unittest.TestCase):
    def test_default_microphone_uses_native_rate_and_overflow_stops_with_failure(self):
        class CallbackAbort(Exception):
            pass

        class PortAudioError(Exception):
            pass

        class FixtureStream:
            active = True
            closed = False

            def __init__(self, **settings):
                self.settings = settings

            def __enter__(self):
                samples = np.zeros((self.settings["blocksize"], 1), dtype=np.float32)
                timing = SimpleNamespace(inputBufferAdcTime=50.0, currentTime=50.01)
                callback = self.settings["callback"]
                callback(samples, len(samples), timing, False)
                try:
                    callback(samples, len(samples), timing, "input overflow")
                except CallbackAbort:
                    self.active = False
                    self.settings["finished_callback"]()
                return self

            def __exit__(self, *exception):
                self.closed = True
                return False

        streams = []

        def make_stream(**settings):
            stream = FixtureStream(**settings)
            streams.append(stream)
            return stream

        backend = SimpleNamespace(
            query_devices=Mock(return_value={
                "name": "LCS USB Audio Mono", "max_input_channels": 1,
                "default_samplerate": 48000.0,
            }),
            check_input_settings=Mock(),
            InputStream=Mock(side_effect=make_stream),
            CallbackAbort=CallbackAbort, PortAudioError=PortAudioError,
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(live, "sounddevice_module", return_value=backend), \
                patch.object(live, "load_model", return_value={
                    "feature_config": asdict(live.FeatureConfig())}), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = live.main(["--expected-sequence", "0", "--json"])
        self.assertEqual(result, 1)
        backend.query_devices.assert_called_once_with("LCS USB Audio Mono", kind="input")
        backend.check_input_settings.assert_called_once_with(
            device="LCS USB Audio Mono", channels=1, samplerate=48000, dtype="float32"
        )
        stream_settings = backend.InputStream.call_args.kwargs
        self.assertEqual(stream_settings["device"], "LCS USB Audio Mono")
        self.assertEqual(stream_settings["samplerate"], 48000)
        self.assertEqual(stream_settings["blocksize"], 480)
        self.assertTrue(streams[0].closed)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("input overflow", stderr.getvalue())
        self.assertIn("dropped audio invalidates event timing and sequence accuracy", stderr.getvalue())

    def test_listing_devices_queries_hardware_without_starting_a_stream(self):
        backend = SimpleNamespace(
            query_devices=Mock(return_value=[{
                "name": "Fixture microphone", "max_input_channels": 1,
                "max_output_channels": 0, "default_samplerate": 48000.0,
                "hostapi": 0, "index": 0,
            }]),
            query_hostapis=Mock(return_value=[{"name": "Fixture host"}]),
            InputStream=Mock(side_effect=AssertionError("Device listing opened a microphone")),
            default=SimpleNamespace(device=(0, 0)),
        )
        stdout = io.StringIO()
        with patch.dict(sys.modules, {"sounddevice": backend}), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            result = live.main(["--list-devices"])
        self.assertIn(result, (None, 0))
        backend.query_devices.assert_called()
        backend.InputStream.assert_not_called()
        self.assertIn("Fixture microphone", stdout.getvalue())

    def test_replay_uses_detection_and_prediction_without_importing_microphone_backend(self):
        sample_rate = 24000
        rng = np.random.default_rng(123)
        waveform = rng.normal(0, 0.00003, sample_rate * 3).astype(np.float32)
        t = np.arange(round(0.1 * sample_rate)) / sample_rate
        sound = 0.5 * np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.015)
        waveform[2 * sample_rate:2 * sample_rate + len(sound)] += sound.astype(np.float32)
        artifact = {"feature_config": asdict(live.FeatureConfig())}
        prediction = {
            "predicted_key": "7", "confidence": 0.91,
            "confidence_type": "Uncalibrated classifier probability",
            "ranked_keys": [{"key": "7", "score": 3.0, "probability": 0.91}],
        }
        original_import = builtins.__import__

        def forbid_microphone_import(name, *args, **kwargs):
            if name == "sounddevice" or name.startswith("sounddevice."):
                raise AssertionError("Offline replay imported the microphone backend")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            replay = Path(temporary) / "fixture.wav"
            model = Path(temporary) / "fixture.joblib"
            wavfile.write(replay, sample_rate, waveform)
            model.write_bytes(b"fixture model; loading is mocked")
            stdout = io.StringIO()
            with patch.object(live, "load_model", return_value=artifact), \
                    patch.object(live, "predict_waveform", return_value=prediction) as predict, \
                    patch("builtins.__import__", side_effect=forbid_microphone_import), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                result = live.main([
                    "--replay", str(replay), "--model", str(model),
                    "--expected-sequence", "7", "--json",
                ])
        self.assertIn(result, (None, 0))
        predict.assert_called_once()
        records = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        events = [record for record in records if "event_index" in record]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["predicted_key"], "7")
        self.assertEqual(events[0]["actual_key"], "7")
        self.assertEqual(events[0]["accuracy"], 1.0)
        self.assertEqual(events[0]["confidence"], 0.91)
        self.assertAlmostEqual(events[0]["elapsed_seconds"], 2.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
