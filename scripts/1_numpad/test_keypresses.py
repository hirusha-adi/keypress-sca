"""Safety checks for separating presses without trimming their release or decay."""

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile


SPEC = importlib.util.spec_from_file_location("numpad_keypresses", Path(__file__).with_name("1_keypresses.py"))
splitter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = splitter
SPEC.loader.exec_module(splitter)


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 48000
        rng = np.random.default_rng(42)
        mono = rng.normal(0, 0.0002, round(3.4 * self.sample_rate))
        self.times = (0.4, 1.19, 2.53)
        for time, amplitude in zip(self.times, (0.4, 0.3, 0.5)):
            # Two distinct transients per press, followed by low-level ringing.
            for offset, strength in ((0, amplitude), (0.09, amplitude * 0.65)):
                start = round((time + offset) * self.sample_rate)
                t = np.arange(round(0.13 * self.sample_rate)) / self.sample_rate
                transient = strength * np.exp(-t / 0.015) * np.sin(2 * np.pi * 1700 * t)
                mono[start:start + len(t)] += transient
        # A soft handling click must not become another training example.
        start = round(1.9 * self.sample_rate)
        mono[start:start + 100] += 0.005
        # Opposite-phase stereo catches detectors that cancel the two channels.
        self.samples = np.column_stack((mono, -mono)).astype(np.float32)

    def test_press_release_pairs_and_irregular_gaps_preserve_all_audio(self):
        recording = splitter.analyze(Path("synthetic.m4a"), self.samples, self.sample_rate, {3})
        self.assertEqual(len(recording.peaks), 3)
        pieces = []
        for time, start, end in zip(self.times, recording.cuts[:-1], recording.cuts[1:]):
            self.assertLess(start / self.sample_rate, time - 0.05)
            self.assertGreater(end / self.sample_rate, time + 0.22)
            pieces.append(self.samples[start:end])
        np.testing.assert_array_equal(np.concatenate(pieces), self.samples)

    def test_detection_adapts_to_recording_level(self):
        for gain in (0.01, 5.0):
            with self.subTest(gain=gain):
                recording = splitter.analyze(
                    Path("synthetic.m4a"), self.samples * gain, self.sample_rate, {3}
                )
                self.assertEqual(len(recording.peaks), 3)

    def test_wrong_count_is_rejected_instead_of_forced(self):
        with self.assertRaisesRegex(ValueError, "detected 3 presses; expected 25, 26"):
            splitter.analyze(Path("synthetic.m4a"), self.samples, self.sample_rate, {25, 26})

    def test_silence_and_background_noise_are_rejected(self):
        rng = np.random.default_rng(13)
        for samples in (np.zeros_like(self.samples), rng.normal(0, 0.001, self.samples.shape)):
            with self.assertRaisesRegex(ValueError, "detected 0 presses"):
                splitter.analyze(Path("noise.m4a"), samples, self.sample_rate, {25, 26})

    def test_unsafe_gap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "refusing to cut through sound"):
            splitter.find_quiet_cut(np.full(300, 0.05), 50, 200, 0.005, 0.001)

    def test_presses_without_enough_room_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "too close"):
            splitter.find_quiet_cut(np.zeros(300), 50, 100, 0.005, 0.001)

    def test_sequence_names_use_press_number_then_actual_key(self):
        self.assertEqual(
            splitter.sequence_clip_names("0789123456"),
            ["1-0.mp3", "2-7.mp3", "3-8.mp3", "4-9.mp3", "5-1.mp3",
             "6-2.mp3", "7-3.mp3", "8-4.mp3", "9-5.mp3", "10-6.mp3"],
        )
        self.assertEqual(
            splitter.sequence_clip_names("0070"),
            ["1-0.mp3", "2-0.mp3", "3-7.mp3", "4-0.mp3"],
        )
        for sequence in ("", "test", "07-0", "０７０"):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                splitter.sequence_clip_names(sequence)

    def test_sequence_count_must_match_filename(self):
        recording = splitter.analyze_sequence(Path("070.m4a"), self.samples, self.sample_rate)
        self.assertEqual(recording.clip_names, ["1-0.mp3", "2-7.mp3", "3-0.mp3"])
        with self.assertRaisesRegex(ValueError, "detected 3 presses; expected 4"):
            splitter.analyze_sequence(Path("0700.m4a"), self.samples, self.sample_rate)

    def test_existing_sequence_clips_require_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            recording = splitter.analyze_sequence(dataset / "070.m4a", self.samples, self.sample_rate)
            directory = dataset / "070"
            directory.mkdir()
            (directory / "0-1.mp3").write_bytes(b"existing clip")
            (directory / "1-0.mp3").write_bytes(b"existing new-format clip")
            with self.assertRaisesRegex(ValueError, "already contains clips"):
                splitter.check_outputs(dataset, [recording], overwrite=False)
            splitter.check_outputs(dataset, [recording], overwrite=True)
            self.assertEqual((directory / "0-1.mp3").read_bytes(), b"existing clip")
            self.assertEqual((directory / "1-0.mp3").read_bytes(), b"existing new-format clip")

    def test_tests_only_export_preserves_training_and_removes_only_stale_generated_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training"
            training.mkdir()
            sentinel = training / splitter.REPORT_NAME
            sentinel.write_bytes(b"training data stays unchanged")
            dataset = root / "tests"
            dataset.mkdir()
            source = dataset / "070.m4a"
            # FFmpeg detects the WAV header regardless of the .m4a suffix; this
            # lossless fixture isolates filename handling from AAC artifacts.
            wavfile.write(source, self.sample_rate, self.samples)
            source_bytes = source.read_bytes()
            arguments = ["--only", "tests", "--dataset-dir", str(training), "--tests-dir", str(dataset)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(splitter.main(arguments + ["--dry-run"]), 0)
                self.assertFalse((dataset / splitter.REPORT_NAME).exists())
                self.assertFalse((dataset / "070").exists())
                self.assertEqual(splitter.main(arguments), 0)
            directory = dataset / "070"
            self.assertEqual({path.name for path in directory.glob("*.mp3")}, {"1-0.mp3", "2-7.mp3", "3-0.mp3"})
            manifest = json.loads((dataset / splitter.REPORT_NAME).read_text())
            recording = manifest["recordings"][0]
            self.assertEqual(recording["key_sequence"], "070")
            self.assertEqual([clip["key"] for clip in recording["clips"]], list("070"))
            self.assertEqual([clip["press_number"] for clip in recording["clips"]], [1, 2, 3])
            self.assertEqual([clip["file"] for clip in recording["clips"]],
                             ["070/1-0.mp3", "070/2-7.mp3", "070/3-0.mp3"])
            self.assertEqual(recording["clips"][0]["start_sample"], 0)
            self.assertEqual(recording["clips"][-1]["end_sample_exclusive"], len(self.samples))
            (directory / "0-99.mp3").write_bytes(b"stale generated clip")
            (directory / "99-0.mp3").write_bytes(b"stale new-format clip")
            (directory / "notes.txt").write_bytes(b"keep notes")
            (directory / "other.mp3").write_bytes(b"keep unrelated audio")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(splitter.main(arguments + ["--overwrite"]), 0)
            self.assertFalse((directory / "0-99.mp3").exists())
            self.assertFalse((directory / "99-0.mp3").exists())
            self.assertEqual((directory / "notes.txt").read_bytes(), b"keep notes")
            self.assertEqual((directory / "other.mp3").read_bytes(), b"keep unrelated audio")
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual(sentinel.read_bytes(), b"training data stays unchanged")

    def test_migration_replaces_overlapping_legacy_names_and_removes_old_clips(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            source = dataset / "210.m4a"
            wavfile.write(source, self.sample_rate, self.samples)
            directory = dataset / "210"
            directory.mkdir()
            for name in ("2-1.mp3", "1-1.mp3", "0-1.mp3"):
                (directory / name).write_bytes(b"old-format audio")
            with contextlib.redirect_stdout(io.StringIO()):
                result = splitter.main(["--only", "tests", "--tests-dir", str(dataset), "--overwrite"])
            self.assertEqual(result, 0)
            # Legacy 2-1 was the first press/key 2; new 2-1 is the second press/key 1.
            self.assertEqual({path.name for path in directory.glob("*.mp3")}, {"1-2.mp3", "2-1.mp3", "3-0.mp3"})
            self.assertNotEqual((directory / "2-1.mp3").read_bytes(), b"old-format audio")

    def test_sequence_count_failure_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            wavfile.write(dataset / "0700.m4a", self.sample_rate, self.samples)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = splitter.main(["--only", "tests", "--tests-dir", str(dataset)])
            self.assertEqual(result, 1)
            self.assertFalse((dataset / "0700").exists())
            self.assertFalse((dataset / splitter.REPORT_NAME).exists())


if __name__ == "__main__":
    unittest.main()
