"""Safety checks for separating presses without trimming their release or decay."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
