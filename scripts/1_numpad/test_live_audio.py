"""Streaming detector checks independent of microphone hardware."""

import unittest

import numpy as np

from live_audio import DetectorConfig, KeystrokeDetector


def recording(times=(0.4, 1.2, 2.0), seconds=2.8, sample_rate=24000, release_gain=0.55):
    rng = np.random.default_rng(682)
    samples = rng.normal(0, 0.0002, round(seconds * sample_rate))
    for onset in times:
        for offset, strength in ((0, 0.50), (0.09, 0.50 * release_gain)):
            start = round((onset + offset) * sample_rate)
            if start >= len(samples):
                continue
            length = min(round(0.13 * sample_rate), len(samples) - start)
            t = np.arange(length) / sample_rate
            pulse = strength * np.exp(-t / 0.015) * (
                np.sin(2 * np.pi * 1700 * t) + 0.2 * np.sin(2 * np.pi * 3700 * t)
            )
            samples[start:start + length] += pulse
    return samples.astype(np.float32)


def collect(detector, samples, sizes=(257,)):
    events = []
    position = 0
    counter = 0
    while position < len(samples):
        size = sizes[counter % len(sizes)]
        events.extend(detector.feed(samples[position:position + size]))
        position += size
        counter += 1
    events.extend(detector.flush())
    return events


class StreamingDetectorTests(unittest.TestCase):
    def test_press_release_events_keep_model_context_and_exact_source_audio(self):
        samples = recording()
        detector = KeystrokeDetector(DetectorConfig(calibration_seconds=0.2))
        events = collect(detector, samples)
        self.assertEqual(len(events), 3)
        for onset, event in zip((0.4, 1.2, 2.0), events):
            self.assertAlmostEqual(event.peak_sample / 24000, onset, delta=0.005)
            self.assertLessEqual(event.start_sample, event.peak_sample - round(0.04 * 24000))
            self.assertGreaterEqual(event.end_sample, event.peak_sample + round(0.28 * 24000))
            self.assertLess(event.start_sample, event.onset_sample - round(0.1 * 24000))
            self.assertEqual(event.quality, ())
            np.testing.assert_array_equal(event.waveform, samples[event.start_sample:event.end_sample])
        self.assertLess(detector.noise_rms, 0.0004)

    def test_chunk_boundaries_do_not_change_detection_or_timestamps(self):
        samples = recording()
        reference = collect(KeystrokeDetector(), samples, (len(samples),))
        for sizes in ((1, 119, 587, 3, 1111), (128,), (8192,)):
            with self.subTest(sizes=sizes):
                events = collect(KeystrokeDetector(), samples, sizes)
                self.assertEqual(
                    [(event.start_sample, event.onset_sample, event.peak_sample, event.end_sample)
                     for event in events],
                    [(event.start_sample, event.onset_sample, event.peak_sample, event.end_sample)
                     for event in reference],
                )
                for actual, expected in zip(events, reference):
                    np.testing.assert_array_equal(actual.waveform, expected.waveform)

    def test_calibration_does_not_discard_early_keypresses(self):
        samples = recording(times=(0.4, 1.2), seconds=2.0)
        detector = KeystrokeDetector(DetectorConfig(calibration_seconds=1.5))
        self.assertEqual(detector.feed(samples[:24000]), [])
        self.assertFalse(detector.calibrated)
        events = detector.feed(samples[24000:]) + detector.flush()
        self.assertTrue(detector.calibrated)
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0].peak_sample / 24000, 0.4, delta=0.005)

    def test_stationary_noise_and_changing_quiet_background_do_not_become_keys(self):
        rng = np.random.default_rng(18)
        noise = rng.normal(0, 0.001, 24000 * 3)
        louder = rng.normal(0, 0.0015, 24000 * 3)
        detector = KeystrokeDetector(DetectorConfig(calibration_seconds=0.3))
        self.assertEqual(collect(detector, np.r_[noise, louder]), [])
        self.assertTrue(0.0008 < detector.noise_rms < 0.0016)

    def test_sustained_background_rise_recovers_before_a_later_real_press(self):
        rng = np.random.default_rng(775)
        samples = rng.normal(0, 0.0002, 24000 * 5).astype(np.float32)
        samples[24000:] = rng.normal(0, 0.003, len(samples) - 24000)
        pulse = recording(times=(3.0,), seconds=5.0)
        # Retain only the synthetic signal, rather than its quieter fixture noise.
        samples += pulse - recording(times=(), seconds=5.0)
        detector = KeystrokeDetector(DetectorConfig(calibration_seconds=0.3))
        events = collect(detector, samples, (257, 960, 17))
        clean = [event for event in events if not event.quality]
        self.assertEqual(len(clean), 1)
        self.assertAlmostEqual(clean[0].peak_sample / 24000, 3.0, delta=0.005)
        self.assertTrue(any("stationary_background" in event.quality for event in events))
        self.assertTrue(0.002 < detector.noise_rms < 0.004)
        self.assertLessEqual(len(detector._stationary_rms), 101)

    def test_fixed_threshold_below_sustained_background_fails_explicitly(self):
        rng = np.random.default_rng(776)
        samples = np.r_[rng.normal(0, 0.0002, 24000), rng.normal(0, 0.003, 24000 * 2)]
        detector = KeystrokeDetector(DetectorConfig(calibration_seconds=0.3, threshold_db=-55))
        with self.assertRaisesRegex(ValueError, "background noise exceeds the fixed --threshold-db"):
            collect(detector, samples)

    def test_fixed_threshold_below_initial_background_fails_calibration(self):
        rng = np.random.default_rng(777)
        detector = KeystrokeDetector(DetectorConfig(calibration_seconds=0.3, threshold_db=-55))
        with self.assertRaisesRegex(ValueError, "Background noise exceeds the fixed --threshold-db"):
            detector.feed(rng.normal(0, 0.003, 24000))

    def test_successive_events_never_capture_the_next_press(self):
        for interval in (0.35, 0.40, 0.45):
            with self.subTest(interval=interval):
                times = (0.4, 0.4 + interval)
                samples = recording(times=times, seconds=1.6)
                detector = KeystrokeDetector(DetectorConfig(calibration_seconds=0.2))
                events = collect(detector, samples)
                self.assertEqual(len(events), 2)
                first, second = events
                self.assertLess(first.end_sample, second.onset_sample)
                self.assertGreaterEqual(first.end_sample, first.peak_sample + round(0.28 * 24000))
                self.assertEqual(first.quality, ())
                # Later source audio, especially the second press, is absent.
                np.testing.assert_array_equal(first.waveform, samples[first.start_sample:first.end_sample])

    def test_faster_ambiguous_transients_are_qualified_instead_of_clean_predictions(self):
        samples = recording(times=(0.4, 0.65), seconds=1.4)
        events = collect(KeystrokeDetector(DetectorConfig(calibration_seconds=0.2)), samples)
        self.assertEqual(len(events), 1)
        self.assertIn("closely_spaced_transients", events[0].quality)

    def test_loud_late_release_and_next_press_report_missing_context(self):
        samples = recording(times=(0.4, 0.75), seconds=1.6, release_gain=1.5)
        events = collect(KeystrokeDetector(DetectorConfig(calibration_seconds=0.2)), samples)
        self.assertEqual(len(events), 2)
        self.assertLess(events[0].end_sample, events[1].onset_sample)
        self.assertIn("truncated_context", events[0].quality)

    def test_flush_marks_incomplete_trailing_context_and_is_idempotent(self):
        samples = recording(times=(0.4,), seconds=0.47)
        detector = KeystrokeDetector(DetectorConfig(calibration_seconds=0.2))
        events = collect(detector, samples)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].end_sample, len(samples))
        self.assertIn("truncated_context", events[0].quality)
        self.assertEqual(detector.flush(), [])

    def test_native_sample_rate_keeps_time_and_context_consistent(self):
        detector = KeystrokeDetector(DetectorConfig(sample_rate=48000, calibration_seconds=0.2))
        samples = recording(sample_rate=48000)
        events = collect(detector, samples, (777, 128, 2048))
        self.assertEqual(len(events), 3)
        for expected, event in zip((0.4, 1.2, 2.0), events):
            self.assertAlmostEqual(event.peak_sample / 48000, expected, delta=0.005)
            self.assertGreaterEqual(event.end_sample - event.peak_sample, round(0.28 * 48000))

    def test_reset_recalibrates_and_keeps_the_sample_clock(self):
        detector = KeystrokeDetector(DetectorConfig(calibration_seconds=0.2))
        detector.feed(np.zeros(10003, dtype=np.float32))
        detector.reset(dropped_samples=777)
        self.assertFalse(detector.calibrated)
        self.assertEqual(detector.samples_seen, 10780)
        samples = recording(times=(0.4,), seconds=1.1)
        events = collect(detector, samples)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual((events[0].peak_sample - 10780) / 24000, 0.4, delta=0.005)
        self.assertEqual(detector.samples_seen, 10780 + len(samples))

    def test_invalid_audio_is_rejected_without_advancing_the_clock(self):
        detector = KeystrokeDetector()
        for samples in (np.array([np.nan]), np.array([np.inf]), np.zeros((10, 2))):
            with self.subTest(shape=samples.shape), self.assertRaises(ValueError):
                detector.feed(samples)
            self.assertEqual(detector.samples_seen, 0)

    def test_buffers_stay_bounded_for_long_running_quiet_input(self):
        detector = KeystrokeDetector(DetectorConfig(calibration_seconds=0.2))
        for _ in range(120):
            self.assertEqual(detector.feed(np.zeros(24000, dtype=np.float32)), [])
        self.assertLessEqual(sum(len(frame.data) for frame in detector._history), round(0.125 * 24000))
        self.assertLess(len(detector._pending), detector.frame_size)
        self.assertFalse(detector._calibration)
        self.assertIsNone(detector._active)


if __name__ == "__main__":
    unittest.main()
