"""Streaming, sample-timed detection of isolated press/release audio events.

This module does not open a microphone. Feed native-rate mono PCM to the detector
and resample complete events, if needed, before passing them to the classifier.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    sample_rate: int = 24000
    calibration_seconds: float = 1.5
    threshold_db: float | None = None
    onset_db_above_noise: float = 18.0
    min_interval_seconds: float = 0.350
    pre_seconds: float = 0.120
    capture_seconds: float = 0.500

    def __post_init__(self) -> None:
        if not isinstance(self.sample_rate, int) or self.sample_rate < 1000:
            raise ValueError("sample_rate must be an integer of at least 1000 Hz")
        values = (self.calibration_seconds, self.onset_db_above_noise,
                  self.min_interval_seconds, self.pre_seconds, self.capture_seconds)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Detector settings must be finite")
        if not 0 <= self.calibration_seconds <= 30:
            raise ValueError("calibration_seconds must be between 0 and 30")
        if self.onset_db_above_noise <= 0:
            raise ValueError("onset_db_above_noise must be positive")
        if self.min_interval_seconds < 0.2:
            raise ValueError("min_interval_seconds must be at least 0.2")
        if self.pre_seconds < 0.1 or self.pre_seconds > 2:
            raise ValueError("pre_seconds must be between 0.1 and 2")
        if not 0.32 <= self.capture_seconds <= 3:
            raise ValueError("capture_seconds must be between 0.32 and 3")
        if self.threshold_db is not None and (
            not math.isfinite(self.threshold_db) or not -120 <= self.threshold_db <= 0
        ):
            raise ValueError("threshold_db must be between -120 and 0 dBFS")


@dataclass(frozen=True)
class AudioEvent:
    waveform: np.ndarray
    start_sample: int
    onset_sample: int
    peak_sample: int
    noise_rms: float
    threshold_rms: float
    quality: tuple[str, ...] = ()

    @property
    def end_sample(self) -> int:
        return self.start_sample + len(self.waveform)


@dataclass(frozen=True)
class _Frame:
    start: int
    data: np.ndarray
    rms: float

    @property
    def end(self) -> int:
        return self.start + len(self.data)


@dataclass
class _ActiveEvent:
    frames: list[_Frame]
    onset: int
    peak: int
    peak_rms: float
    noise_rms: float
    threshold_rms: float
    target_end: int
    quality: set[str] = field(default_factory=set)


class KeystrokeDetector:
    """Detect slow, isolated presses in arbitrary-sized mono PCM chunks.

    Calibration buffers are replayed after the noise estimate is available, so a
    press during calibration is delayed rather than silently discarded. All
    timestamps count samples supplied to feed(), even across reset().
    """

    def __init__(self, config: DetectorConfig = DetectorConfig()) -> None:
        self.config = config
        self.frame_size = max(1, round(0.005 * config.sample_rate))
        self._pre_samples = round(config.pre_seconds * config.sample_rate)
        self._capture_samples = round(config.capture_seconds * config.sample_rate)
        self._interval_samples = round(config.min_interval_seconds * config.sample_rate)
        self._post_samples = round(0.280 * config.sample_rate)
        self._model_pre_samples = round(0.040 * config.sample_rate)
        self._input_end = 0
        self._frame_cursor = 0
        self.reset()

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def samples_seen(self) -> int:
        return self._input_end

    @property
    def noise_rms(self) -> float:
        return self._noise_rms

    @property
    def threshold_rms(self) -> float:
        if self.config.threshold_db is not None:
            return 10 ** (self.config.threshold_db / 20)
        ratio = 10 ** (self.config.onset_db_above_noise / 20)
        return max(self._noise_rms * ratio, self._reference_rms * 0.15, 1e-5)

    def reset(self, dropped_samples: int = 0) -> None:
        """Discard unfinished audio and recalibrate, preserving the sample clock."""
        if not isinstance(dropped_samples, int) or dropped_samples < 0:
            raise ValueError("dropped_samples must be a non-negative integer")
        self._input_end += dropped_samples
        self._frame_cursor = self._input_end
        self._pending = np.empty(0, dtype=np.float32)
        self._history: deque[_Frame] = deque()
        self._calibration: list[_Frame] = []
        self._calibration_samples = 0
        self._calibration_target = max(self.frame_size, round(
            self.config.calibration_seconds * self.config.sample_rate
        ))
        self._quiet_rms: deque[float] = deque(maxlen=max(20, round(5 / 0.005)))
        self._stationary_rms: deque[tuple[int, int, float]] = deque()
        self._noise_rms = 1e-8
        self._reference_rms = 0.0
        self._above = False
        self._active: _ActiveEvent | None = None
        self._calibrated = self.config.calibration_seconds == 0 and self.config.threshold_db is not None

    def feed(self, mono: np.ndarray) -> list[AudioEvent]:
        samples = np.asarray(mono, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("feed() expects finite, one-dimensional mono PCM")
        if not len(samples):
            return []
        self._input_end += len(samples)
        data = np.concatenate((self._pending, samples))
        count = len(data) // self.frame_size
        result = []
        for offset in range(0, count * self.frame_size, self.frame_size):
            frame_data = data[offset:offset + self.frame_size].copy()
            result.extend(self._consume(frame_data))
        self._pending = data[count * self.frame_size:].copy()
        return result

    def flush(self) -> list[AudioEvent]:
        """Finish replay/file input; incomplete trailing context is marked."""
        result = []
        if len(self._pending):
            tail = self._pending
            self._pending = np.empty(0, dtype=np.float32)
            result.extend(self._consume(tail))
        if self._calibration:
            result.extend(self._finish_calibration())
        if self._active is not None:
            result.append(self._finish_event(self._active.frames[-1].end))
        return result

    def _consume(self, data: np.ndarray) -> list[AudioEvent]:
        frame = _Frame(self._frame_cursor, data, float(np.sqrt(np.mean(data.astype(np.float64) ** 2))))
        self._frame_cursor += len(data)
        if not self._calibrated:
            self._calibration.append(frame)
            self._calibration_samples += len(data)
            if self._calibration_samples >= self._calibration_target:
                return self._finish_calibration()
            return []
        return self._process(frame)

    def _finish_calibration(self) -> list[AudioEvent]:
        frames = self._calibration
        self._calibration = []
        rms = np.asarray([frame.rms for frame in frames])
        self._noise_rms = max(float(np.percentile(rms, 20)), 1e-8)
        self._quiet_rms.extend(float(value) for value in rms if value <= self._noise_rms * 2)
        peak = float(np.max(rms))
        self._reference_rms = peak if peak > self._noise_rms * 6 else 0.0
        if self.config.threshold_db is not None and self._noise_rms >= self.threshold_rms:
            raise ValueError(
                "Background noise exceeds the fixed --threshold-db setting; "
                "raise the threshold or recalibrate in quieter conditions"
            )
        self._calibrated = True
        result = []
        for frame in frames:
            result.extend(self._process(frame))
        return result

    def _start_event(self, frame: _Frame, threshold: float) -> None:
        self._active = _ActiveEvent(
            list(self._history) + [frame], frame.start,
            frame.start + len(frame.data) // 2, frame.rms,
            self._noise_rms, threshold, frame.start + self._capture_samples,
        )

    def _process(self, frame: _Frame) -> list[AudioEvent]:
        result = []
        self._stationary_rms.append((frame.start, frame.end, frame.rms))
        earliest = frame.end - round(0.500 * self.config.sample_rate)
        while self._stationary_rms and self._stationary_rms[0][1] <= earliest:
            self._stationary_rms.popleft()
        self._recover_background()
        self._reference_rms *= math.exp(-len(frame.data) / (30 * self.config.sample_rate))
        threshold = self.threshold_rms
        above = frame.rms >= threshold
        rising = above and not self._above
        active = self._active
        if active is None:
            if rising:
                self._start_event(frame, threshold)
        else:
            elapsed = frame.start - active.onset
            if rising and elapsed >= self._interval_samples:
                end, unsafe = self._cut_before_next_press(frame.start)
                if unsafe:
                    active.quality.add("no_quiet_boundary")
                result.append(self._finish_event(end))
                self._start_event(frame, threshold)
            else:
                if (rising and elapsed >= round(0.180 * self.config.sample_rate)
                        and frame.rms >= active.peak_rms * 0.25):
                    active.quality.add("closely_spaced_transients")
                active.frames.append(frame)
                if frame.rms > active.peak_rms:
                    active.peak_rms = frame.rms
                    active.peak = frame.start + len(frame.data) // 2
                    # Retain the model's post-peak context, with a bounded timeout.
                    active.target_end = min(
                        max(active.target_end, active.peak + self._post_samples),
                        active.onset + self._capture_samples + self._interval_samples,
                    )
                if frame.end >= active.target_end:
                    result.append(self._finish_event(frame.end))
        if self._active is not None:
            self._reference_rms = max(self._reference_rms, self._active.peak_rms)
        elif not above and frame.rms <= threshold * 0.4:
            self._quiet_rms.append(frame.rms)
            if self._quiet_rms:
                estimate = max(float(np.percentile(self._quiet_rms, 50)), 1e-8)
                alpha = 1 - math.exp(-len(frame.data) / self.config.sample_rate)
                self._noise_rms += alpha * (estimate - self._noise_rms)
        self._above = above
        self._history.append(frame)
        earliest = frame.end - self._pre_samples
        while self._history and self._history[0].end <= earliest:
            self._history.popleft()
        return result

    def _recover_background(self) -> None:
        """Recover from a stationary ambient jump without swallowing key transients.

        Updating only below the old trigger can otherwise leave the detector
        permanently above threshold. Require half a second with no active event
        and no strong peak anywhere in that window before accepting a new floor.
        """
        if self._active is not None or not self._stationary_rms:
            return
        first, last = self._stationary_rms[0], self._stationary_rms[-1]
        if last[1] - first[0] < round(0.500 * self.config.sample_rate):
            return
        values = np.asarray([item[2] for item in self._stationary_rms])
        estimate = max(float(np.percentile(values, 20)), 1e-8)
        if estimate < self._noise_rms * 2 or np.max(values) > estimate * 2:
            return
        if self.config.threshold_db is not None and estimate >= self.threshold_rms:
            raise ValueError(
                "Sustained background noise exceeds the fixed --threshold-db setting; "
                "raise the threshold or recalibrate in quieter conditions"
            )
        self._noise_rms = estimate
        self._quiet_rms.clear()
        self._quiet_rms.extend(float(value) for value in values)
        # The ambient level is now below the new adaptive threshold, so a future
        # rise can trigger even if the old absolute-threshold state was stuck.
        self._above = False

    def _cut_before_next_press(self, onset: int) -> tuple[int, bool]:
        active = self._active
        required_end = active.peak + self._post_samples
        upper = min(active.frames[-1].end, onset - self._model_pre_samples)
        lower = min(required_end, upper)
        quiet_limit = max(active.noise_rms * 4, active.threshold_rms * 0.15)
        width = max(2, math.ceil(0.050 * self.config.sample_rate / self.frame_size))
        energies = np.asarray([frame.rms for frame in active.frames])
        candidates = []
        for index in range(width // 2, len(active.frames) - (width + 1) // 2 + 1):
            cut = active.frames[index].start
            if lower <= cut <= upper:
                window = energies[index - width // 2:index + (width + 1) // 2]
                if len(window) == width and np.max(window) <= quiet_limit:
                    candidates.append(cut)
        if candidates:
            midpoint = (active.peak + onset) / 2
            return min(candidates, key=lambda value: abs(value - midpoint)), False
        return upper, True

    def _finish_event(self, end: int) -> AudioEvent:
        active = self._active
        self._active = None
        start = active.frames[0].start
        waveform = np.concatenate([frame.data for frame in active.frames])[:max(0, end - start)]
        event_rms = np.asarray([frame.rms for frame in active.frames if frame.start >= active.onset])
        if (end - active.onset >= self._interval_samples and len(event_rms)
                and np.max(event_rms) <= 2 * max(float(np.percentile(event_rms, 20)), 1e-8)):
            active.quality.add("stationary_background")
        if start > active.peak - self._model_pre_samples or end < active.peak + self._post_samples:
            active.quality.add("truncated_context")
        return AudioEvent(
            waveform, start, active.onset, active.peak,
            active.noise_rms, active.threshold_rms, tuple(sorted(active.quality)),
        )
