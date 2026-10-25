#!/usr/bin/env python3
"""Listen to a microphone and predict isolated numpad presses, or replay audio.

conf is an uncalibrated model probability, when available. acc is measured only
against --expected-sequence; without known labels it is unavailable. Ctrl+C stops.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from threadpoolctl import threadpool_limits

from keypress_model import DEFAULT_MODEL, FeatureConfig, load_model, predict_waveform
from live_audio import AudioEvent, DetectorConfig, KeystrokeDetector


DEFAULT_DEVICE = "LCS USB Audio Mono"
DEFAULT_TIMEZONE = "Australia/Melbourne"


class PredictionTracker:
    """Positional sequence scoring; never substitute confidence for accuracy."""

    def __init__(self, expected_sequence: str | None = None):
        if expected_sequence is not None and (
            not expected_sequence or not expected_sequence.isascii() or not expected_sequence.isdecimal()
        ):
            raise ValueError("--expected-sequence must be a nonempty string of digits 0-9")
        self.expected_sequence = expected_sequence
        self.count = 0
        self.correct_count = 0
        self.rejected_count = 0

    def record(self, predicted_key: str | None) -> dict:
        self.count += 1
        if predicted_key is None:
            self.rejected_count += 1
        actual, correct = None, None
        if self.expected_sequence is not None:
            if self.count <= len(self.expected_sequence):
                actual = self.expected_sequence[self.count - 1]
            correct = actual is not None and predicted_key == actual
            self.correct_count += int(correct)
        compared = self.count if self.expected_sequence is not None else 0
        return {
            "index": self.count, "actual_key": actual, "correct": correct,
            "correct_count": self.correct_count, "compared_count": compared,
            "accuracy": self.correct_count / compared if compared else None,
        }

    def summary(self) -> dict:
        expected = len(self.expected_sequence) if self.expected_sequence is not None else None
        compared = max(expected, self.count) if expected is not None else 0
        return {
            "event_count": self.count, "expected_count": expected,
            "rejected_count": self.rejected_count,
            "correct_count": self.correct_count, "compared_count": compared,
            "accuracy": self.correct_count / compared if compared else None,
            "missing_count": max(0, expected - self.count) if expected is not None else None,
            "extra_count": max(0, self.count - expected) if expected is not None else None,
        }


def dbfs(amplitude: float) -> float:
    return 20 * math.log10(max(float(amplitude), 1e-12))


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_prediction(
    event_record: dict, time_format: str = "elapsed", timezone_name: str = DEFAULT_TIMEZONE
) -> str:
    if time_format == "wall":
        stamp = datetime.fromtimestamp(event_record["timestamp"], ZoneInfo(timezone_name))
        timestamp = stamp.strftime("%H:%M:%S.%f")[:-3]
    else:
        milliseconds = max(0, round(event_record["elapsed_seconds"] * 1000))
        hours, rest = divmod(milliseconds, 3600000)
        minutes, rest = divmod(rest, 60000)
        seconds, millis = divmod(rest, 1000)
        timestamp = f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"
    ranked = []
    for item in event_record["ranked_keys"]:
        probability = item.get("probability")
        value = percent(probability) if probability is not None else f"{item['score']:+.3f}score"
        ranked.append(f"{item['key']}:{value}")
    parts = [
        f"acc={percent(event_record['accuracy'])}",
        f"conf={percent(event_record['confidence'])}",
        f"top3={'|'.join(ranked) or 'n/a'}",
        f"snr={event_record['snr_db']:.1f}dB",
        f"peak={event_record['peak_dbfs']:.1f}dBFS",
        f"latency={event_record['latency_ms']:.0f}ms",
        f"infer={event_record['inference_ms']:.1f}ms",
        f"n={event_record['event_index']}",
    ]
    if event_record.get("actual_key") is not None:
        parts.extend((f"actual={event_record['actual_key']}",
                      f"hits={event_record['correct_count']}/{event_record['compared_count']}"))
    if event_record.get("margin") is not None:
        parts.append(f"margin={event_record['margin']:.3f}")
    if event_record.get("quality"):
        parts.append(f"quality={'|'.join(event_record['quality'])}")
    if event_record.get("status") == "rejected":
        parts.append("status=rejected")
    return f"{timestamp}: {event_record['predicted_key'] or '?'} ({', '.join(parts)})"


def prediction_record(
    event: AudioEvent, artifact: dict, sample_rate: int, tracker: PredictionTracker,
    epoch: float, samples_seen: int, replay: bool,
) -> dict:
    started = time.perf_counter()
    quality = list(event.quality)
    prediction = {"predicted_key": None, "confidence": None, "ranked_keys": []}
    reason = None
    if quality:
        reason = f"Unreliable audio event: {', '.join(quality)}"
    else:
        try:
            prediction = predict_waveform(event.waveform, artifact, top_k=3, sample_rate=sample_rate)
        except ValueError as error:
            reason = str(error)
            quality.append("invalid_audio")
    inference_ms = (time.perf_counter() - started) * 1000
    tracked = tracker.record(prediction["predicted_key"])
    peak = float(np.max(np.abs(event.waveform))) if len(event.waveform) else 0.0
    frame_size = max(1, round(0.005 * sample_rate))
    count = len(event.waveform) // frame_size
    if count:
        frame_rms = np.sqrt(np.mean(
            event.waveform[:count * frame_size].reshape(count, frame_size).astype(np.float64) ** 2, axis=1
        ))
        signal_rms = float(frame_rms.max())
    else:
        signal_rms = peak
    elapsed = event.peak_sample / sample_rate
    timestamp = epoch + elapsed
    latency_ms = ((samples_seen - event.peak_sample) / sample_rate * 1000 + inference_ms
                  if replay else max(0, (time.time() - timestamp) * 1000))
    return {
        "event_index": tracked.pop("index"), **tracked, **prediction,
        "elapsed_seconds": elapsed, "timestamp": timestamp,
        "timestamp_source": "replay_start_plus_elapsed" if replay else "microphone_adc_clock",
        "onset_seconds": event.onset_sample / sample_rate,
        "snr_db": dbfs(signal_rms) - dbfs(event.noise_rms), "peak_dbfs": dbfs(peak),
        "noise_dbfs": dbfs(event.noise_rms), "threshold_dbfs": dbfs(event.threshold_rms),
        "latency_ms": latency_ms, "inference_ms": inference_ms,
        "sample_rate": sample_rate, "clip_seconds": len(event.waveform) / sample_rate,
        "quality": quality, "status": "rejected" if reason else "predicted", "reason": reason,
    }


def sounddevice_module():
    try:
        import sounddevice
    except (ImportError, OSError) as error:
        raise ValueError("Microphone capture needs sounddevice and PortAudio; install the requirements and PortAudio") from error
    return sounddevice


def list_devices() -> None:
    sd = sounddevice_module()
    try:
        for index, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                default = " [script default]" if DEFAULT_DEVICE.lower() in device["name"].lower() else ""
                print(f"{index}: {device['name']} ({device['max_input_channels']} input channels, "
                      f"{device['default_samplerate']:.0f} Hz){default}")
    except sd.PortAudioError as error:
        raise ValueError(f"Cannot query microphone devices: {error}") from error


def input_settings(device: str | int, channel: int, sample_rate: int | None):
    sd = sounddevice_module()
    try:
        info = sd.query_devices(device, kind="input")
        rate = sample_rate or round(info["default_samplerate"])
        if channel > info["max_input_channels"]:
            raise ValueError(f"{info['name']} has only {info['max_input_channels']} input channels")
        sd.check_input_settings(device=device, channels=channel, samplerate=rate, dtype="float32")
    except sd.PortAudioError as error:
        raise ValueError(f"Cannot use microphone {device!r}: {error}. Run --list-devices and choose --device.") from error
    return sd, info, rate


def microphone_blocks(sd, device, channel: int, sample_rate: int, frames: int, duration: float | None):
    """Only copy/enqueue in the audio callback; classification runs in the consumer."""
    blocks = queue.Queue(maxsize=max(2, math.ceil(2 * sample_rate / frames)))
    finished = threading.Event()
    failure = []
    anchor = []

    def callback(indata, frame_count, timing, status):
        try:
            if status:
                raise RuntimeError(f"Microphone audio discontinuity: {status}")
            if not anchor:
                anchor.append(time.time() + timing.inputBufferAdcTime - timing.currentTime)
            blocks.put_nowait((indata[:, channel - 1].copy(), anchor[0]))
        except Exception as error:
            failure.append(str(error) if not isinstance(error, queue.Full) else "Audio queue overflow")
            finished.set()
            raise sd.CallbackAbort

    try:
        with sd.InputStream(device=device, channels=channel, samplerate=sample_rate, dtype="float32",
                            blocksize=frames, latency="low", callback=callback,
                            finished_callback=finished.set) as stream:
            started = time.monotonic()
            while duration is None or time.monotonic() - started < duration:
                if failure:
                    raise RuntimeError(f"{failure[0]}; stopped because dropped audio invalidates event timing and sequence accuracy")
                try:
                    yield blocks.get(timeout=0.1)
                except queue.Empty:
                    if failure:
                        raise RuntimeError(f"{failure[0]}; dropped audio invalidates sequence accuracy")
                    if finished.is_set() or not stream.active:
                        raise RuntimeError("Microphone stream stopped unexpectedly")
            if failure:
                raise RuntimeError(f"{failure[0]}; dropped audio invalidates sequence accuracy")
    except sd.PortAudioError as error:
        raise ValueError(f"Microphone capture failed: {error}") from error


def replay_blocks(path: Path, sample_rate: int, frames: int, duration: float | None):
    """Stream the complete file through FFmpeg; never open the microphone backend."""
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Replay file does not exist: {path}")
    if shutil.which("ffmpeg") is None:
        raise ValueError("Replay needs FFmpeg on PATH")
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0"]
    if duration is not None:
        command += ["-t", str(duration)]
    command += ["-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1"]
    epoch = time.time()
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            while True:
                raw = process.stdout.read(frames * 4)
                if not raw:
                    break
                if len(raw) % 4:
                    raise ValueError("Replay produced an incomplete PCM sample")
                yield np.frombuffer(raw, dtype="<f4"), epoch
            if process.wait() != 0:
                errors.seek(0)
                raise ValueError(f"Cannot decode replay: {errors.read().decode(errors='replace').strip()}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def device_identifier(value: str) -> str | int:
    return int(value) if value.lstrip("-").isdecimal() else value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=device_identifier, default=DEFAULT_DEVICE,
                        help=f"input device ID/name (default: {DEFAULT_DEVICE})")
    parser.add_argument("--list-devices", action="store_true", help="list input devices without recording")
    parser.add_argument("--channel", type=int, default=1, help="one-based input channel (default: 1)")
    parser.add_argument("--sample-rate", type=int, help="capture rate; defaults to the device's native rate")
    parser.add_argument("--replay", type=Path, help="process an audio file instead of opening the microphone")
    parser.add_argument("--expected-sequence", help="known key order, enabling measured positional accuracy")
    parser.add_argument("--calibration-seconds", type=float, default=1.5, help="initial background calibration (default: 1.5)")
    parser.add_argument("--threshold-db", type=float, help="fixed trigger threshold in dBFS; otherwise adaptive")
    parser.add_argument("--trigger-db", type=float, default=18, help="adaptive trigger dB above noise (default: 18)")
    parser.add_argument("--capture-seconds", type=float, default=0.5, help="post-onset context (default: 0.5)")
    parser.add_argument("--block-ms", type=float, default=10, help="input block duration (default: 10 ms)")
    parser.add_argument("--duration", type=float, help="stop after this many seconds; otherwise Ctrl+C")
    parser.add_argument("--time-format", choices=("auto", "wall", "elapsed"), default="auto")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE, help="wall-clock IANA timezone")
    parser.add_argument("--json", action="store_true", help="print prediction records as JSON lines")
    args = parser.parse_args(argv)
    if args.channel < 1 or not math.isfinite(args.block_ms) or not 1 <= args.block_ms <= 200:
        parser.error("--channel must be positive; --block-ms must be between 1 and 200")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive and finite")
    if args.sample_rate is not None and args.sample_rate < 1000:
        parser.error("--sample-rate must be at least 1000 Hz")
    if not math.isfinite(args.capture_seconds) or not 0.32 <= args.capture_seconds <= 3:
        parser.error("--capture-seconds must be between 0.32 and 3")
    blocks = None
    tracker = None
    detector = None
    epoch = None
    interrupted = False
    try:
        if args.list_devices:
            list_devices()
            return 0
        ZoneInfo(args.timezone)
        tracker = PredictionTracker(args.expected_sequence)
        artifact = load_model(args.model)
        feature_config = FeatureConfig(**artifact["feature_config"])
        replay = args.replay is not None
        if replay:
            rate = args.sample_rate or feature_config.sample_rate
        else:
            sd, info, rate = input_settings(args.device, args.channel, args.sample_rate)
        config = DetectorConfig(
            sample_rate=rate, calibration_seconds=args.calibration_seconds,
            threshold_db=args.threshold_db, onset_db_above_noise=args.trigger_db,
            capture_seconds=max(args.capture_seconds,
                                feature_config.duration_seconds - feature_config.pre_peak_seconds + 0.04),
        )
        detector = KeystrokeDetector(config)
        frames = max(1, round(rate * args.block_ms / 1000))
        time_format = ("elapsed" if replay else "wall") if args.time_format == "auto" else args.time_format
        if replay:
            blocks = replay_blocks(args.replay, rate, frames, args.duration)
            print(f"Replaying {args.replay} at {rate} Hz; microphone stays closed.", file=sys.stderr)
        else:
            blocks = microphone_blocks(sd, args.device, args.channel, rate, frames, args.duration)
            print(f"Listening on {info['name']}, channel {args.channel}, {rate} Hz. Ctrl+C stops.", file=sys.stderr)
        print(f"Calibrating background for {config.calibration_seconds:g}s. Use isolated numpad taps at least "
              f"{config.min_interval_seconds:g}s apart. conf is uncalibrated; acc requires --expected-sequence.",
              file=sys.stderr)

        def emit(event):
            record = prediction_record(event, artifact, rate, tracker, epoch, detector.samples_seen, replay)
            print(json.dumps(record) if args.json else format_prediction(record, time_format, args.timezone), flush=True)

        notified = False
        with threadpool_limits(limits=1):
            try:
                for waveform, block_epoch in blocks:
                    if epoch is None:
                        epoch = block_epoch
                    events = detector.feed(waveform)
                    if detector.calibrated and not notified:
                        print(f"Ready: noise={dbfs(detector.noise_rms):.1f}dBFS, "
                              f"trigger={dbfs(detector.threshold_rms):.1f}dBFS.", file=sys.stderr)
                        notified = True
                    for event in events:
                        emit(event)
            except KeyboardInterrupt:
                interrupted = True
            for event in detector.flush():
                emit(event)
        summary = tracker.summary()
        print(f"Stopped{' by Ctrl+C' if interrupted else ''}: events={summary['event_count']}, "
              f"rejected={summary['rejected_count']}, acc={percent(summary['accuracy'])} "
              f"({summary['correct_count']}/{summary['compared_count']} known/expected positions).", file=sys.stderr)
        return 0
    except (ValueError, OSError, RuntimeError, ZoneInfoNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if blocks is not None:
            blocks.close()


if __name__ == "__main__":
    raise SystemExit(main())
