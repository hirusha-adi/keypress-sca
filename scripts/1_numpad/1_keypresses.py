#!/usr/bin/env python3
"""Split numpad training recordings and digit-sequence test recordings.

Run with --dry-run to inspect detections without writing anything. See the
README in this directory for dependencies, safeguards, and output details.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.signal import find_peaks


DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "datasets" / "1_numpad"
DEFAULT_TESTS = Path(__file__).resolve().parents[2] / "datasets" / "1_tests"
FRAME_SECONDS = 0.005
MIN_PRESS_INTERVAL_SECONDS = 0.350
QUIET_SECONDS = 0.050
AFTER_PRESS_SECONDS = 0.200
BEFORE_PRESS_SECONDS = 0.100
REPORT_NAME = "split_manifest.json"
MP3_SAMPLE_RATES = {8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000}


@dataclass
class Recording:
    path: Path
    sample_rate: int
    samples: np.ndarray
    frame_size: int
    peaks: np.ndarray
    cuts: list[int]
    threshold: float
    noise_floor: float
    quiet_threshold: float
    sequence: str | None = None

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def clip_names(self) -> list[str]:
        if self.sequence is not None:
            names = sequence_clip_names(self.sequence)
            if len(names) != len(self.peaks):
                raise ValueError("Sequence labels do not match the detected press count")
            return names
        return [f"{index}.mp3" for index in range(1, len(self.peaks) + 1)]


def sequence_clip_names(sequence: str) -> list[str]:
    """Name clips by one-based press number followed by the actual key."""
    if not sequence or not sequence.isascii() or not sequence.isdecimal():
        raise ValueError("Test recording filenames must contain only digits 0-9, in press order")
    return [f"{index}-{key}.mp3" for index, key in enumerate(sequence, 1)]


def run_audio_tool(command: list[str], **kwargs) -> bytes:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    if result.returncode:
        raise ValueError(f"{command[0]} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def decode(path: Path) -> tuple[np.ndarray, int]:
    metadata = json.loads(run_audio_tool([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path),
    ]))
    streams = metadata.get("streams", [])
    if not streams:
        raise ValueError(f"{path.name}: no audio stream")
    sample_rate = int(streams[0]["sample_rate"])
    channels = int(streams[0]["channels"])
    if channels not in (1, 2) or sample_rate not in MP3_SAMPLE_RATES:
        raise ValueError(f"{path.name}: cannot preserve {sample_rate} Hz / {channels} channels in MP3")
    pcm = run_audio_tool([
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-map", "0:a:0", "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1",
    ])
    samples = np.frombuffer(pcm, dtype="<f4").reshape(-1, channels)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError(f"{path.name}: empty or non-finite audio")
    return samples, sample_rate


def rms_envelope(samples: np.ndarray, frame_size: int) -> np.ndarray:
    """Use energy across channels; averaging waveforms could cancel stereo audio."""
    count = len(samples) // frame_size
    frames = samples[:count * frame_size].reshape(count, frame_size, samples.shape[1])
    energy = np.mean(np.square(frames, dtype=np.float64), axis=(1, 2))
    if count * frame_size < len(samples):
        tail = np.mean(np.square(samples[count * frame_size:], dtype=np.float64))
        energy = np.append(energy, tail)
    return np.sqrt(energy)


def find_quiet_cut(
    envelope: np.ndarray,
    left_peak: int,
    right_peak: int,
    frame_seconds: float,
    quiet_threshold: float,
) -> int:
    """Return a frame boundary surrounded by at least 50 ms of quiet audio."""
    lo = left_peak + math.ceil(AFTER_PRESS_SECONDS / frame_seconds)
    hi = right_peak - math.ceil(BEFORE_PRESS_SECONDS / frame_seconds)
    width = max(2, math.ceil(QUIET_SECONDS / frame_seconds))
    if hi - lo < width:
        raise ValueError("presses are too close to leave safe margins")
    quiet = (envelope[lo:hi] <= quiet_threshold).astype(np.int32)
    windows = np.convolve(quiet, np.ones(width, dtype=np.int32), mode="valid")
    # A cut is at a frame boundary, rather than at an amplitude peak.
    candidates = np.flatnonzero(windows == width) + lo + width // 2
    if not len(candidates):
        raise ValueError("no sufficiently quiet gap; refusing to cut through sound")
    midpoint = (left_peak + right_peak) / 2
    return int(candidates[np.argmin(np.abs(candidates - midpoint))])


def detect_presses(
    envelope: np.ndarray,
    frame_seconds: float,
    threshold_db: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Detect press peaks without count validation or segmentation, for plotting too."""
    noise_floor = float(np.percentile(envelope, 20))
    # Estimate signal level independently of the expected count. Do not take the
    # top 25 peaks: that would hide missed presses or turn handling noise into data.
    threshold = max(noise_floor * 10, float(np.percentile(envelope, 99.5)) * 0.20, 1e-7)
    if threshold_db is not None:
        threshold = 10 ** (threshold_db / 20)
    # The refractory interval groups a press, its release click, and ringing.
    # Padding allows a genuine peak at the recording's first or last frame.
    peaks, _ = find_peaks(
        np.pad(envelope, (1, 1)),
        height=threshold,
        prominence=threshold / 2,
        distance=math.ceil(MIN_PRESS_INTERVAL_SECONDS / frame_seconds),
    )
    peaks = peaks - 1
    return peaks, threshold, noise_floor


def analyze(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    expected_counts: set[int],
    threshold_db: float | None = None,
) -> Recording:
    frame_size = max(1, round(FRAME_SECONDS * sample_rate))
    frame_seconds = frame_size / sample_rate
    envelope = rms_envelope(samples, frame_size)
    peaks, threshold, noise_floor = detect_presses(envelope, frame_seconds, threshold_db)
    if len(peaks) not in expected_counts:
        allowed = ", ".join(str(count) for count in sorted(expected_counts))
        raise ValueError(
            f"{path.name}: detected {len(peaks)} presses; expected {allowed}. "
            f"RMS threshold {20 * math.log10(threshold):.1f} dBFS. "
            "Inspect the recording or adjust --threshold-db; no files have been written."
        )
    quiet_threshold = max(noise_floor * 4, threshold * 0.15)
    cuts = [0]
    for left, right in zip(peaks[:-1], peaks[1:]):
        try:
            cut = find_quiet_cut(envelope, int(left), int(right), frame_seconds, quiet_threshold)
        except ValueError as error:
            raise ValueError(
                f"{path.name}: between {left * frame_seconds:.3f}s and "
                f"{right * frame_seconds:.3f}s: {error}"
            ) from error
        cuts.append(cut * frame_size)
    # Keep ALL decoded samples, including leading/trailing quiet, rather than
    # trying to trim weak onsets or tails with an amplitude threshold.
    cuts.append(len(samples))
    return Recording(path, sample_rate, samples, frame_size, peaks, cuts,
                     threshold, noise_floor, quiet_threshold)


def analyze_sequence(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    threshold_db: float | None = None,
) -> Recording:
    names = sequence_clip_names(path.stem)
    recording = analyze(path, samples, sample_rate, {len(names)}, threshold_db)
    recording.sequence = path.stem
    return recording


def numbered_mp3s(directory: Path) -> list[Path]:
    return [path for path in directory.glob("*.mp3") if path.stem.isdecimal()]


def generated_mp3s(directory: Path, recording: Recording) -> list[Path]:
    if recording.sequence is None:
        return numbered_mp3s(directory)
    # Recognize the former key-occurrence format too, so --overwrite removes
    # obsolete files when regenerating a sequence with the corrected naming.
    return [path for path in directory.glob("*.mp3")
            if path.is_file() and re.fullmatch(r"(?:[1-9][0-9]*-[0-9]|[0-9]-[1-9][0-9]*)", path.stem)]


def check_outputs(dataset: Path, recordings: list[Recording], overwrite: bool) -> None:
    report = dataset / REPORT_NAME
    if report.exists() and (not report.is_file() or not overwrite):
        raise ValueError(f"{report} already exists; use --overwrite to replace previous results")
    for recording in recordings:
        directory = dataset / recording.path.stem
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"{directory} exists and is not a directory")
        if generated_mp3s(directory, recording) and not overwrite:
            raise ValueError(f"{directory} already contains clips; use --overwrite to replace them")
        for name in recording.clip_names:
            target = directory / name
            if target.exists() and not target.is_file():
                raise ValueError(f"{target} exists and is not a file")


def export(dataset: Path, recordings: list[Recording], gain: float, overwrite: bool) -> None:
    check_outputs(dataset, recordings, overwrite)
    manifest = {
        "schema_version": 1,
        "description": "One press + release per clip; contiguous sample ranges cover each recording completely.",
        "encoder": "libmp3lame -q:a 0",
        "gain": gain,
        "gain_db": 20 * math.log10(gain),
        "analysis": {
            "frame_seconds": FRAME_SECONDS,
            "min_press_interval_seconds": MIN_PRESS_INTERVAL_SECONDS,
            "quiet_window_seconds": QUIET_SECONDS,
            "after_press_guard_seconds": AFTER_PRESS_SECONDS,
            "before_press_guard_seconds": BEFORE_PRESS_SECONDS,
        },
        "recordings": [],
    }
    # Stage every encoded file before replacing any previous dataset output.
    with tempfile.TemporaryDirectory(prefix=".keypress-split-", dir=dataset) as staging:
        staging = Path(staging)
        for recording in recordings:
            directory = staging / recording.path.stem
            directory.mkdir()
            clips = []
            for index, (start, end, peak, name) in enumerate(
                zip(recording.cuts[:-1], recording.cuts[1:], recording.peaks, recording.clip_names), 1
            ):
                target = directory / name
                pcm = (recording.samples[start:end] * gain).astype("<f4").tobytes()
                run_audio_tool([
                    "ffmpeg", "-v", "error", "-nostdin", "-f", "f32le",
                    "-ar", str(recording.sample_rate), "-ac", str(recording.channels),
                    "-i", "pipe:0", "-map_metadata", "-1", "-c:a", "libmp3lame",
                    "-q:a", "0", str(target),
                ], input=pcm)
                clips.append({
                    "file": f"{recording.path.stem}/{name}",
                    "press_number": index,
                    **({"key": recording.sequence[index - 1]} if recording.sequence is not None else {}),
                    "start_sample": start,
                    "end_sample_exclusive": end,
                    "start_seconds": start / recording.sample_rate,
                    "end_seconds": end / recording.sample_rate,
                    "press_peak_sample": int(peak) * recording.frame_size + recording.frame_size // 2,
                })
            manifest["recordings"].append({
                "source": recording.path.name,
                **({"key_sequence": recording.sequence} if recording.sequence is not None else {}),
                "source_sha256": hashlib.sha256(recording.path.read_bytes()).hexdigest(),
                "sample_rate": recording.sample_rate,
                "channels": recording.channels,
                "decoded_samples": len(recording.samples),
                "detected_presses": len(recording.peaks),
                "noise_floor_rms": recording.noise_floor,
                "detection_threshold_rms": recording.threshold,
                "quiet_threshold_rms": recording.quiet_threshold,
                "clips": clips,
            })
            print(f"Encoded {recording.path.stem}/: {len(clips)} clips", flush=True)
        (staging / REPORT_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for recording in recordings:
            destination = dataset / recording.path.stem
            destination.mkdir(exist_ok=True)
            staged_files = list((staging / recording.path.stem).glob("*.mp3"))
            names = {path.name for path in staged_files}
            for path in staged_files:
                path.replace(destination / path.name)
            if overwrite:
                for stale in generated_mp3s(destination, recording):
                    if stale.name not in names:
                        stale.unlink()
        (staging / REPORT_NAME).replace(dataset / REPORT_NAME)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET,
                        help="directory containing 0.m4a through 9.m4a (default: repository datasets/1_numpad)")
    parser.add_argument("--tests-dir", type=Path, default=DEFAULT_TESTS,
                        help="directory of digit-sequence M4As (default: repository datasets/1_tests)")
    parser.add_argument("--only", choices=("all", "training", "tests"), default="all",
                        help="which recordings to process (default: all); tests avoids rewriting training clips")
    parser.add_argument("--dry-run", action="store_true", help="analyze and validate without writing files")
    parser.add_argument("--overwrite", action="store_true", help="replace generated MP3s and remove stale generated clips")
    parser.add_argument("--expected-counts", nargs="+", type=int, default=[25, 26],
                        help="allowed training counts; tests must match their filename length (default: 25 26)")
    parser.add_argument("--threshold-db", type=float, help="override the adaptive RMS detection threshold in dBFS")
    args = parser.parse_args(argv)
    if any(count < 1 for count in args.expected_counts):
        parser.error("--expected-counts must be positive")
    if args.threshold_db is not None and not (-120 <= args.threshold_db <= 0):
        parser.error("--threshold-db must be between -120 and 0")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            parser.error(f"{tool} is required on PATH")
    groups = []
    if args.only != "tests":
        dataset = args.dataset_dir.resolve()
        sources = [dataset / f"{key}.m4a" for key in range(10)]
        missing = [path.name for path in sources if not path.is_file()]
        if missing:
            parser.error(f"missing recordings in {dataset}: {', '.join(missing)}")
        groups.append((dataset, sources, False))
    if args.only != "training":
        tests = args.tests_dir.resolve()
        sources = sorted(tests.glob("*.m4a"))
        if not sources:
            if args.only == "tests":
                parser.error(f"no .m4a test recordings in {tests}")
            print(f"No test recordings in {tests}; processing training recordings only.")
        else:
            groups.append((tests, sources, True))
    try:
        jobs = []
        # Validate all selected recordings before writing any outputs.
        for dataset, sources, is_test in groups:
            recordings = []
            for path in sources:
                if is_test:
                    sequence_clip_names(path.stem)
                samples, sample_rate = decode(path)
                if is_test:
                    recording = analyze_sequence(path, samples, sample_rate, args.threshold_db)
                else:
                    recording = analyze(path, samples, sample_rate, set(args.expected_counts), args.threshold_db)
                recordings.append(recording)
                print(
                    f"{path.name}: {len(recording.peaks)} presses, "
                    f"{len(samples) / sample_rate:.3f}s, {sample_rate} Hz, {recording.channels} channels, "
                    f"threshold {20 * math.log10(recording.threshold):.1f} dBFS", flush=True,
                )
            # Keep training gain independent of any new test recordings.
            maximum_peak = max(float(np.max(np.abs(recording.samples))) for recording in recordings)
            gain = min(1.0, 0.90 / max(maximum_peak, 1e-10))
            total = sum(len(recording.peaks) for recording in recordings)
            print(f"{dataset.name}: validated {total} clips; shared output gain {20 * math.log10(gain):.2f} dB", flush=True)
            jobs.append((dataset, recordings, gain, total))
        if args.dry_run:
            print("Dry run: no files written.")
        else:
            for dataset, recordings, gain, total in jobs:
                check_outputs(dataset, recordings, args.overwrite)
            for dataset, recordings, gain, total in jobs:
                export(dataset, recordings, gain, args.overwrite)
                print(f"Saved {total} MP3s and {dataset / REPORT_NAME}")
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
