#!/usr/bin/env python3
"""Plot dataset 2 RMS envelopes, press peaks, and amplitude percentiles.

Writes an overview, readable 30-second panels for each training key, and an
overview of digit-sequence test recordings to work/2_numpad by default. Audio
analysis shares 1_keypresses.py's detector, without requiring valid split gaps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


# Load the sibling by path because Python module names cannot begin with digits.
SPEC = importlib.util.spec_from_file_location(
    "numpad_splitter", Path(__file__).with_name("1_keypresses.py")
)
splitter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = splitter
SPEC.loader.exec_module(splitter)

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "work" / "2_numpad" / "rms_peaks.png"
PERCENTILES = (10, 20, 50, 80, 95)
MEDIA_SUFFIXES = {".mp4", ".m4a", ".wav", ".mp3", ".flac", ".aac", ".ogg", ".mov"}
DETAIL_SECONDS = 30.0


@dataclass
class Analysis:
    path: Path
    sample_rate: int
    channels: int
    duration: float
    times: np.ndarray
    envelope: np.ndarray
    peaks: np.ndarray
    threshold: float
    noise_floor: float
    percentiles: np.ndarray

    @property
    def percentile_text(self) -> str:
        return "   ".join(
            f"P{percentile}={value:.5f}"
            for percentile, value in zip(PERCENTILES, self.percentiles)
        )

    @property
    def title(self) -> str:
        return (
            f"{self.path.name} | {len(self.peaks)} detected presses | {self.duration:.3f} s | "
            f"{self.sample_rate:,} Hz / {self.channels} ch | "
            f"threshold {20 * math.log10(self.threshold):.1f} dBFS\n"
            f"RMS percentiles: {self.percentile_text}"
        )


def discover_recordings(directory: Path, *, sequences: bool = False) -> list[Path]:
    """Discover original media only; refuse competing files for the same label."""
    if not directory.is_dir():
        raise ValueError(f"Recording directory does not exist: {directory}")
    by_stem: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        stem = path.stem
        if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        if not stem.isascii() or not stem.isdecimal() or (not sequences and len(stem) != 1):
            continue
        if stem in by_stem:
            raise ValueError(
                f"Multiple recordings for {stem}: {by_stem[stem].name}, {path.name}. "
                "Keep one source format per label."
            )
        by_stem[stem] = path
    if not by_stem:
        label = "digit-sequence" if sequences else "single-digit key"
        raise ValueError(f"No {label} recordings found in {directory}")
    return sorted(by_stem.values(), key=lambda path: (int(path.stem), path.stem))


def analyze(path: Path, threshold_db: float | None) -> Analysis:
    samples, sample_rate = splitter.decode(path)
    frame_size = max(1, round(splitter.FRAME_SECONDS * sample_rate))
    envelope = splitter.rms_envelope(samples, frame_size)
    peaks, threshold, noise_floor = splitter.detect_presses(
        envelope, frame_size / sample_rate, threshold_db
    )
    # The last frame can be shorter than 5 ms; retain its actual center time.
    starts = np.arange(len(envelope)) * frame_size
    ends = np.minimum(starts + frame_size, len(samples))
    result = Analysis(
        path=path,
        sample_rate=sample_rate,
        channels=samples.shape[1],
        duration=len(samples) / sample_rate,
        times=(starts + ends) / (2 * sample_rate),
        envelope=envelope,
        peaks=peaks,
        threshold=threshold,
        noise_floor=noise_floor,
        percentiles=np.percentile(envelope, PERCENTILES),
    )
    print(f"{path.name}: {len(peaks)} presses | {result.percentile_text}", flush=True)
    return result


def draw_panel(
    ax: plt.Axes,
    recording: Analysis,
    start: float,
    end: float,
    *,
    annotate: bool,
    sequence: str | None = None,
) -> None:
    visible = (recording.times >= start) & (recording.times < end)
    # Include the final frame when it happens to lie at the plotted boundary.
    if end >= recording.duration:
        visible |= recording.times == end
    times = recording.times
    envelope = recording.envelope
    ax.plot(times[visible], envelope[visible], color="#2672a9", linewidth=0.75, label="5 ms RMS")
    peak_numbers = np.flatnonzero(visible[recording.peaks])
    peaks = recording.peaks[peak_numbers]
    ax.scatter(
        times[peaks], envelope[peaks], marker="x", s=24, linewidths=1.1,
        color="#c82f35", zorder=3, label="Detected press",
    )
    if annotate:
        for number, peak in zip(peak_numbers, peaks):
            label = str(number + 1)
            if sequence is not None:
                label += f":{sequence[number]}\n{times[peak]:.2f}s"
            ax.annotate(
                label, (times[peak], envelope[peak]), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=8,
                color="#a31f26",
            )
    ax.axhline(
        recording.threshold, color="#bc801a", linewidth=0.9, linestyle="--",
        label="Detection threshold",
    )
    ax.axhline(
        recording.noise_floor, color="#777777", linewidth=0.8, linestyle=":",
        label="P20 noise estimate",
    )
    ax.set_xlim(start, max(end, start + 0.001))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    print(f"Saved {output}", flush=True)


def overview(recordings: list[Analysis], output: Path, dpi: int, *, tests: bool = False) -> None:
    fig, axes = plt.subplots(
        len(recordings), 1, figsize=(20, 3.3 * len(recordings) + 1),
        sharey=True, squeeze=False, constrained_layout=True,
    )
    try:
        for recording, ax in zip(recordings, axes[:, 0]):
            sequence = recording.path.stem if tests and len(recording.peaks) == len(recording.path.stem) else None
            draw_panel(ax, recording, 0, recording.duration, annotate=tests, sequence=sequence)
            title = recording.title
            if tests:
                status = "count matches" if sequence is not None else "COUNT MISMATCH: inspect detections"
                title += f"\nFilename labels: {recording.path.stem} | expected {len(recording.path.stem)} presses | {status}"
            ax.set_title(title, fontsize=10, loc="left", pad=8)
        maximum = max(max(float(np.max(item.envelope)), item.threshold) for item in recordings)
        axes[0, 0].set_ylim(0, max(maximum * (1.45 if tests else 1.2), 1e-7))
        axes[0, 0].legend(loc="upper right", ncol=4, fontsize=8, framealpha=0.95)
        total = sum(len(item.peaks) for item in recordings)
        subject = "Test sequence" if tests else "Numpad training"
        detail = "Peak labels: press number : filename key, then peak time" if tests else "Numbered peaks are shown in the per-key detail images"
        fig.suptitle(
            f"{subject} recordings — {total} detected presses\n{detail}", fontsize=15,
        )
        fig.supxlabel("Time in recording (seconds)", fontsize=12)
        fig.supylabel("RMS amplitude (linear full-scale units)", fontsize=12)
        save_figure(fig, output, dpi)
    finally:
        plt.close(fig)


def detail(recording: Analysis, output: Path, dpi: int) -> None:
    windows = max(1, math.ceil(recording.duration / DETAIL_SECONDS))
    fig, axes = plt.subplots(
        windows, 1, figsize=(20, 2.8 * windows + 1.5),
        sharey=True, squeeze=False, constrained_layout=True,
    )
    try:
        for index, ax in enumerate(axes[:, 0]):
            start = index * DETAIL_SECONDS
            end = min((index + 1) * DETAIL_SECONDS, recording.duration)
            draw_panel(ax, recording, start, end, annotate=True)
            visible_peaks = (recording.times[recording.peaks] >= start) & (recording.times[recording.peaks] < end)
            ax.set_title(
                f"{start:.0f}–{end:.3f} seconds | {np.count_nonzero(visible_peaks)} presses in this window",
                fontsize=10, loc="left", pad=8,
            )
        axes[0, 0].set_ylim(0, max(float(np.max(recording.envelope)), recording.threshold, 1e-7) * 1.32)
        axes[0, 0].legend(loc="upper right", ncol=4, fontsize=8, framealpha=0.95)
        fig.suptitle(
            f"Key {recording.path.stem} — numbered press detections\n{recording.title}", fontsize=13,
        )
        fig.supxlabel("Time in original recording (seconds); press numbers continue across panels", fontsize=12)
        fig.supylabel("RMS amplitude (linear full-scale units)", fontsize=12)
        save_figure(fig, output, dpi)
    finally:
        plt.close(fig)


def visualize(
    dataset: Path,
    output: Path,
    dpi: int,
    threshold_db: float | None,
    tests_dir: Path | None = None,
) -> None:
    recordings = [analyze(path, threshold_db) for path in discover_recordings(dataset)]
    overview(recordings, output, dpi)
    for recording in recordings:
        detail(recording, output.parent / f"key_{recording.path.stem}_rms_peaks.png", dpi)
    if tests_dir is not None:
        tests = [analyze(path, threshold_db) for path in discover_recordings(tests_dir, sequences=True)]
        overview(tests, output.parent / "test_rms_peaks.png", dpi, tests=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=splitter.DEFAULT_DATASET,
        help="directory with single-digit training recordings (default: datasets/2_numpad)",
    )
    parser.add_argument(
        "--tests-dir", type=Path, default=splitter.DEFAULT_TESTS,
        help="directory with digit-sequence recordings (default: datasets/2_tests)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="overview PNG path; detail images use its folder")
    parser.add_argument("--dpi", type=int, default=180, help="image resolution (default: 180)")
    parser.add_argument(
        "--threshold-db", type=float,
        help="override the adaptive RMS detection threshold in dBFS, as in 1_keypresses.py",
    )
    args = parser.parse_args(argv)
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must have a .png extension")
    if args.threshold_db is not None and not (-120 <= args.threshold_db <= 0):
        parser.error("--threshold-db must be between -120 and 0")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            parser.error(f"{tool} is required on PATH")
    try:
        visualize(args.dataset_dir.resolve(), args.output.resolve(), args.dpi, args.threshold_db, args.tests_dir.resolve())
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
