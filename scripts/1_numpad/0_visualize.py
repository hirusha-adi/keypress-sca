#!/usr/bin/env python3
"""Plot RMS envelopes, numbered press peaks, and percentiles for numpad recordings.

The default output is work/1_numpad/rms_peaks.png. Audio analysis uses the same
detector as 1_keypresses.py, without requiring a valid press count or quiet cuts.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")  # Save images without needing a desktop/display server.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


# Python's import statement cannot name a module starting with a digit. Load the
# sibling script by its path so plotting and splitting share the exact detector.
SPEC = importlib.util.spec_from_file_location(
    "numpad_splitter", Path(__file__).with_name("1_keypresses.py")
)
splitter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = splitter
SPEC.loader.exec_module(splitter)

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "work" / "1_numpad" / "rms_peaks.png"
PERCENTILES = (10, 20, 50, 80, 95)


def visualize(dataset: Path, output: Path, dpi: int, threshold_db: float | None) -> None:
    fig, axes = plt.subplots(
        10, 1, figsize=(20, 25), sharey=True, constrained_layout=True
    )
    total = 0
    maximum_rms = 0.0
    try:
        for key, ax in enumerate(axes):
            path = dataset / f"{key}.m4a"
            samples, sample_rate = splitter.decode(path)
            frame_size = max(1, round(splitter.FRAME_SECONDS * sample_rate))
            envelope = splitter.rms_envelope(samples, frame_size)
            peaks, threshold, noise_floor = splitter.detect_presses(
                envelope, frame_size / sample_rate, threshold_db
            )
            # Use each frame's true center, including a shorter final frame.
            starts = np.arange(len(envelope)) * frame_size
            ends = np.minimum(starts + frame_size, len(samples))
            times = (starts + ends) / (2 * sample_rate)
            percentiles = np.percentile(envelope, PERCENTILES)
            percentile_text = "   ".join(
                f"P{percentile}={value:.5f}"
                for percentile, value in zip(PERCENTILES, percentiles)
            )
            db = 20 * math.log10(threshold)
            duration = len(samples) / sample_rate
            title = (
                f"{path.name} | {len(peaks)} detected presses | {duration:.3f} s | "
                f"{sample_rate:,} Hz / {samples.shape[1]} ch | threshold {db:.1f} dBFS\n"
                f"RMS percentiles: {percentile_text}"
            )
            ax.set_title(title, fontsize=10, loc="left", pad=8)
            ax.plot(times, envelope, color="#2672a9", linewidth=0.8, label="5 ms RMS")
            ax.scatter(
                times[peaks], envelope[peaks], marker="x", s=24,
                linewidths=1.1, color="#c82f35", zorder=3, label="Detected press",
            )
            for number, peak in enumerate(peaks, 1):
                ax.annotate(
                    str(number), (times[peak], envelope[peak]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#a31f26",
                )
            ax.axhline(
                threshold, color="#bc801a", linewidth=0.9, linestyle="--",
                label="Detection threshold",
            )
            ax.axhline(
                noise_floor, color="#777777", linewidth=0.8, linestyle=":",
                label="P20 noise estimate",
            )
            ax.set_xlim(0, duration)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", color="#dddddd", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            maximum_rms = max(maximum_rms, float(np.max(envelope)), threshold)
            total += len(peaks)
            print(f"{path.name}: {len(peaks)} presses | {percentile_text}", flush=True)
        # One shared scale makes amplitudes comparable across the ten keys.
        axes[0].set_ylim(0, max(maximum_rms * 1.35, 1e-7))
        axes[0].legend(loc="upper right", ncol=4, fontsize=8, framealpha=0.95)
        fig.suptitle(
            f"Numpad keystroke recordings — {total} detected presses\n"
            "RMS across all channels · source audio before output gain · press and release grouped",
            fontsize=16,
        )
        fig.supxlabel("Time in recording (seconds)", fontsize=12)
        fig.supylabel("RMS amplitude (linear full-scale units)", fontsize=12)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, facecolor="white")
        print(f"Saved {output}")
    finally:
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=splitter.DEFAULT_DATASET,
        help="directory containing 0.m4a through 9.m4a (default: repository datasets/1_numpad)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG output path")
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
    dataset = args.dataset_dir.resolve()
    missing = [f"{key}.m4a" for key in range(10) if not (dataset / f"{key}.m4a").is_file()]
    if missing:
        parser.error(f"missing recordings in {dataset}: {', '.join(missing)}")
    try:
        visualize(dataset, args.output.resolve(), args.dpi, args.threshold_db)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
