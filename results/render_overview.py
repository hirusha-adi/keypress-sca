"""Render the showcase comparison from the committed experiment reports.

Run from the repository root: .venv/bin/python results/render_overview.py
This redraws the figure only; it does not rerun or change any experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).resolve().parent


def read_report(relative_path: str) -> dict:
    return json.loads((RESULTS / relative_path).read_text())


def main() -> None:
    phone = read_report("1_numpad/model_comparison.json")
    microphone = [
        read_report(f"2_numpad/{name}_evaluation.json")
        for name in ("baseline", "neural", "hybrid")
    ]
    phone_total = phone["sample_counts"]["external"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(12.8, 5.8), gridspec_kw={"wspace": 0.26}
    )
    figure.set_facecolor("#f7f9fc")
    panels = [
        (
            axes[0],
            ["Baseline", "Weighted MFCC + LDA"],
            [100 * phone[name]["external_accuracy"] for name in ("baseline", "improved")],
            [f"{phone[name]['external_correct']}/{phone_total}" for name in ("baseline", "improved")],
            "Phone recordings · 10 keys",
            "Previously observed external recording",
        ),
        (
            axes[1],
            ["Linear SVM", "Three Swin models", "Neural + spectral"],
            [100 * report["accuracy"] for report in microphone],
            [f"{report['correct']}/{report['sample_count']}" for report in microphone],
            "USB microphone · 4 keys",
            "Development recording reused during tuning",
        ),
    ]
    for axis, names, values, counts, title, subtitle in panels:
        axis.set_facecolor("#f7f9fc")
        positions = np.arange(len(names))
        bars = axis.bar(
            positions,
            values,
            color=["#74869c", "#4969bb", "#19a085"][: len(names)],
            width=0.58,
            zorder=3,
        )
        axis.set_xticks(positions, names, fontsize=10)
        axis.set_ylim(0, 118)
        ticks = [0, 25, 50, 75, 100]
        axis.set_yticks(ticks, [f"{value}%" for value in ticks])
        axis.grid(axis="y", alpha=0.2, zorder=0)
        axis.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=36)
        axis.text(
            0, 1.04, subtitle, transform=axis.transAxes, fontsize=10, color="#53637c"
        )
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_color("#c4ceda")
        axis.tick_params(axis="y", length=0)
        for bar, value, count in zip(bars, values, counts):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 3,
                f"{value:g}%\n({count})",
                ha="center",
                fontsize=11,
                fontweight="bold",
                color="#26364d",
            )
    axes[0].set_ylabel("Correct key predictions")
    figure.suptitle(
        "Acoustic keypress recognition: the experiments",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#1a2940",
    )
    figure.text(
        0.08,
        0.035,
        "Different datasets and benchmark sizes; these bars are not a controlled comparison.\n"
        "The 100% result is 20 development presses, not a fresh live accuracy estimate.",
        fontsize=10,
        color="#53637c",
        linespacing=1.6,
    )
    figure.subplots_adjust(top=0.76, bottom=0.22, left=0.08, right=0.97)
    figure.savefig(RESULTS / "overview.png", dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
