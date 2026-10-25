#!/usr/bin/env python3
"""Predict the numpad key (0-9) from one isolated keypress audio file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from keypress_model import DEFAULT_MODEL, load_model, predict_key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="a single-keypress MP3, M4A, WAV, or other FFmpeg-readable file")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="trained .joblib model path")
    parser.add_argument("--top-k", type=int, default=3, help="number of ranked keys to show (default: 3)")
    parser.add_argument("--json", action="store_true", dest="json_output", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    if not (1 <= args.top_k <= 10):
        parser.error("--top-k must be between 1 and 10")
    try:
        result = predict_key(args.audio, load_model(args.model), args.top_k)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        print(f"Predicted key: {result['predicted_key']}")
        print("Ranked SVM scores (not probabilities):")
        for entry in result["ranked_keys"]:
            print(f"  {entry['key']}: {entry['score']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
