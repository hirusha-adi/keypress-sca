#!/usr/bin/env python3
"""Evaluate a frozen numpad model on labeled, split test recordings.

Expected input: datasets/1_tests/<key-sequence>/<press-number>-<actual-key>.mp3.
Reports are separated by model and test-data fingerprints; evaluation never fits
or changes the model. Classifier decision scores are uncalibrated rankings,
not probabilities.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile

from keypress_model import DEFAULT_MODEL, DEFAULT_WORK, ROOT, load_model, predict_key


DEFAULT_TESTS = ROOT / "datasets" / "1_tests"
DEFAULT_OUTPUT = DEFAULT_WORK / "test_results"
CLIP_PATTERN = re.compile(r"([1-9][0-9]*)-([0-9])\.mp3")
CSV_FIELDS = [
    "sequence", "press_number", "file", "actual_key", "predicted_key", "correct", "top3_correct",
    "top1_key", "top1_score", "top2_key", "top2_score", "top3_key", "top3_score", "sha256",
]


@dataclass(frozen=True)
class TestClip:
    path: Path
    press_number: int
    actual_key: str


@dataclass(frozen=True)
class TestRecording:
    sequence: str
    clips: tuple[TestClip, ...]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_recordings(tests_dir: Path) -> list[TestRecording]:
    """Validate complete press sequences before any prediction is attempted."""
    if not tests_dir.is_dir():
        raise ValueError(f"Test directory does not exist: {tests_dir}")
    recordings = []
    for directory in sorted(tests_dir.iterdir()):
        sequence = directory.name
        if not directory.is_dir() or not sequence.isascii() or not sequence.isdecimal():
            continue
        clips = []
        seen = set()
        for path in directory.glob("*.mp3"):
            match = CLIP_PATTERN.fullmatch(path.name)
            if not path.is_file() or match is None:
                raise ValueError(f"Invalid test clip name: {path}; expected PRESS-ACTUAL_KEY.mp3")
            press_number, actual_key = int(match[1]), match[2]
            if press_number in seen:
                raise ValueError(f"Duplicate press number {press_number} in {directory}")
            seen.add(press_number)
            clips.append(TestClip(path, press_number, actual_key))
        if seen != set(range(1, len(sequence) + 1)):
            raise ValueError(f"{sequence}: clip press numbers must cover 1 through {len(sequence)} exactly")
        clips.sort(key=lambda clip: clip.press_number)
        for clip in clips:
            if clip.actual_key != sequence[clip.press_number - 1]:
                raise ValueError(f"{clip.path}: actual key does not match the source sequence {sequence}")
        recordings.append(TestRecording(sequence, tuple(clips)))
    if not recordings:
        raise ValueError(f"No split test recordings in {tests_dir}; run 1_keypresses.py --only tests first")
    return recordings


def validate_provenance(tests_dir: Path, recordings: list[TestRecording]) -> dict:
    """Check labels against a splitter manifest when one is present."""
    manifest_path = tests_dir / "split_manifest.json"
    manifest_records = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("recordings", []):
            source = item.get("source")
            if not isinstance(source, str) or Path(source).name != source:
                raise ValueError("Invalid source filename in split_manifest.json")
            sequence = Path(source).stem
            if source != f"{sequence}.m4a" or item.get("key_sequence") != sequence:
                raise ValueError(f"{source}: manifest source and key sequence disagree")
            if sequence in manifest_records:
                raise ValueError(f"Duplicate sequence {sequence} in split_manifest.json")
            manifest_records[sequence] = item
        if set(manifest_records) != {recording.sequence for recording in recordings}:
            raise ValueError("Test sequence folders do not match split_manifest.json")
    provenance = {"split_manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
                  "recordings": {}}
    for recording in recordings:
        sequence = recording.sequence
        source = tests_dir / f"{sequence}.m4a"
        actual_hash = sha256(source) if source.is_file() else None
        item = manifest_records.get(sequence)
        expected_hash = item.get("source_sha256") if item else None
        if actual_hash is not None and expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(f"{source.name}: source hash differs from split_manifest.json")
        if item is not None:
            clips = item.get("clips", [])
            if item.get("detected_presses") != len(recording.clips) or len(clips) != len(recording.clips):
                raise ValueError(f"{sequence}: manifest press count does not match its clips")
            for actual, declared in zip(recording.clips, clips):
                expected_file = f"{sequence}/{actual.path.name}"
                if (declared.get("file") != expected_file
                        or declared.get("press_number") != actual.press_number
                        or declared.get("key") != actual.actual_key):
                    raise ValueError(f"{expected_file}: clip labels/order differ from split_manifest.json")
        provenance["recordings"][sequence] = {
            "source": source.name,
            "source_present": source.is_file(),
            "source_sha256": actual_hash,
            "manifest_source_sha256": expected_hash,
            "source_hash_verified": actual_hash is not None and actual_hash == expected_hash,
        }
    return provenance


def metrics(rows: list[dict]) -> dict:
    count = len(rows)
    correct = sum(row["correct"] for row in rows)
    top3_correct = sum(row["top3_correct"] for row in rows)
    return {"sample_count": count, "correct": correct, "accuracy": correct / count,
            "top3_correct": top3_correct, "top3_accuracy": top3_correct / count}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            flat = {field: row[field] for field in CSV_FIELDS if field in row}
            for rank, item in enumerate(row["ranked_keys"], 1):
                flat[f"top{rank}_key"] = item["key"]
                flat[f"top{rank}_score"] = item["score"]
            writer.writerow(flat)


def evaluate(tests_dir: Path, model_path: Path = DEFAULT_MODEL, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    tests_dir, model_path, output_dir = tests_dir.resolve(), model_path.resolve(), output_dir.resolve()
    recordings = discover_recordings(tests_dir)
    provenance = validate_provenance(tests_dir, recordings)
    model_hash = sha256(model_path)
    artifact = load_model(model_path)
    rows = []
    recording_reports = []
    for recording in recordings:
        predictions = []
        for clip in recording.clips:
            clip_hash = sha256(clip.path)
            result = predict_key(clip.path, artifact, top_k=3)
            ranked = result["ranked_keys"]
            if (len(ranked) != 3 or len({item["key"] for item in ranked}) != 3
                    or any(item["key"] not in "0123456789" or len(item["key"]) != 1
                           or not math.isfinite(item["score"]) for item in ranked)
                    or result["predicted_key"] != ranked[0]["key"]):
                raise ValueError(f"{clip.path.name}: model returned invalid prediction rankings")
            if sha256(clip.path) != clip_hash:
                raise ValueError(f"{clip.path}: audio changed during evaluation")
            row = {
                "sequence": recording.sequence,
                "press_number": clip.press_number,
                "file": clip.path.relative_to(tests_dir).as_posix(),
                "actual_key": clip.actual_key,
                "predicted_key": result["predicted_key"],
                "correct": clip.actual_key == result["predicted_key"],
                "top3_correct": clip.actual_key in [item["key"] for item in ranked],
                "ranked_keys": ranked,
                "sha256": clip_hash,
            }
            predictions.append(row)
            rows.append(row)
        recording_report = {
            "schema_version": 1, "test_sequence": recording.sequence,
            "expected_sequence": recording.sequence,
            "predicted_sequence": "".join(row["predicted_key"] for row in predictions),
            **metrics(predictions), **provenance["recordings"][recording.sequence],
            "predictions": predictions,
        }
        recording_reports.append(recording_report)
        print(f"{recording.sequence}: {recording_report['correct']}/{recording_report['sample_count']} "
              f"({recording_report['accuracy']:.1%}); predicted {recording_report['predicted_sequence']}", flush=True)
    if sha256(model_path) != model_hash:
        raise ValueError("Model file changed during evaluation; no reports were saved")
    fingerprint_input = {"clips": [{key: row[key] for key in ("file", "actual_key", "sha256")} for row in rows],
                         "provenance": provenance}
    data_hash = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()
    run_dir = output_dir / f"model-{model_hash[:16]}" / f"data-{data_hash[:16]}"
    common = {
        "model": str(model_path), "model_sha256": model_hash,
        "model_training_sample_count": artifact.get("training_metadata", {}).get("training_sample_count"),
        "evaluation_protocol": "Frozen model; no fitting, retraining, or parameter selection performed.",
        "score_type": "Uncalibrated classifier decision score; not a probability",
        "tests_directory": str(tests_dir), "test_data_fingerprint": data_hash,
        "split_manifest_sha256": provenance["split_manifest_sha256"],
    }
    for item in recording_reports:
        item.update(common)
    report = {"schema_version": 1, **common, **metrics(rows),
              "recordings": recording_reports, "report_directory": str(run_dir)}
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".numpad-evaluation-", dir=output_dir) as temporary:
        staging = Path(temporary)
        (staging / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        write_csv(staging / "predictions.csv", rows)
        for item in recording_reports:
            (staging / f"{item['test_sequence']}.json").write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
            write_csv(staging / f"{item['test_sequence']}.csv", item["predictions"])
        run_dir.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            path.replace(run_dir / path.name)
        latest = staging / "latest.json"
        latest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        latest.replace(output_dir / "latest.json")
    print(f"Overall: {report['correct']}/{report['sample_count']} ({report['accuracy']:.1%}); "
          f"top-3 {report['top3_accuracy']:.1%}", flush=True)
    print(f"Saved evaluation to {run_dir}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="frozen trained .joblib model")
    parser.add_argument("--tests-dir", type=Path, default=DEFAULT_TESTS, help="directory containing split sequence folders")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="parent directory for fingerprinted reports")
    args = parser.parse_args(argv)
    try:
        evaluate(args.tests_dir, args.model, args.output_dir)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
