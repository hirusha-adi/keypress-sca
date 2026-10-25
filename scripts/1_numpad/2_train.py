#!/usr/bin/env python3
"""Train and evaluate a numpad sound classifier with chronological held-out clips.

Outputs models, metrics, predictions, and a confusion matrix in work/1_numpad.
Only development clips participate in feature/hyperparameter selection. See the
local README for the limits of evaluation with one source recording per key.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import tempfile

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from keypress_model import (
    FEATURE_MODES,
    MODEL_SCHEMA_VERSION,
    FeatureConfig,
    build_classifier,
    extract_features,
    feature_variants,
)


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPOSITORY / "datasets" / "1_numpad"
DEFAULT_OUTPUT = REPOSITORY / "work" / "1_numpad"
CLASSES = tuple(str(key) for key in range(10))
C_VALUES = (1.0, 10.0)
CV_FOLDS = 4
HOLDOUT_FRACTION = 0.20
PURGE_NEIGHBORS = 1
LIMITATIONS = [
    "Each key has one source recording. Recording identity is confounded with the key label.",
    "Chronological held-out clips measure later presses within the same recordings; "
    "they do not measure generalization to a new recording session, phone, keyboard, or environment.",
    "The final keypress_model.joblib is refitted on all clips, including the evaluation holdout. "
    "Use evaluation_model.joblib to reproduce held-out predictions.",
    "SVM decision scores are ranking scores, not calibrated probabilities.",
]


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    ordinal: int


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_samples(dataset: Path) -> list[Sample]:
    """Order clips numerically within each class, rather than lexically."""
    samples = []
    for label in CLASSES:
        directory = dataset / label
        if not directory.is_dir():
            raise ValueError(f"missing class directory: {directory}")
        numbered = []
        for path in directory.glob("*.mp3"):
            if not path.is_file() or not path.stem.isascii() or not path.stem.isdecimal():
                continue
            ordinal = int(path.stem)
            if ordinal < 1:
                raise ValueError(f"clip numbers must start at 1: {path}")
            numbered.append(Sample(path, label, ordinal))
        numbered.sort(key=lambda sample: sample.ordinal)
        ordinals = [sample.ordinal for sample in numbered]
        if len(set(ordinals)) != len(ordinals):
            raise ValueError(f"duplicate numeric clip numbers in {directory}")
        if not numbered:
            raise ValueError(f"no numbered MP3 clips in {directory}; run 1_keypresses.py first")
        samples.extend(numbered)
    return samples


def holdout_split(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reserve each class's latest 20%, with one unused adjacent training clip."""
    development, purged, holdout = [], [], []
    for label in CLASSES:
        indices = [index for index, sample in enumerate(samples) if sample.label == label]
        indices.sort(key=lambda index: samples[index].ordinal)
        held_count = math.ceil(len(indices) * HOLDOUT_FRACTION)
        development_end = len(indices) - held_count - PURGE_NEIGHBORS
        if development_end < CV_FOLDS * 2:
            raise ValueError(
                f"key {label}: too few clips ({len(indices)}) for the chronological holdout "
                f"and {CV_FOLDS}-fold development validation; collect at least 12 per key"
            )
        development.extend(indices[:development_end])
        purged.extend(indices[development_end:len(indices) - held_count])
        holdout.extend(indices[len(indices) - held_count:])
    return tuple(np.asarray(indices, dtype=np.int64) for indices in (development, purged, holdout))


def development_folds(
    samples: list[Sample], development: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split each class into four contiguous blocks and purge adjacent presses."""
    folds = []
    for fold_index in range(CV_FOLDS):
        training, validation, purged = [], [], []
        for label in CLASSES:
            indices = np.asarray(
                sorted(
                    (int(index) for index in development if samples[index].label == label),
                    key=lambda index: samples[index].ordinal,
                ),
                dtype=np.int64,
            )
            block = np.array_split(indices, CV_FOLDS)[fold_index]
            if not len(block):
                raise ValueError(f"key {label}: an empty validation block was produced")
            first = int(np.flatnonzero(indices == block[0])[0])
            last = int(np.flatnonzero(indices == block[-1])[0]) + 1
            excluded = indices[max(0, first - PURGE_NEIGHBORS):min(len(indices), last + PURGE_NEIGHBORS)]
            purged_for_class = np.setdiff1d(excluded, block)
            training_for_class = np.setdiff1d(indices, excluded)
            if not len(training_for_class):
                raise ValueError(f"key {label}: no training clips remain after purging")
            training.extend(training_for_class.tolist())
            validation.extend(block.tolist())
            purged.extend(purged_for_class.tolist())
        folds.append(tuple(np.asarray(part, dtype=np.int64) for part in (training, validation, purged)))
    return folds


def source_provenance(dataset: Path) -> dict:
    manifest_path = dataset / "split_manifest.json"
    if not manifest_path.is_file():
        return {"split_manifest": None, "source_recordings": [], "note": "No splitter manifest was supplied."}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = []
    for recording in manifest.get("recordings", []):
        source_name = recording.get("source")
        if not isinstance(source_name, str) or Path(source_name).name != source_name:
            raise ValueError("invalid source filename in split_manifest.json")
        source = dataset / source_name
        actual = sha256(source) if source.is_file() else None
        expected = recording.get("source_sha256")
        if actual is not None and expected is not None and actual != expected:
            raise ValueError(f"{source_name}: source hash differs from split_manifest.json; regenerate clips")
        sources.append({
            "source": source_name,
            "manifest_sha256": expected,
            "actual_sha256": actual,
            "hash_verified": actual is not None and actual == expected,
        })
    return {"split_manifest": {"file": manifest_path.name, "sha256": sha256(manifest_path)},
            "source_recordings": sources}


def rank_classes(pipeline, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(pipeline.decision_function(features))
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.asarray(pipeline.classes_)[order], np.take_along_axis(scores, order, axis=1)


def make_metrics(truth: np.ndarray, predicted: np.ndarray, ranked: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=CLASSES, average="macro", zero_division=0)),
        "top3_accuracy": float(np.mean(np.any(ranked[:, :3] == truth[:, None], axis=1))),
        "correct": int(np.sum(truth == predicted)),
        "sample_count": len(truth),
        "classification_report": classification_report(
            truth, predicted, labels=CLASSES, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=CLASSES).tolist(),
    }


def save_confusion_matrix(matrix: np.ndarray, accuracy: float, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            value = int(matrix[row, column])
            ax.text(column, row, str(value), ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted key", ylabel="Actual key",
           title=f"Chronological held-out clips: {accuracy:.1%} accuracy\nSame source recordings; not an unseen-session test")
    fig.colorbar(image, ax=ax, label="Clips")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def train(dataset: Path, output: Path) -> dict:
    samples = discover_samples(dataset)
    development, boundary_purge, holdout = holdout_split(samples)
    folds = development_folds(samples, development)
    provenance = source_provenance(dataset)
    print(f"Dataset: {len(samples)} clips; development={len(development)}, "
          f"boundary purge={len(boundary_purge)}, holdout={len(holdout)}", flush=True)
    config = FeatureConfig()
    cache = {mode: {} for mode in FEATURE_MODES}
    for position, index in enumerate(development, 1):
        sample = samples[index]
        try:
            variants = feature_variants(sample.path, config)
        except (ValueError, OSError) as error:
            raise ValueError(f"{sample.path}: {error}") from error
        for mode in FEATURE_MODES:
            cache[mode][int(index)] = variants[mode]
        if position % 25 == 0 or position == len(development):
            print(f"Extracted development features: {position}/{len(development)}", flush=True)

    labels = np.asarray([sample.label for sample in samples])
    local_index = {int(index): position for position, index in enumerate(development)}
    candidates = []
    for mode in FEATURE_MODES:
        matrix = np.stack([cache[mode][int(index)] for index in development])
        for c_value in C_VALUES:
            oof_predictions = np.empty(len(development), dtype="<U1")
            fold_results = []
            for fold_index, (training, validation, purge) in enumerate(folds):
                local_train = [local_index[int(index)] for index in training]
                local_validation = [local_index[int(index)] for index in validation]
                pipeline = build_classifier(c_value)
                pipeline.fit(matrix[local_train], labels[training])
                predictions = pipeline.predict(matrix[local_validation])
                oof_predictions[local_validation] = predictions
                fold_results.append({
                    "fold": fold_index + 1,
                    "training_count": len(training),
                    "validation_count": len(validation),
                    "purged_count": len(purge),
                    "correct": int(np.sum(predictions == labels[validation])),
                    "accuracy": float(accuracy_score(labels[validation], predictions)),
                })
            correct = int(np.sum(oof_predictions == labels[development]))
            accuracy = correct / len(development)
            candidates.append({
                "mode": mode, "C": c_value, "correct": correct,
                "sample_count": len(development), "pooled_cv_accuracy": accuracy,
                "pooled_cv_macro_f1": float(f1_score(
                    labels[development], oof_predictions, labels=CLASSES, average="macro", zero_division=0
                )),
                "feature_count": int(matrix.shape[1]), "folds": fold_results,
            })
            print(f"Development CV: {mode}, C={c_value:g}: {accuracy:.1%} ({correct}/{len(development)})", flush=True)
    selected = min(candidates, key=lambda item: (-item["correct"], item["C"], FEATURE_MODES.index(item["mode"])))
    selected_config = replace(config, mode=selected["mode"])
    print(f"Selected {selected['mode']}, C={selected['C']:g}; evaluating held-out clips once", flush=True)
    chosen_cache = cache[selected["mode"]]
    features = []
    for index, sample in enumerate(samples):
        if index not in chosen_cache:
            try:
                chosen_cache[index] = extract_features(sample.path, selected_config)
            except (ValueError, OSError) as error:
                raise ValueError(f"{sample.path}: {error}") from error
        features.append(chosen_cache[index])
    matrix = np.stack(features)
    evaluation_model = build_classifier(selected["C"])
    evaluation_model.fit(matrix[development], labels[development])
    predictions = evaluation_model.predict(matrix[holdout])
    ranked, ranked_scores = rank_classes(evaluation_model, matrix[holdout])
    metrics = make_metrics(labels[holdout], predictions, ranked)
    print(f"Held-out accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{len(holdout)}); "
          f"top-3: {metrics['top3_accuracy']:.1%}; macro-F1: {metrics['macro_f1']:.3f}", flush=True)
    print("Refitting the selected configuration on all clips for deployment", flush=True)
    final_model = build_classifier(selected["C"])
    final_model.fit(matrix, labels)

    def names(indices):
        return [samples[int(index)].path.relative_to(dataset).as_posix() for index in indices]

    roles = {int(index): "development" for index in development}
    roles.update({int(index): "boundary_purge" for index in boundary_purge})
    roles.update({int(index): "holdout" for index in holdout})
    sample_metadata = [{
        "file": sample.path.relative_to(dataset).as_posix(), "key": sample.label,
        "ordinal": sample.ordinal, "sha256": sha256(sample.path), "split": roles[index],
    } for index, sample in enumerate(samples)]
    fingerprint = hashlib.sha256(json.dumps(sample_metadata, sort_keys=True).encode()).hexdigest()
    versions = {"python": platform.python_version()}
    for distribution in ("numpy", "scipy", "scikit-learn", "joblib", "matplotlib"):
        versions[distribution] = importlib.metadata.version(distribution)
    common_metadata = {
        "dataset_fingerprint": fingerprint,
        "dataset_directory": str(dataset),
        "selected_parameters": {"mode": selected["mode"], "C": selected["C"], "kernel": "rbf", "gamma": "scale"},
        "library_versions": versions,
        "limitations": LIMITATIONS,
        "heldout_accuracy": metrics["accuracy"],
        "heldout_macro_f1": metrics["macro_f1"],
    }
    evaluation_artifact = {
        "schema_version": MODEL_SCHEMA_VERSION, "pipeline": evaluation_model,
        "feature_config": asdict(selected_config), "classes": list(CLASSES),
        "training_metadata": {**common_metadata, "fit_split": "development", "training_sample_count": len(development)},
    }
    final_artifact = {
        "schema_version": MODEL_SCHEMA_VERSION, "pipeline": final_model,
        "feature_config": asdict(selected_config), "classes": list(CLASSES),
        "training_metadata": {**common_metadata, "fit_split": "all_clips", "training_sample_count": len(samples)},
    }
    report = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "classes": list(CLASSES), "dataset_fingerprint": fingerprint,
        "sample_count": len(samples),
        "class_counts": {label: int(np.sum(labels == label)) for label in CLASSES},
        "feature_config": asdict(selected_config), "selected_candidate": selected,
        "selection_rule": "Highest pooled development CV accuracy; ties prefer lower C, then FEATURE_MODES order.",
        "candidate_results": candidates,
        "split_protocol": {
            "description": "Latest ceil(20%) per key held out chronologically; one preceding clip per key purged. "
            "Four contiguous development validation blocks per key, each with one adjacent clip on both sides purged from training.",
            "holdout_fraction": HOLDOUT_FRACTION, "purge_neighbors": PURGE_NEIGHBORS,
            "development_count": len(development), "boundary_purge_count": len(boundary_purge), "holdout_count": len(holdout),
            "development": names(development), "boundary_purge": names(boundary_purge), "holdout": names(holdout),
            "development_folds": [{"fold": i + 1, "training": names(training), "validation": names(validation),
                                   "purged": names(purge)} for i, (training, validation, purge) in enumerate(folds)],
        },
        "heldout_metrics": metrics, "samples": sample_metadata,
        "provenance": provenance, "library_versions": versions,
        "final_model_training_sample_count": len(samples),
        "evaluation_model_training_sample_count": len(development),
        "limitations": LIMITATIONS,
    }
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".numpad-training-", dir=output) as temporary:
        staging = Path(temporary)
        joblib.dump(evaluation_artifact, staging / "evaluation_model.joblib", compress=3)
        joblib.dump(final_artifact, staging / "keypress_model.joblib", compress=3)
        (staging / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with (staging / "heldout_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "file", "actual_key", "predicted_key", "correct", "top1_key", "top1_score",
                "top2_key", "top2_score", "top3_key", "top3_score",
            ])
            writer.writeheader()
            for position, index in enumerate(holdout):
                row = {"file": names([index])[0], "actual_key": samples[index].label,
                       "predicted_key": str(predictions[position]),
                       "correct": bool(predictions[position] == samples[index].label)}
                for rank in range(3):
                    row[f"top{rank + 1}_key"] = str(ranked[position, rank])
                    row[f"top{rank + 1}_score"] = float(ranked_scores[position, rank])
                writer.writerow(row)
        save_confusion_matrix(np.asarray(metrics["confusion_matrix"]), metrics["accuracy"], staging / "confusion_matrix.png")
        for path in staging.iterdir():
            path.replace(output / path.name)
    print(f"Saved models and evaluation artifacts to {output}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET,
                        help="directory containing numbered MP3 folders 0 through 9")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="directory for models and evaluation artifacts")
    args = parser.parse_args(argv)
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            parser.error(f"{tool} is required on PATH")
    try:
        train(args.dataset_dir.resolve(), args.output_dir.resolve())
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
