#!/usr/bin/env python3
"""Train and evaluate a numpad sound classifier with chronological held-out clips.

Outputs models, metrics, predictions, and a confusion matrix in work/1_numpad.
Only development clips participate in feature/hyperparameter selection. See the
local README for the limits of evaluation with one source recording per key.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
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
from threadpoolctl import threadpool_limits
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from keypress_model import (
    MODEL_SCHEMA_VERSION,
    FeatureConfig,
    build_classifier,
    decode_audio,
    feature_variants_from_waveform,
)


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPOSITORY / "datasets" / "1_numpad"
DEFAULT_OUTPUT = REPOSITORY / "work" / "1_numpad"
CLASSES = tuple(str(key) for key in range(10))
CV_FOLDS = 4
CV_PARTITIONS = (3, 4, 5)
HOLDOUT_FRACTION = 0.20
PURGE_NEIGHBORS = 1
RANDOM_SEED = 42
SELECTION_RULE = (
    "Highest pooled repeated development CV accuracy, then highest worst-fold accuracy, "
    "then fewer features; remaining ties prefer linear, LDA, RBF, lower C, "
    "the default gamma, and stable profile order."
)
LIMITATIONS = [
    "Each key has one source recording. Recording identity is confounded with the key label.",
    "Chronological held-out clips measure later presses within the same recordings; "
    "they do not measure generalization to a new recording session, phone, keyboard, or environment.",
    "The final keypress_model.joblib is refitted on all clips, including the evaluation holdout. "
    "Use evaluation_model.joblib to reproduce held-out predictions.",
    "Classifier decision scores are ranking scores, not calibrated probabilities.",
    "Repeated CV reuses development clips across three partitions. Its prediction count "
    "is not a count of independent recordings or unique clips.",
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
    samples: list[Sample], development: np.ndarray, n_splits: int = CV_FOLDS
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split each class into contiguous blocks and purge adjacent presses."""
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    folds = []
    for fold_index in range(n_splits):
        training, validation, purged = [], [], []
        for label in CLASSES:
            indices = np.asarray(
                sorted(
                    (int(index) for index in development if samples[index].label == label),
                    key=lambda index: samples[index].ordinal,
                ),
                dtype=np.int64,
            )
            block = np.array_split(indices, n_splits)[fold_index]
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


def repeated_development_folds(samples: list[Sample], development: np.ndarray) -> list[dict]:
    """Keep each validation clip contiguous within its source class in all repeats."""
    return [
        {"n_splits": n_splits, "fold": fold_index + 1,
         "training": training, "validation": validation, "purged": purged}
        for n_splits in CV_PARTITIONS
        for fold_index, (training, validation, purged) in enumerate(
            development_folds(samples, development, n_splits)
        )
    ]


def feature_profiles() -> list[tuple[str, FeatureConfig]]:
    return [
        ("legacy_mfcc20", FeatureConfig(mode="mfcc_stats", n_mfcc=20)),
        ("weighted_mfcc20", FeatureConfig(mode="mfcc_weighted", n_mfcc=20)),
        ("weighted_mfcc30", FeatureConfig(mode="mfcc_weighted", n_mfcc=30)),
        ("weighted_mfcc30_sqrt", FeatureConfig(
            mode="mfcc_weighted", n_mfcc=30, energy_weight_power=0.5
        )),
        ("weighted_mfcc40_sqrt", FeatureConfig(
            mode="mfcc_weighted", n_mfcc=40, energy_weight_power=0.5
        )),
        ("short_mfcc30", FeatureConfig(
            mode="mfcc_stats", n_mfcc=30, duration_seconds=0.096,
            pre_peak_seconds=0.008, hop_length=64
        )),
    ]


def classifier_candidates(dimensions: int) -> list[dict]:
    """Twelve fixed classifier settings per profile; no test-score feedback."""
    return (
        [{"kind": "rbf", "C": c_value, "gamma": gamma}
         for c_value in (1.0, 10.0) for gamma in ("scale", 0.3 / dimensions, 3.0 / dimensions)]
        + [{"kind": "linear", "C": c_value} for c_value in (0.001, 0.01, 0.1)]
        + [{"kind": "lda", "shrinkage": shrinkage} for shrinkage in (0.1, 0.5, "auto")]
    )


def candidate_pipeline(settings: dict):
    return build_classifier(
        settings.get("C", 1.0), kind=settings["kind"],
        gamma=settings.get("gamma", "scale"), shrinkage=settings.get("shrinkage", 0.5),
    )


def candidate_selection_key(candidate: dict) -> tuple:
    """Deterministic accuracy/robustness/simplicity ordering for model selection."""
    settings = candidate["classifier"]
    gamma = settings.get("gamma", "scale")
    return (
        -candidate["pooled_cv_accuracy"], -candidate["worst_fold_accuracy"],
        candidate["feature_count"], {"linear": 0, "lda": 1, "rbf": 2}[settings["kind"]],
        settings.get("C", 0.0), gamma != "scale", 0.0 if gamma == "scale" else gamma,
        candidate["profile_order"], str(settings.get("shrinkage", "")),
    )


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
    folds = repeated_development_folds(samples, development)
    provenance = source_provenance(dataset)
    print(f"Dataset: {len(samples)} clips; development={len(development)}, "
          f"boundary purge={len(boundary_purge)}, holdout={len(holdout)}", flush=True)
    profiles = feature_profiles()
    configs = dict(profiles)
    cache = {name: {} for name, _ in profiles}
    decoded = {}
    for position, index in enumerate(development, 1):
        sample = samples[index]
        try:
            waveform = decode_audio(sample.path, profiles[0][1].sample_rate)
            decoded[int(index)] = waveform
            for name, config in profiles:
                cache[name][int(index)] = feature_variants_from_waveform(waveform, config)[config.mode]
        except (ValueError, OSError) as error:
            raise ValueError(f"{sample.path}: {error}") from error
        if position % 25 == 0 or position == len(development):
            print(f"Extracted development features: {position}/{len(development)}", flush=True)

    labels = np.asarray([sample.label for sample in samples])
    local_index = {int(index): position for position, index in enumerate(development)}
    candidates = []
    repeated_truth = np.tile(labels[development], len(CV_PARTITIONS))
    with threadpool_limits(limits=1):
        for profile_order, (name, config) in enumerate(profiles):
            matrix = np.stack([cache[name][int(index)] for index in development])
            for settings in classifier_candidates(matrix.shape[1]):
                partition_predictions = {
                    n_splits: np.empty(len(development), dtype=labels.dtype)
                    for n_splits in CV_PARTITIONS
                }
                fold_results = []
                for fold in folds:
                    training, validation, purge = fold["training"], fold["validation"], fold["purged"]
                    local_train = [local_index[int(index)] for index in training]
                    local_validation = [local_index[int(index)] for index in validation]
                    pipeline = candidate_pipeline(settings)
                    pipeline.fit(matrix[local_train], labels[training])
                    predictions = pipeline.predict(matrix[local_validation])
                    partition_predictions[fold["n_splits"]][local_validation] = predictions
                    fold_results.append({
                        "n_splits": fold["n_splits"], "fold": fold["fold"],
                        "training_count": len(training),
                        "validation_count": len(validation),
                        "purged_count": len(purge),
                        "correct": int(np.sum(predictions == labels[validation])),
                        "accuracy": float(accuracy_score(labels[validation], predictions)),
                    })
                oof_predictions = np.concatenate([partition_predictions[n] for n in CV_PARTITIONS])
                correct = int(np.sum(oof_predictions == repeated_truth))
                accuracy = correct / len(repeated_truth)
                worst = min(fold["accuracy"] for fold in fold_results)
                candidates.append({
                    "profile": name, "profile_order": profile_order,
                    "mode": config.mode, "feature_config": asdict(config),
                    "classifier": settings, "kind": settings["kind"],
                    **({key: value for key, value in settings.items() if key != "kind"}),
                    "correct": correct, "sample_count": len(repeated_truth),
                    "validation_prediction_count": len(repeated_truth),
                    "unique_sample_count": len(development),
                    "pooled_cv_accuracy": accuracy, "worst_fold_accuracy": worst,
                    "pooled_cv_macro_f1": float(f1_score(
                        repeated_truth, oof_predictions, labels=CLASSES, average="macro", zero_division=0
                    )),
                    "feature_count": int(matrix.shape[1]), "folds": fold_results,
                    "partitions": [{
                        "n_splits": n_splits,
                        "correct": int(np.sum(partition_predictions[n_splits] == labels[development])),
                        "validation_prediction_count": len(development),
                        "accuracy": float(accuracy_score(labels[development], partition_predictions[n_splits])),
                    } for n_splits in CV_PARTITIONS],
                })
                print(f"Development CV: {name}, {settings}: {accuracy:.2%} "
                      f"({correct}/{len(repeated_truth)} predictions; {len(development)} unique clips), "
                      f"worst fold {worst:.2%}", flush=True)
    selected = min(candidates, key=candidate_selection_key)
    selected_config = configs[selected["profile"]]
    print(f"Selected {selected['profile']}, {selected['classifier']}; evaluating held-out clips once", flush=True)
    chosen_cache = cache[selected["profile"]]
    features = []
    for index, sample in enumerate(samples):
        if index not in chosen_cache:
            try:
                if index not in decoded:
                    decoded[index] = decode_audio(sample.path, selected_config.sample_rate)
                chosen_cache[index] = feature_variants_from_waveform(
                    decoded[index], selected_config
                )[selected_config.mode]
            except (ValueError, OSError) as error:
                raise ValueError(f"{sample.path}: {error}") from error
        features.append(chosen_cache[index])
    matrix = np.stack(features)
    with threadpool_limits(limits=1):
        evaluation_model = candidate_pipeline(selected["classifier"])
        evaluation_model.fit(matrix[development], labels[development])
        predictions = evaluation_model.predict(matrix[holdout])
        ranked, ranked_scores = rank_classes(evaluation_model, matrix[holdout])
    metrics = make_metrics(labels[holdout], predictions, ranked)
    print(f"Held-out accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{len(holdout)}); "
          f"top-3: {metrics['top3_accuracy']:.1%}; macro-F1: {metrics['macro_f1']:.3f}", flush=True)
    print("Refitting the selected configuration on all clips for deployment", flush=True)
    with threadpool_limits(limits=1):
        final_model = candidate_pipeline(selected["classifier"])
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
    implementation_hashes = {
        name: sha256(Path(__file__).with_name(name))
        for name in ("2_train.py", "keypress_model.py")
    }
    common_metadata = {
        "dataset_fingerprint": fingerprint,
        "dataset_directory": str(dataset),
        "selected_parameters": {
            "profile": selected["profile"], "mode": selected["mode"], **selected["classifier"]
        },
        "library_versions": versions,
        "implementation_sha256": implementation_hashes,
        "random_seed": RANDOM_SEED,
        "selection_rule": SELECTION_RULE,
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
        "selection_rule": SELECTION_RULE,
        "candidate_results": candidates,
        "search": {
            "feature_profiles": [{"name": name, "config": asdict(config)} for name, config in profiles],
            "candidate_count": len(candidates),
            "partition_fold_counts": list(CV_PARTITIONS), "fold_count": len(folds),
            "validation_prediction_count": len(repeated_truth),
            "unique_validation_clip_count": len(development),
            "prediction_count_note": "Each development clip is evaluated once per partition. "
            "The repeated prediction denominator is not a count of independent audio clips.",
            "random_seed": RANDOM_SEED, "shuffle": False, "threadpool_limit": 1,
            "external_test_data_used": False,
        },
        "split_protocol": {
            "description": "Latest ceil(20%) per key held out chronologically; one preceding clip per key purged. "
            "Development validation repeats contiguous 3-, 4-, and 5-fold partitions per key, "
            "each with one adjacent clip on both sides purged from training.",
            "holdout_fraction": HOLDOUT_FRACTION, "purge_neighbors": PURGE_NEIGHBORS,
            "development_count": len(development), "boundary_purge_count": len(boundary_purge), "holdout_count": len(holdout),
            "development": names(development), "boundary_purge": names(boundary_purge), "holdout": names(holdout),
            "development_folds": [{
                "n_splits": fold["n_splits"], "fold": fold["fold"],
                "training": names(fold["training"]), "validation": names(fold["validation"]),
                "purged": names(fold["purged"]),
            } for fold in folds],
        },
        "heldout_metrics": metrics, "samples": sample_metadata,
        "provenance": provenance, "library_versions": versions,
        "implementation_sha256": implementation_hashes,
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
