"""Train a large pretrained spectrogram ensemble using every supplied clip.

No test directory, holdout, validation split, or model search is used here.
Epoch metrics describe augmented training examples, never test accuracy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time
from typing import Sequence

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deep_keypress import (
    ARCHITECTURES,
    DeepConfig,
    SpectrogramFrontend,
    create_network,
    prepare_waveform,
    shared_feature_config,
    state_to_numpy,
    validate_artifact,
)
from keypress_model import decode_audio


DEFAULT_ARCHITECTURES = ("swin_b", "swin_t", "swin_t")
CLASSES = ("0", "1", "2", "3")
BASE_SEED = 42
TRAINING_RECIPE = {
    "optimizer": "AdamW", "backbone_learning_rate": 5e-5,
    "head_learning_rate": 5e-4, "weight_decay": 0.01,
    "label_smoothing": 0.03, "warmup_epochs": 5,
    "schedule": "cosine", "target_effective_batch_size": 32,
    "gradient_clip_norm": 1.0,
    "member_combination": "equal_weight_mean_probabilities",
    "external_test_data_used": False,
}
LIMITATIONS = [
    "Each key has one source recording; recording conditions can be confounded with the label.",
    "All supplied clips are used for fitting, with no internal validation or test split.",
    "Epoch accuracy describes augmented training examples and is not generalization accuracy.",
    "Ensemble softmax probabilities are uncalibrated confidence, not measured accuracy.",
    "A separate labeled recording from the live microphone is required to measure live accuracy.",
]
RESEARCH_SOURCES = [
    "https://arxiv.org/html/2308.01074v1",
    "https://www.usenix.org/system/files/woot25-ayati.pdf",
    "https://docs.pytorch.org/vision/main/models/generated/torchvision.models.swin_t.html",
    "https://docs.pytorch.org/vision/main/models/generated/torchvision.models.swin_b.html",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value: dict | list) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _atomic_joblib(value: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        joblib.dump(value, temporary, compress=3)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _sample_metadata(dataset: Path, samples: Sequence) -> list[dict]:
    metadata = [{
        "file": Path(sample.path).relative_to(dataset).as_posix(),
        "key": str(sample.label), "ordinal": int(sample.ordinal),
        "sha256": sha256(Path(sample.path)), "split": "training",
    } for sample in samples]
    if not metadata or set(item["key"] for item in metadata) != set(CLASSES):
        raise ValueError("Deep training requires at least one clip for each key 0, 1, 2, and 3")
    names = [item["file"] for item in metadata]
    if len(names) != len(set(names)):
        raise ValueError("Deep training received duplicate clip paths")
    return metadata


def _prepared_audio(
    dataset: Path, output: Path, samples: Sequence, metadata: list[dict], config: DeepConfig,
    implementation: dict,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Cache only deterministic training preprocessing, keyed by every clip hash."""
    cache_hash = fingerprint({
        "clips": metadata, "deep_config": asdict(config),
        "preprocessing_implementation_sha256": implementation["deep_keypress.py"],
        "shared_audio_implementation_sha256": implementation["keypress_model.py"],
    })
    cache = output / "cache" / f"deep-waveforms-{cache_hash[:24]}.npz"
    labels = np.asarray([CLASSES.index(str(sample.label)) for sample in samples], dtype=np.int64)
    expected_length = round(config.sample_rate * config.duration_seconds)
    if cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as saved:
                waveforms = np.asarray(saved["waveforms"], dtype=np.float32)
                valid = (
                    str(saved["fingerprint"].item()) == cache_hash
                    and waveforms.shape == (len(samples), expected_length)
                    and np.isfinite(waveforms).all()
                    and np.array_equal(saved["labels"], labels)
                )
            if valid:
                print(f"Using cached prepared training audio: {len(samples)} clips", flush=True)
                return waveforms, labels, cache_hash
        except (OSError, ValueError, KeyError, EOFError):
            pass
        print("Ignoring an invalid prepared-audio cache; decoding the training clips again", flush=True)

    def prepare(index_and_sample):
        index, sample = index_and_sample
        path = Path(sample.path)
        try:
            waveform = prepare_waveform(decode_audio(path, config.sample_rate), config)
        except (ValueError, OSError) as error:
            raise ValueError(f"{path}: {error}") from error
        if sha256(path) != metadata[index]["sha256"]:
            raise ValueError(f"{path}: audio changed during feature extraction")
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.shape != (expected_length,) or not np.isfinite(waveform).all():
            raise ValueError(f"{path}: preprocessing produced invalid audio")
        return waveform

    prepared = []
    with ThreadPoolExecutor(max_workers=min(4, len(samples))) as executor:
        for position, waveform in enumerate(executor.map(prepare, enumerate(samples)), 1):
            prepared.append(waveform)
            if position % 25 == 0 or position == len(samples):
                print(f"Prepared training audio: {position}/{len(samples)}", flush=True)
    waveforms = np.stack(prepared)
    cache.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".deep-waveforms-", suffix=".npz", dir=cache.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, waveforms=waveforms, labels=labels,
                            fingerprint=np.asarray(cache_hash))
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    return waveforms, labels, cache_hash


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _learning_rate_factor(epoch: int, epochs: int) -> float:
    warmup = min(TRAINING_RECIPE["warmup_epochs"], max(0, epochs - 1))
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = min(1.0, max(0.0, (epoch - warmup) / max(1, epochs - warmup - 1)))
    return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2


def _weight_source(architecture: str) -> str:
    return "torchvision.models." + architecture + ": IMAGENET1K_V1 pretrained weights"


def _mixed_precision(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    with torch.cuda.device(device):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _train_member(
    architecture: str, seed: int, waveforms: np.ndarray, labels: np.ndarray,
    config: DeepConfig, epochs: int, batch_size: int, device: torch.device,
    member_index: int, member_count: int,
) -> tuple[dict, list[dict]]:
    _seed_everything(seed)
    network = create_network(architecture, len(CLASSES), pretrained=True).to(device)
    frontend = SpectrogramFrontend(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    head_ids = {id(parameter) for parameter in network.head.parameters()}
    backbone = [parameter for parameter in network.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": TRAINING_RECIPE["backbone_learning_rate"]},
        {"params": network.head.parameters(), "lr": TRAINING_RECIPE["head_learning_rate"]},
    ], weight_decay=TRAINING_RECIPE["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: _learning_rate_factor(epoch, epochs)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=TRAINING_RECIPE["label_smoothing"])
    amp_dtype = _mixed_precision(device)
    use_amp = amp_dtype is not None
    # BF16 retains FP32's exponent range and needs no gradient scaling. Older
    # GPUs fall back to scaled FP16, allowing GradScaler to skip overflow steps.
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(waveforms), torch.from_numpy(labels)),
        batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
        pin_memory=use_amp, generator=generator,
    )
    accumulation = max(1, math.ceil(TRAINING_RECIPE["target_effective_batch_size"] / batch_size))
    history = []
    print(f"Member {member_index}/{member_count}: {architecture}, seed={seed}, "
          f"parameters={parameter_count:,}, pretrained=ImageNet-1K; "
          f"all {len(labels)} clips in every epoch", flush=True)
    print(f"Batch size={batch_size}; accumulation={accumulation}; device={device}; "
          f"mixed precision={amp_dtype or 'disabled'}", flush=True)
    network.train()
    frontend.train()
    for epoch in range(epochs):
        started = time.monotonic()
        last_notice = started
        seen = 0
        loss_sum = 0.0
        correct = 0
        optimizer.zero_grad(set_to_none=True)
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        for batch_index, (waveform_batch, targets) in enumerate(loader):
            waveform_batch = waveform_batch.to(device, non_blocking=use_amp)
            targets = targets.to(device, non_blocking=use_amp)
            group_start = (batch_index // accumulation) * accumulation * batch_size
            group_count = min(accumulation * batch_size, len(labels) - group_start)
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=use_amp):
                images = frontend(waveform_batch, augment=True)
                logits = network(images)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss in member {member_index}, epoch {epoch + 1}")
            scaler.scale(loss * len(targets) / group_count).backward()
            seen += len(targets)
            loss_sum += float(loss.detach()) * len(targets)
            correct += int((logits.detach().argmax(dim=1) == targets).sum())
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), TRAINING_RECIPE["gradient_clip_norm"],
                                               error_if_nonfinite=not scaler.is_enabled())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if time.monotonic() - last_notice >= 10:
                print(f"Member {member_index}/{member_count}, epoch {epoch + 1}/{epochs}: "
                      f"{seen}/{len(labels)} training clips", flush=True)
                last_notice = time.monotonic()
        if seen != len(labels):
            raise RuntimeError("An epoch did not use every supplied training clip")
        scheduler.step()
        row = {
            "member": member_index, "architecture": architecture, "seed": seed,
            "epoch": epoch + 1, "training_examples_seen": seen,
            "augmented_training_loss": loss_sum / seen,
            "augmented_training_correct": correct,
            "augmented_training_accuracy": correct / seen,
            "learning_rates": learning_rates, "seconds": time.monotonic() - started,
        }
        history.append(row)
        print(f"Member {member_index}/{member_count}, epoch {epoch + 1}/{epochs}: "
              f"augmented training loss={row['augmented_training_loss']:.4f}, "
              f"augmented training accuracy={correct}/{seen} ({correct / seen:.1%}), "
              f"{row['seconds']:.1f}s; not test accuracy", flush=True)
    network.eval()
    member = {
        "architecture": architecture, "seed": seed,
        "parameter_count": parameter_count,
        "pretrained_weights": _weight_source(architecture),
        "state_dict": state_to_numpy(network),
    }
    del network, frontend, optimizer, scheduler, scaler, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return member, history


def _archive_previous_run(output: Path) -> Path | None:
    previous = output / "keypress_model.joblib"
    if not previous.is_file():
        return None
    archive = output / "runs" / f"pre-deep-{sha256(previous)[:16]}"
    archive.mkdir(parents=True, exist_ok=True)
    for name in (
        "keypress_model.joblib", "training.json", "summary.json", "training.log",
        "training_history.json", "evaluation_model.joblib", "evaluation.json",
        "heldout_predictions.csv", "confusion_matrix.png",
    ):
        source = output / name
        destination = archive / name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
    return archive


def train_deep(
    dataset: Path, output: Path, *, samples: Sequence, provenance: dict,
    epochs: int = 80, batch_size: int = 8, device: str = "auto",
    architectures: Sequence[str] = DEFAULT_ARCHITECTURES, config: DeepConfig | None = None,
) -> dict:
    """Fit each ensemble member on all clips and publish only after all succeed."""
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    architectures = tuple(architectures)
    if not architectures or any(name not in ARCHITECTURES for name in architectures):
        raise ValueError(f"architectures must be a nonempty selection from {', '.join(ARCHITECTURES)}")
    samples = list(samples)
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    config = config or DeepConfig()
    metadata = _sample_metadata(dataset, samples)
    dataset_hash = fingerprint(metadata)
    implementation = {
        name: sha256(Path(__file__).with_name(name))
        for name in ("deep_training.py", "deep_keypress.py", "keypress_model.py", "2_train.py")
    }
    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if selected_device.type not in ("cuda", "cpu"):
        raise ValueError("deep training supports CPU or CUDA devices")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no compatible CUDA device is available")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    output.mkdir(parents=True, exist_ok=True)
    print(f"Deep ensemble: {len(samples)} training clips; withheld=0; "
          f"internal validation disabled; device={selected_device}", flush=True)
    waveforms, labels, cache_hash = _prepared_audio(
        dataset, output, samples, metadata, config, implementation
    )
    run_config = {
        "dataset_fingerprint": dataset_hash, "prepared_audio_fingerprint": cache_hash,
        "deep_config": asdict(config), "architectures": list(architectures),
        "epochs": epochs, "batch_size": batch_size, "recipe": TRAINING_RECIPE,
        "seeds": [BASE_SEED + index for index in range(len(architectures))],
        "implementation_sha256": implementation,
    }
    run_hash = fingerprint(run_config)
    checkpoint_directory = output / "deep_checkpoints" / run_hash[:24]
    members, history = [], []
    for index, architecture in enumerate(architectures):
        seed = BASE_SEED + index
        checkpoint = checkpoint_directory / f"member-{index + 1}-{architecture}-seed-{seed}.joblib"
        saved = None
        if checkpoint.is_file():
            try:
                candidate = joblib.load(checkpoint)
                if (candidate.get("run_fingerprint") == run_hash and candidate.get("complete") is True
                        and candidate["member"]["architecture"] == architecture
                        and candidate["member"]["seed"] == seed
                        and len(candidate["history"]) == epochs
                        and candidate["history"][-1]["epoch"] == epochs):
                    saved = candidate
            except (OSError, ValueError, KeyError, EOFError):
                pass
        if saved is not None:
            print(f"Reusing completed member {index + 1}/{len(architectures)} from a matching run", flush=True)
            member, member_history = saved["member"], saved["history"]
        else:
            member, member_history = _train_member(
                architecture, seed, waveforms, labels, config, epochs, batch_size,
                selected_device, index + 1, len(architectures),
            )
            _atomic_joblib({"complete": True, "run_fingerprint": run_hash,
                            "member": member, "history": member_history}, checkpoint)
        members.append(member)
        history.extend(member_history)

    # Detect accidental dataset changes before replacing the deployed artifact.
    for sample, item in zip(samples, metadata):
        if sha256(Path(sample.path)) != item["sha256"]:
            raise ValueError(f"{sample.path}: audio changed during training; no model was published")
    versions = {"python": platform.python_version()}
    for distribution in ("numpy", "scipy", "scikit-learn", "joblib", "torch", "torchvision"):
        versions[distribution] = importlib.metadata.version(distribution)
    training_metadata = {
        "training_mode": "full_data", "fit_split": "all_clips",
        "training_sample_count": len(samples), "withheld_sample_count": 0,
        "validation_performed": False, "cross_validation_performed": False,
        "external_test_data_used": False,
        "dataset_fingerprint": dataset_hash, "dataset_directory": str(dataset),
        "run_fingerprint": run_hash, "deep_config": asdict(config),
        "architectures": list(architectures), "member_count": len(members),
        "parameter_count": sum(member["parameter_count"] for member in members),
        "member_parameter_counts": [member["parameter_count"] for member in members],
        "pretrained_weights": [member["pretrained_weights"] for member in members],
        "epochs_per_member": epochs, "batch_size": batch_size,
        "training_recipe": TRAINING_RECIPE, "seeds": run_config["seeds"],
        "library_versions": versions, "implementation_sha256": implementation,
        "training_device": str(selected_device), "cuda_version": torch.version.cuda,
        "mixed_precision": str(_mixed_precision(selected_device) or "disabled"),
        "gpu_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None,
        "deterministic_algorithms": "enabled with warnings for unsupported operations",
        "research_sources": RESEARCH_SOURCES,
        "limitations": LIMITATIONS,
    }
    artifact = {
        "schema_version": 2, "backend": "torch_spectrogram_ensemble",
        "classes": list(CLASSES), "members": members,
        "deep_config": asdict(config), "feature_config": asdict(shared_feature_config(config)),
        "training_metadata": training_metadata,
    }
    validate_artifact(artifact)
    report = {
        "schema_version": 2, "backend": artifact["backend"], **training_metadata,
        "classes": list(CLASSES), "sample_count": len(samples),
        "class_counts": {label: int(np.sum(labels == index)) for index, label in enumerate(CLASSES)},
        "feature_config": artifact["feature_config"], "samples": metadata,
        "provenance": provenance,
        "history_file": "training_history.json",
        "history_metric_note": "All epoch losses and accuracies are measured on augmented training examples, not test data.",
        "final_augmented_training_metrics": [
            member_history for member_history in history if member_history["epoch"] == epochs
        ],
    }
    with tempfile.TemporaryDirectory(prefix=".numpad-deep-training-", dir=output) as temporary:
        staging = Path(temporary)
        joblib.dump(artifact, staging / "keypress_model.joblib", compress=3)
        report["model_sha256"] = sha256(staging / "keypress_model.joblib")
        archive = _archive_previous_run(output)
        if archive is not None:
            report["previous_model_archive"] = archive.relative_to(output).as_posix()
        (staging / "training.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (staging / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        summary = {key: report[key] for key in (
            "training_mode", "backend", "classes", "training_sample_count", "withheld_sample_count",
            "validation_performed", "cross_validation_performed", "parameter_count", "architectures",
            "class_counts", "epochs_per_member", "model_sha256", "limitations",
        )}
        summary.update({"training_report": "training.json", "separate_test": None,
                        "separate_test_note": "Run 4_evaluate.py on this frozen model; no test accuracy was measured during training."})
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for name in ("training_history.json", "training.json", "summary.json"):
            (staging / name).replace(output / name)
        # Publish the model only after completed members and reports are staged.
        (staging / "keypress_model.joblib").replace(output / "keypress_model.joblib")
        if archive is not None:
            for name in ("evaluation_model.joblib", "evaluation.json", "heldout_predictions.csv", "confusion_matrix.png"):
                (output / name).unlink(missing_ok=True)
    print(f"Saved {len(members)}-member ensemble ({training_metadata['parameter_count']:,} parameters) "
          f"trained on all {len(samples)} clips to {output}", flush=True)
    return report
