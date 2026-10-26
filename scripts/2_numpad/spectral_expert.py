"""Training-only augmented spectral experts for the neural/spectral hybrid.

Features contain sound measurements only. No sequence position, filename,
expected key, or evaluation recording is used during feature extraction.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.fft import dct, irfft, rfft
from scipy.signal import get_window
from scipy.special import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from keypress_model import FeatureConfig, mel_filters


CLASSES = ("0", "1", "2", "3")
SCHEMA_VERSION = 1
VIEWS = ("logmel", "temporal_mfcc", "attack160")
CONFIG = {
    "sample_rate": 32000, "duration_seconds": 0.32, "pre_peak_seconds": 0.08,
    "n_fft": 1024, "hop_length": 128, "n_mels": 96, "n_mfcc": 40,
    "fmin": 100.0, "fmax": 15500.0, "db_floor": -80.0,
    "temporal_intervals_seconds": [[-0.020, 0.025], [0.025, 0.075], [0.075, 0.180]],
    "attack_seconds": 0.160, "attack_pre_peak_seconds": 0.008,
    "attack_n_fft": 2048, "attack_n_mels": 128,
    "energy_weight_power": 0.5, "linear_c": 0.01, "linear_temperature": 1.0,
    "lda_shrinkage": 0.1,
    "augmentation_eq_anchor_count": 8,
    "augmentation_variants": [
        {"eq_std_db": 1.5, "noise_snr_db": 35.0},
        {"eq_std_db": 3.0, "noise_snr_db": 30.0},
        {"eq_std_db": 1.5, "noise_snr_db": 40.0},
    ],
    "combination": "equal_weight_mean_probabilities",
}
DIMENSIONS = {"logmel": 192, "temporal_mfcc": 120, "attack160": 256}


def _matrix(prepared: np.ndarray) -> np.ndarray:
    waveforms = np.asarray(prepared, dtype=np.float32)
    expected = round(CONFIG["sample_rate"] * CONFIG["duration_seconds"])
    if (waveforms.ndim != 2 or waveforms.shape[0] < 1 or waveforms.shape[1] != expected
            or not np.isfinite(waveforms).all()):
        raise ValueError(f"Expected a nonempty finite batch of prepared {expected}-sample waveforms")
    return waveforms


def _spectrogram(waveforms: np.ndarray, n_fft: int, n_mels: int) -> tuple[np.ndarray, np.ndarray]:
    config = FeatureConfig(
        sample_rate=CONFIG["sample_rate"], n_fft=n_fft, hop_length=CONFIG["hop_length"],
        n_mels=n_mels, n_mfcc=CONFIG["n_mfcc"], fmin=CONFIG["fmin"], fmax=CONFIG["fmax"],
    )
    padded = np.pad(waveforms, ((0, 0), (n_fft // 2, n_fft // 2)), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft, axis=1)[:, ::CONFIG["hop_length"]]
    spectrum = rfft(frames * get_window("hann", n_fft), axis=-1, workers=2)
    power = np.abs(spectrum) ** 2
    mel = np.matmul(power, mel_filters(config).T)
    logmel = 10 * np.log10(np.maximum(mel, 1e-12))
    logmel -= logmel.max(axis=(1, 2), keepdims=True)
    logmel = np.maximum(logmel, CONFIG["db_floor"])
    weights = np.maximum(mel.sum(axis=2), 1e-24) ** CONFIG["energy_weight_power"]
    weights /= weights.sum(axis=1, keepdims=True)
    return logmel, weights


def _weighted_statistics(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = np.sum(values * weights[:, :, None], axis=1)
    variance = np.sum((values - mean[:, None]) ** 2 * weights[:, :, None], axis=1)
    return np.concatenate((mean, np.sqrt(np.maximum(variance, 0))), axis=1)


def extract_features(prepared: np.ndarray) -> dict[str, np.ndarray]:
    """Return three deterministic feature views, without augmentation."""
    waveforms = _matrix(prepared)
    pieces = {name: [] for name in VIEWS}
    with threadpool_limits(limits=2):
        for offset in range(0, len(waveforms), 64):
            batch = waveforms[offset:offset + 64]
            logmel, weights = _spectrogram(batch, CONFIG["n_fft"], CONFIG["n_mels"])
            pieces["logmel"].append(_weighted_statistics(logmel, weights))
            coefficients = dct(logmel, type=2, axis=2, norm="ortho")[:, :, :CONFIG["n_mfcc"]]
            times = np.arange(logmel.shape[1]) * CONFIG["hop_length"] / CONFIG["sample_rate"] - CONFIG["pre_peak_seconds"]
            temporal = []
            for start, end in CONFIG["temporal_intervals_seconds"]:
                selected = (times >= start) & (times < end)
                local_weights = weights[:, selected]
                local_weights /= local_weights.sum(axis=1, keepdims=True)
                temporal.append(np.sum(coefficients[:, selected] * local_weights[:, :, None], axis=1))
            pieces["temporal_mfcc"].append(np.concatenate(temporal, axis=1))
            start = round((CONFIG["pre_peak_seconds"] - CONFIG["attack_pre_peak_seconds"]) * CONFIG["sample_rate"])
            length = round(CONFIG["attack_seconds"] * CONFIG["sample_rate"])
            attack, attack_weights = _spectrogram(batch[:, start:start + length], CONFIG["attack_n_fft"], CONFIG["attack_n_mels"])
            pieces["attack160"].append(_weighted_statistics(attack, attack_weights))
    features = {name: np.concatenate(rows).astype(np.float64) for name, rows in pieces.items()}
    if any(values.shape != (len(waveforms), DIMENSIONS[name]) or not np.isfinite(values).all()
           for name, values in features.items()):
        raise ValueError("Spectral expert feature extraction produced invalid values")
    return features


def _augment_training(waveforms: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    variants = [waveforms]
    frequencies = np.fft.rfftfreq(waveforms.shape[1], 1 / CONFIG["sample_rate"])
    anchors = np.linspace(0, CONFIG["sample_rate"] / 2, CONFIG["augmentation_eq_anchor_count"])
    original_spectrum = rfft(waveforms, axis=1, workers=2)
    for settings in CONFIG["augmentation_variants"]:
        responses = rng.normal(0, settings["eq_std_db"], (len(waveforms), len(anchors)))
        eq = np.stack([np.interp(frequencies, anchors, response) for response in responses])
        augmented = irfft(original_spectrum * 10 ** (eq / 20), n=waveforms.shape[1], axis=1, workers=2)
        scale = np.sqrt(np.mean(augmented ** 2, axis=1, keepdims=True)) / 10 ** (settings["noise_snr_db"] / 20)
        augmented += rng.normal(size=augmented.shape) * scale
        augmented -= augmented.mean(axis=1, keepdims=True)
        augmented /= np.maximum(np.max(np.abs(augmented), axis=1, keepdims=True), 1e-8)
        variants.append(augmented.astype(np.float32))
    return np.concatenate(variants)


def fit_expert(prepared: np.ndarray, labels: np.ndarray, seed: int = 42) -> dict:
    """Fit all six fixed experts on every clip plus three training variants."""
    waveforms = _matrix(prepared)
    labels = np.asarray(labels, dtype=str)
    if labels.shape != (len(waveforms),) or set(labels.tolist()) != set(CLASSES):
        raise ValueError("Expert training requires one label per waveform and examples of all keys 0-3")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("The augmentation seed must be an integer")
    augmented = _augment_training(waveforms, np.random.default_rng(seed))
    augmented_labels = np.tile(labels, 1 + len(CONFIG["augmentation_variants"]))
    features = extract_features(augmented)
    members = []
    with threadpool_limits(limits=2):
        for view in VIEWS:
            for kind in ("linear", "lda"):
                classifier = (
                    LinearSVC(C=CONFIG["linear_c"], dual="auto", max_iter=50000, random_state=int(seed))
                    if kind == "linear" else LinearDiscriminantAnalysis(solver="lsqr", shrinkage=CONFIG["lda_shrinkage"])
                )
                pipeline = make_pipeline(StandardScaler(), classifier)
                pipeline.fit(features[view], augmented_labels)
                members.append({"view": view, "kind": kind, "pipeline": pipeline,
                                "feature_count": DIMENSIONS[view]})
    expert = {
        "schema_version": SCHEMA_VERSION, "config": json.loads(json.dumps(CONFIG)),
        "classes": list(CLASSES), "members": members,
        "training_metadata": {
            "original_training_sample_count": len(waveforms),
            "augmented_training_sample_count": len(augmented),
            "augmentation_factor": 1 + len(CONFIG["augmentation_variants"]),
            "class_counts": {label: int(np.sum(labels == label)) for label in CLASSES},
            "seed": int(seed), "withheld_sample_count": 0,
            "validation_performed": False, "cross_validation_performed": False,
            "benchmark_used_for_development": True,
            "benchmark_note": "The existing 20-press recording informed feature design; its score is a development benchmark, not an unseen test estimate.",
            "test_audio_used_for_fitting": False,
            "parameter_search_performed": False,
            "confidence_type": "Uncalibrated equal-weight expert probabilities, not measured accuracy",
        },
    }
    validate_expert(expert)
    return expert


def validate_expert(expert: dict) -> None:
    """Reject mismatched feature versions, class order, and unfitted heads."""
    try:
        if (expert["schema_version"] != SCHEMA_VERSION or expert["config"] != CONFIG
                or expert["classes"] != list(CLASSES)):
            raise ValueError("Unsupported spectral expert configuration or classes")
        members = expert["members"]
        expected = {(view, kind) for view in VIEWS for kind in ("linear", "lda")}
        actual = {(member["view"], member["kind"]) for member in members}
        if len(members) != len(expected) or actual != expected:
            raise ValueError("Spectral expert is missing one or more of its six fixed members")
        for member in members:
            pipeline = member["pipeline"]
            if (list(pipeline.classes_) != list(CLASSES)
                    or pipeline.n_features_in_ != DIMENSIONS[member["view"]]):
                raise ValueError("Spectral classifier dimensions or class order do not match")
            estimator = pipeline.steps[-1][1]
            expected_type = LinearSVC if member["kind"] == "linear" else LinearDiscriminantAnalysis
            if not isinstance(estimator, expected_type):
                raise ValueError("Spectral classifier type does not match its declared kind")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Spectral expert is missing its configuration or fitted classifiers") from error


def predict_probabilities(prepared: np.ndarray, expert: dict) -> np.ndarray:
    """Predict one prepared event, combining all six experts equally."""
    validate_expert(expert)
    waveform = np.asarray(prepared, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError("Expected one prepared mono waveform")
    features = extract_features(waveform[None])
    probabilities = []
    with threadpool_limits(limits=2):
        for member in expert["members"]:
            values = features[member["view"]]
            pipeline = member["pipeline"]
            if member["kind"] == "linear":
                probability = softmax(pipeline.decision_function(values)[0] / CONFIG["linear_temperature"])
            else:
                probability = pipeline.predict_proba(values)[0]
            probabilities.append(probability)
    result = np.mean(probabilities, axis=0)
    if result.shape != (len(CLASSES),) or not np.isfinite(result).all() or np.any(result < 0) or not np.isclose(result.sum(), 1):
        raise ValueError("Spectral experts produced invalid probabilities")
    return result / result.sum()
