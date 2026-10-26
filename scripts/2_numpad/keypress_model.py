"""Shared audio features and inference for the numpad sound classifier."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import pickle
import shutil
import subprocess

import joblib
import numpy as np
from scipy.fft import dct, rfft
from scipy.signal import find_peaks, get_window, peak_widths, resample, resample_poly
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets" / "2_numpad"
DEFAULT_WORK = ROOT / "work" / "2_numpad"
DEFAULT_MODEL = DEFAULT_WORK / "keypress_model.joblib"
MODEL_SCHEMA_VERSION = 1
NEURAL_MODEL_SCHEMA_VERSION = 2
NEURAL_BACKEND = "torch_spectrogram_ensemble"
HYBRID_BACKEND = "neural_spectral_hybrid"
FEATURE_MODES = (
    "mfcc_stats", "logmel_stats", "logmel_patch", "logmel_temporal",
    "mfcc_weighted", "logmel_weighted",
)
MAX_AUDIO_SECONDS = 30


@dataclass(frozen=True)
class FeatureConfig:
    mode: str = "mfcc_stats"
    sample_rate: int = 24000
    duration_seconds: float = 0.32
    pre_peak_seconds: float = 0.04
    n_fft: int = 512
    hop_length: int = 128
    n_mels: int = 64
    n_mfcc: int = 20
    fmin: float = 100.0
    fmax: float = 11000.0
    db_floor: float = -80.0
    patch_frames: int = 32
    temporal_segments: int = 4
    energy_weight_power: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in FEATURE_MODES:
            raise ValueError(f"Unknown feature mode: {self.mode}")
        if not (0 < self.fmin < self.fmax < self.sample_rate / 2):
            raise ValueError("Feature frequencies must lie below the sample rate's Nyquist frequency")
        if not (0 <= self.pre_peak_seconds < self.duration_seconds):
            raise ValueError("Invalid feature window timing")
        if not (0 < self.hop_length <= self.n_fft <= round(self.duration_seconds * self.sample_rate)):
            raise ValueError("Invalid FFT size or hop length")
        if not (1 <= self.n_mfcc <= self.n_mels) or self.db_floor >= 0:
            raise ValueError("Invalid MFCC count or dB floor")
        frame_count = 1 + (round(self.duration_seconds * self.sample_rate) - self.n_fft) // self.hop_length
        if not (1 <= self.temporal_segments <= frame_count) or self.patch_frames < 1:
            raise ValueError("Invalid temporal feature dimensions")
        if not math.isfinite(self.energy_weight_power) or self.energy_weight_power <= 0:
            raise ValueError("Energy weighting power must be positive and finite")


def decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Decode a local audio file to mono at the model's analysis sample rate."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Audio file does not exist: {path}")
    if shutil.which("ffmpeg") is None:
        raise ValueError("FFmpeg is required on PATH")
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
        "-t", str(MAX_AUDIO_SECONDS + 0.1), "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError(f"Cannot decode {path.name}: {result.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)


def aligned_segment(waveform: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Validate an isolated press and align a fixed window to its strongest transient."""
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError("Expected finite, one-dimensional mono audio")
    seconds = len(waveform) / config.sample_rate
    if not (0.05 <= seconds <= MAX_AUDIO_SECONDS):
        raise ValueError(f"Provide a single-key clip between 0.05 and {MAX_AUDIO_SECONDS} seconds")
    frame_size = max(1, round(0.005 * config.sample_rate))
    count = len(waveform) // frame_size
    energy = np.mean(waveform[:count * frame_size].reshape(count, frame_size) ** 2, axis=1)
    if count * frame_size < len(waveform):
        energy = np.append(energy, np.mean(waveform[count * frame_size:] ** 2))
    envelope = np.sqrt(energy)
    peak_rms = float(np.max(envelope))
    noise_floor = float(np.percentile(envelope, 20))
    if peak_rms < 1e-5 or peak_rms < 6 * max(noise_floor, 1e-10):
        raise ValueError("No distinct keypress transient: the clip is silent or dominated by background noise")
    threshold = max(noise_floor * 6, peak_rms * 0.18)
    peaks, _ = find_peaks(
        np.pad(envelope, (1, 1)), height=threshold,
        distance=math.ceil(0.350 * config.sample_rate / frame_size),
    )
    # Match the splitter's transient shape checks: slow background rises are
    # preserved in the clips but must not become additional keypresses here.
    frame_seconds = frame_size / config.sample_rate
    widths = peak_widths(np.pad(envelope, (1, 1)), peaks, rel_height=0.5)[0] * frame_seconds
    transient_peaks = []
    for peak, width in zip(peaks - 1, widths):
        start = max(0, int(peak) - math.ceil(0.050 / frame_seconds))
        end = max(0, int(peak) - math.ceil(0.015 / frame_seconds))
        baseline = float(np.median(envelope[start:end])) if end > start else noise_floor
        contrast = envelope[peak] / max(baseline, 1e-8)
        if width <= 0.050 and contrast >= 4.0:
            transient_peaks.append(int(peak))
    peaks = np.asarray(transient_peaks, dtype=np.int64)
    if len(peaks) != 1:
        raise ValueError(f"Expected one isolated keypress; detected {len(peaks)} separated presses")
    peak_frame = int(np.argmax(envelope))
    frame_start = peak_frame * frame_size
    frame_end = min(frame_start + frame_size, len(waveform))
    peak_center = (frame_start + frame_end) // 2
    start = peak_center - round(config.pre_peak_seconds * config.sample_rate)
    length = round(config.duration_seconds * config.sample_rate)
    segment = np.zeros(length, dtype=np.float64)
    source_start = max(0, start)
    source_end = min(len(waveform), start + length)
    target_start = source_start - start
    segment[target_start:target_start + source_end - source_start] = waveform[source_start:source_end]
    segment -= segment.mean()
    segment /= max(float(np.max(np.abs(segment))), 1e-8)
    return segment


@lru_cache(maxsize=16)
def mel_filters(config: FeatureConfig) -> np.ndarray:
    """HTK mel triangles, each normalized by its sum over FFT bins."""
    def mel(hz):
        return 2595 * np.log10(1 + np.asarray(hz) / 700)

    def hz(mels):
        return 700 * (10 ** (np.asarray(mels) / 2595) - 1)

    frequencies = np.fft.rfftfreq(config.n_fft, 1 / config.sample_rate)
    points = hz(np.linspace(mel(config.fmin), mel(config.fmax), config.n_mels + 2))
    rising = (frequencies[None, :] - points[:-2, None]) / (points[1:-1, None] - points[:-2, None])
    falling = (points[2:, None] - frequencies[None, :]) / (points[2:, None] - points[1:-1, None])
    filters = np.maximum(0, np.minimum(rising, falling))
    filters /= np.maximum(filters.sum(axis=1, keepdims=True), 1e-12)
    filters.setflags(write=False)
    return filters


def logmel_spectrogram(waveform: np.ndarray, config: FeatureConfig) -> np.ndarray:
    segment = aligned_segment(waveform, config)
    frames = np.lib.stride_tricks.sliding_window_view(segment, config.n_fft)[::config.hop_length]
    window = get_window("hann", config.n_fft, fftbins=True)
    spectrum = rfft(frames * window, axis=1).T / window.sum()
    mel_power = mel_filters(config) @ (np.abs(spectrum) ** 2)
    logmel = 10 * np.log10(np.maximum(mel_power, 1e-12))
    return np.maximum(logmel - logmel.max(), config.db_floor)


def representations(logmel: np.ndarray, config: FeatureConfig) -> dict[str, np.ndarray]:
    coefficients = dct(logmel, type=2, axis=0, norm="ortho")[:config.n_mfcc]
    chunks = np.array_split(logmel, config.temporal_segments, axis=1)
    # Relative mel power is sufficient: a common gain cancels when normalizing
    # these weights. Quiet tails contribute very little to the event statistics.
    weights = np.sum(10 ** (logmel / 10), axis=0) ** config.energy_weight_power
    weights /= weights.sum()

    def weighted_stats(values: np.ndarray) -> np.ndarray:
        mean = values @ weights
        variance = ((values - mean[:, None]) ** 2) @ weights
        return np.r_[mean, np.sqrt(np.maximum(variance, 0))]

    variants = {
        "mfcc_stats": np.r_[coefficients.mean(axis=1), coefficients.std(axis=1)],
        "logmel_stats": np.r_[logmel.mean(axis=1), logmel.std(axis=1)],
        "logmel_patch": resample(logmel, config.patch_frames, axis=1).ravel(),
        "logmel_temporal": np.concatenate(
            [chunk.mean(axis=1) for chunk in chunks] + [logmel.std(axis=1)]
        ),
        "mfcc_weighted": weighted_stats(coefficients),
        "logmel_weighted": weighted_stats(logmel),
    }
    if any(not np.isfinite(values).all() for values in variants.values()):
        raise ValueError("Audio feature extraction produced non-finite values")
    return variants


def feature_variants_from_waveform(
    waveform: np.ndarray, config: FeatureConfig = FeatureConfig()
) -> dict[str, np.ndarray]:
    return representations(logmel_spectrogram(waveform, config), config)


def feature_variants(path: Path, config: FeatureConfig = FeatureConfig()) -> dict[str, np.ndarray]:
    return feature_variants_from_waveform(decode_audio(path, config.sample_rate), config)


def extract_features(path: Path, config: FeatureConfig = FeatureConfig()) -> np.ndarray:
    return feature_variants(path, config)[config.mode]


def build_classifier(
    c: float = 10.0,
    *,
    kind: str = "rbf",
    gamma: str | float = "scale",
    shrinkage: str | float = 0.5,
):
    # The scaler is fit only on each training fold, never on validation examples.
    if kind == "rbf":
        estimator = SVC(C=c, kernel="rbf", gamma=gamma, decision_function_shape="ovr", break_ties=True)
    elif kind == "linear":
        estimator = LinearSVC(C=c, dual="auto", max_iter=20000, random_state=42)
    elif kind == "lda":
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
    else:
        raise ValueError(f"Unknown classifier kind: {kind}")
    return make_pipeline(StandardScaler(), estimator)


def load_model(path: Path = DEFAULT_MODEL) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Model does not exist: {path}. Run scripts/2_numpad/2_train.py first.")
    try:
        artifact = joblib.load(path)
    except (EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f"Invalid model file: {path}") from error
    if not isinstance(artifact, dict):
        raise ValueError("Unsupported keypress model format; retrain with 2_train.py")
    if artifact.get("schema_version") == NEURAL_MODEL_SCHEMA_VERSION:
        if artifact.get("backend") not in (NEURAL_BACKEND, HYBRID_BACKEND):
            raise ValueError("Unsupported neural keypress model backend; retrain with 2_train.py")
        try:
            FeatureConfig(**artifact["feature_config"])
        except (KeyError, TypeError) as error:
            raise ValueError("Neural model is missing its audio feature configuration") from error
        if artifact["backend"] == HYBRID_BACKEND:
            from hybrid_keypress import validate_artifact
        else:
            from deep_keypress import validate_artifact

        validate_artifact(artifact)
        return artifact
    if artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("Unsupported keypress model format; retrain with 2_train.py")
    try:
        FeatureConfig(**artifact["feature_config"])
        classes = [str(key) for key in artifact["pipeline"].classes_]
        if (len(classes) < 2 or len(set(classes)) != len(classes)
                or any(len(key) != 1 or key not in "0123456789" for key in classes)
                or classes != artifact["classes"]):
            raise ValueError("The model must contain at least two distinct numpad digits matching its declared classes")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Model is missing its fitted classifier or feature configuration") from error
    return artifact


def predict_waveform(
    waveform: np.ndarray,
    artifact: dict,
    top_k: int = 3,
    *,
    sample_rate: int | None = None,
) -> dict:
    """Classify an in-memory mono event; resample the complete event if needed.

    Confidence is the estimator's uncalibrated probability when available. It
    is not measured accuracy. Estimators without probabilities return None.
    """
    if artifact.get("schema_version") == NEURAL_MODEL_SCHEMA_VERSION:
        if artifact.get("backend") not in (NEURAL_BACKEND, HYBRID_BACKEND):
            raise ValueError("Unsupported neural keypress model backend; retrain with 2_train.py")
        if artifact["backend"] == HYBRID_BACKEND:
            from hybrid_keypress import predict_waveform as predict_neural_waveform
        else:
            from deep_keypress import predict_waveform as predict_neural_waveform

        return predict_neural_waveform(waveform, artifact, top_k=top_k, sample_rate=sample_rate)
    pipeline = artifact["pipeline"]
    classes = [str(key) for key in pipeline.classes_]
    if not isinstance(top_k, (int, np.integer)) or not (1 <= top_k <= len(classes)):
        raise ValueError(f"top_k must be between 1 and {len(classes)} for this model")
    config = FeatureConfig(**artifact["feature_config"])
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError("Expected finite, one-dimensional mono audio")
    if sample_rate is not None:
        if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if sample_rate != config.sample_rate:
            divisor = math.gcd(int(sample_rate), config.sample_rate)
            waveform = resample_poly(waveform, config.sample_rate // divisor, int(sample_rate) // divisor)
    features = feature_variants_from_waveform(waveform, config)[config.mode].reshape(1, -1)
    decision = np.asarray(pipeline.decision_function(features))
    if len(classes) == 2 and decision.shape == (1,):
        # Binary sklearn classifiers expose the score for the second class.
        scores = np.array([-decision[0], decision[0]])
    else:
        scores = decision[0]
    if scores.shape != (len(classes),) or not np.isfinite(scores).all():
        raise ValueError("Classifier returned invalid decision scores")
    order = np.argsort(-scores, kind="stable")
    predicted = str(pipeline.predict(features)[0])
    probabilities = None
    probability_method = getattr(pipeline, "predict_proba", None)
    if probability_method is not None:
        probabilities = np.asarray(probability_method(features))[0]
        if (probabilities.shape != scores.shape or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.isclose(probabilities.sum(), 1)):
            raise ValueError("Classifier returned invalid probability estimates")
    return {
        "predicted_key": predicted,
        "ranked_keys": [
            {"key": classes[index], "score": float(scores[index]),
             "probability": float(probabilities[index]) if probabilities is not None else None}
            for index in order[:top_k]
        ],
        "confidence": float(probabilities[classes.index(predicted)]) if probabilities is not None else None,
        "confidence_type": "uncalibrated_model_probability" if probabilities is not None else "unavailable",
        "margin": float(scores[order[0]] - scores[order[1]]),
        "score_type": "Uncalibrated classifier decision score; not a probability",
    }


def predict_key(path: Path, artifact: dict, top_k: int = 3) -> dict:
    config = FeatureConfig(**artifact["feature_config"])
    result = predict_waveform(decode_audio(path, config.sample_rate), artifact, top_k)
    return {"audio_file": str(Path(path).resolve()), **result}
