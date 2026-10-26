"""Shared spectrogram preprocessing and a pretrained Swin ensemble.

Training, file inference, and live inference all use this frontend. Neural
checkpoints embed NumPy weights, so old sklearn artifacts need no torch imports.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
import os

import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURES = ("swin_b", "swin_t")
DEFAULT_ARCHITECTURES = ("swin_b", "swin_t", "swin_t")


@dataclass(frozen=True)
class DeepConfig:
    sample_rate: int = 32000
    duration_seconds: float = 0.32
    pre_peak_seconds: float = 0.08
    n_fft: int = 1024
    hop_length: int = 128
    n_mels: int = 224
    fmin: float = 100.0
    fmax: float = 15500.0
    db_floor: float = -80.0
    image_size: int = 224

    def __post_init__(self):
        # This also validates window timing and frequency limits.
        shared_feature_config(self)
        if self.image_size != 224:
            raise ValueError("The pretrained Swin frontend requires image_size=224")
        if self.n_mels < 64:
            raise ValueError("Use at least 64 mel bands")


def shared_feature_config(config: DeepConfig):
    from keypress_model import FeatureConfig
    values = asdict(config)
    values.pop("image_size")
    return FeatureConfig(**values, mode="mfcc_weighted", n_mfcc=40, energy_weight_power=0.5)


def prepare_waveform(waveform: np.ndarray, config: DeepConfig) -> np.ndarray:
    from keypress_model import aligned_segment
    return aligned_segment(waveform, shared_feature_config(config)).astype(np.float32)


def create_network(architecture: str, classes: int, pretrained: bool = False) -> nn.Module:
    from torchvision.models import Swin_B_Weights, Swin_T_Weights, swin_b, swin_t
    if architecture not in ARCHITECTURES or classes < 2:
        raise ValueError("Unsupported Swin architecture or class count")
    factory, weights = {
        "swin_b": (swin_b, Swin_B_Weights.IMAGENET1K_V1),
        "swin_t": (swin_t, Swin_T_Weights.IMAGENET1K_V1),
    }[architecture]
    model = factory(weights=weights if pretrained else None)
    model.head = nn.Linear(model.head.in_features, classes)
    return model


class SpectrogramFrontend(nn.Module):
    """224-band log-mel images, with augmentation restricted to training.

    Augmentations preserve the high-frequency resonances separating nearby keys:
    small time shifts, 20–40 dB noise, smooth microphone-response changes and
    small time/frequency masks. Frequency/pitch is never shifted.
    """

    def __init__(self, config: DeepConfig):
        super().__init__()
        from keypress_model import mel_filters
        self.config = config
        self.register_buffer("window", torch.hann_window(config.n_fft))
        self.register_buffer("mel", torch.tensor(mel_filters(shared_feature_config(config)).copy(), dtype=torch.float32))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def forward(self, waveforms: torch.Tensor, augment: bool = False) -> torch.Tensor:
        if waveforms.ndim != 2:
            raise ValueError("Expected a batch of mono waveforms")
        # FFT remains float32 even when the backbone uses mixed precision.
        with torch.autocast(device_type=waveforms.device.type, enabled=False):
            waveforms = waveforms.float()
            if augment:
                batch, length = waveforms.shape
                maximum = round(0.020 * self.config.sample_rate)
                shifts = torch.randint(-maximum, maximum + 1, (batch, 1), device=waveforms.device)
                positions = torch.arange(length, device=waveforms.device)[None, :] - shifts
                valid = (positions >= 0) & (positions < length)
                waveforms = torch.gather(waveforms, 1, positions.clamp(0, length - 1)) * valid
                snr_db = torch.empty(batch, 1, device=waveforms.device).uniform_(20, 40)
                scale = waveforms.square().mean(dim=1, keepdim=True).sqrt() / (10 ** (snr_db / 20))
                # Mix clean and noisy events rather than corrupt every example.
                enabled = (torch.rand(batch, 1, device=waveforms.device) < 0.6)
                waveforms = waveforms + torch.randn_like(waveforms) * scale * enabled
            spectrum = torch.stft(waveforms, n_fft=self.config.n_fft,
                                  hop_length=self.config.hop_length, window=self.window,
                                  center=True, return_complex=True)
            power = spectrum.abs().square()
            mel = torch.matmul(self.mel, power).clamp_min(1e-12)
            decibels = 10 * mel.log10()
            if augment:
                response = torch.randn(len(waveforms), 1, 8, device=waveforms.device) * 2.0
                response = F.interpolate(response, size=self.config.n_mels, mode="linear", align_corners=True)
                decibels = decibels + response.transpose(1, 2)
            decibels = decibels - decibels.amax(dim=(1, 2), keepdim=True)
            image = (decibels.clamp_min(self.config.db_floor) - self.config.db_floor) / -self.config.db_floor
            if augment:
                # Fill masks with the floor. Masks never cover more than 3%.
                frequency_width = max(1, round(self.config.n_mels * 0.03))
                time_width = max(1, round(image.shape[-1] * 0.03))
                for index in range(len(image)):
                    for axis, maximum in ((0, frequency_width), (1, time_width)):
                        width = int(torch.randint(0, maximum + 1, ()).item())
                        if not width:
                            continue
                        size = image.shape[axis + 1]
                        start = int(torch.randint(0, size - width + 1, ()).item())
                        if axis == 0:
                            image[index, start:start + width, :] = 0
                        else:
                            image[index, :, start:start + width] = 0
            image = F.interpolate(image[:, None], size=(self.config.image_size, self.config.image_size),
                                  mode="bilinear", align_corners=False)
            return (image.expand(-1, 3, -1, -1) - self.mean) / self.std


def state_to_numpy(model: nn.Module) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().to(torch.float16).numpy().copy()
        if value.is_floating_point() else value.detach().cpu().numpy().copy()
        for name, value in model.state_dict().items()
    }


def validate_artifact(artifact: dict) -> None:
    if artifact.get("schema_version") != 2 or artifact.get("backend") != "torch_spectrogram_ensemble":
        raise ValueError("Unsupported neural keypress model format")
    try:
        config = DeepConfig(**artifact["deep_config"])
        if artifact["feature_config"] != asdict(shared_feature_config(config)):
            raise ValueError("Neural and shared feature configurations differ")
        classes = artifact["classes"]
        if (not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes)
                or any(not isinstance(key, str) or len(key) != 1 or key not in "0123456789" for key in classes)):
            raise ValueError("Invalid neural model classes")
        members = artifact["members"]
        if not isinstance(members, list) or not members:
            raise ValueError("Neural ensemble is missing its trained members")
        for member in members:
            if member["architecture"] not in ARCHITECTURES or not isinstance(member["state_dict"], dict):
                raise ValueError("Invalid neural ensemble member")
            weights = member["state_dict"]
            if (not weights or "head.weight" not in weights
                    or weights["head.weight"].shape[0] != len(classes)):
                raise ValueError("Neural classifier head does not match its declared classes")
            if any(not isinstance(value, np.ndarray) or not np.isfinite(value).all() for value in weights.values()):
                raise ValueError("Neural model weights must be finite NumPy arrays")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Neural model is missing its configuration or fitted weights") from error


class EnsemblePredictor:
    def __init__(self, artifact: dict):
        self.config = DeepConfig(**artifact["deep_config"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.frontend = SpectrogramFrontend(self.config).to(self.device).eval()
        self.models = []
        for member in artifact["members"]:
            network = create_network(member["architecture"], len(artifact["classes"]))
            state = {name: torch.from_numpy(value) for name, value in member["state_dict"].items()}
            try:
                network.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise ValueError(f"Neural checkpoint does not match its architecture: {error}") from error
            self.models.append(network.to(self.device).eval())

    @torch.inference_mode()
    def probabilities(self, prepared: np.ndarray) -> np.ndarray:
        waveform = torch.from_numpy(np.asarray(prepared, dtype=np.float32)).reshape(1, -1).to(self.device)
        images = self.frontend(waveform)
        context = torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()
        with context:
            probabilities = torch.stack([network(images).float().softmax(dim=1) for network in self.models]).mean(dim=0)
        return probabilities[0].cpu().numpy().astype(np.float64)


def warmup(artifact: dict) -> None:
    """Initialize and warm the GPU before opening a microphone stream."""
    if "_deep_predictor" not in artifact:
        artifact["_deep_predictor"] = EnsemblePredictor(artifact)
        config = artifact["_deep_predictor"].config
        artifact["_deep_predictor"].probabilities(np.zeros(round(config.duration_seconds * config.sample_rate), np.float32))


def predict_waveform(waveform: np.ndarray, artifact: dict, top_k: int = 3,
                     *, sample_rate: int | None = None) -> dict:
    classes = artifact["classes"]
    if not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= len(classes):
        raise ValueError(f"top_k must be between 1 and {len(classes)} for this model")
    config = DeepConfig(**artifact["deep_config"])
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError("Expected finite, one-dimensional mono audio")
    if sample_rate is not None:
        if not isinstance(sample_rate, (int, np.integer)) or sample_rate < 1000:
            raise ValueError("sample_rate must be an integer of at least 1000 Hz")
        if sample_rate != config.sample_rate:
            divisor = math.gcd(int(sample_rate), config.sample_rate)
            waveform = resample_poly(waveform, config.sample_rate // divisor, int(sample_rate) // divisor)
    prepared = prepare_waveform(waveform, config)
    warmup(artifact)
    probabilities = artifact["_deep_predictor"].probabilities(prepared)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError("The neural ensemble returned invalid probabilities")
    order = np.argsort(-probabilities, kind="stable")
    ranked = [{"key": classes[index], "score": float(np.log(max(probabilities[index], 1e-12))),
               "probability": float(probabilities[index])} for index in order[:top_k]]
    return {
        "predicted_key": ranked[0]["key"], "confidence": ranked[0]["probability"],
        "confidence_type": "Uncalibrated ensemble softmax probability; not measured accuracy",
        "ranked_keys": ranked,
        "margin": float(np.log(max(probabilities[order[0]], 1e-12)) - np.log(max(probabilities[order[1]], 1e-12))),
    }
