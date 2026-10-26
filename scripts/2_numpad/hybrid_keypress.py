"""Fuse invariant acoustic experts with the large pretrained neural ensemble."""

from __future__ import annotations

import math
import numpy as np
from scipy.signal import resample_poly

NEURAL_WEIGHT = 0.2


def validate_artifact(artifact: dict) -> None:
    from deep_keypress import validate_artifact as validate_neural
    from spectral_expert import validate_expert
    try:
        if artifact['schema_version'] != 2 or artifact['backend'] != 'neural_spectral_hybrid':
            raise ValueError('Unsupported hybrid model format')
        neural = artifact['neural_model']
        validate_neural(neural)
        validate_expert(artifact['spectral_expert'])
        if (artifact['classes'] != neural['classes']
                or artifact['classes'] != artifact['spectral_expert']['classes']
                or artifact['feature_config'] != neural['feature_config']):
            raise ValueError('Hybrid expert classes or preprocessing do not match')
        if artifact['neural_weight'] != NEURAL_WEIGHT:
            raise ValueError('Unsupported hybrid fusion weight')
    except (KeyError, TypeError) as error:
        raise ValueError('Hybrid model is missing its fitted experts or configuration') from error


def warmup(artifact: dict) -> None:
    from deep_keypress import warmup as warm_neural
    warm_neural(artifact['neural_model'])


def predict_waveform(waveform: np.ndarray, artifact: dict, top_k: int = 3,
                     *, sample_rate: int | None = None) -> dict:
    from deep_keypress import DeepConfig, prepare_waveform, predict_waveform as neural_predict
    from spectral_expert import predict_probabilities
    classes = artifact['classes']
    if not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= len(classes):
        raise ValueError(f'top_k must be between 1 and {len(classes)} for this model')
    config = DeepConfig(**artifact['neural_model']['deep_config'])
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError('Expected finite, one-dimensional mono audio')
    if sample_rate is not None:
        if not isinstance(sample_rate, (int, np.integer)) or sample_rate < 1000:
            raise ValueError('sample_rate must be a positive integer of at least 1000 Hz')
        if sample_rate < config.sample_rate:
            raise ValueError('This model needs capture at 32 kHz or higher to retain key resonances')
        if sample_rate != config.sample_rate:
            divisor = math.gcd(int(sample_rate), config.sample_rate)
            waveform = resample_poly(waveform, config.sample_rate // divisor, int(sample_rate) // divisor)
    prepared = prepare_waveform(waveform, config)
    acoustic = predict_probabilities(prepared, artifact['spectral_expert'])
    neural = neural_predict(waveform, artifact['neural_model'], top_k=len(classes), sample_rate=config.sample_rate)
    neural_probabilities = np.zeros(len(classes))
    for item in neural['ranked_keys']:
        neural_probabilities[classes.index(item['key'])] = item['probability']
    weight = artifact['neural_weight']
    probabilities = (1 - weight) * acoustic + weight * neural_probabilities
    if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1):
        raise ValueError('Hybrid experts returned invalid probabilities')
    order = np.argsort(-probabilities, kind='stable')
    ranked = [{'key': classes[index], 'score': float(np.log(max(probabilities[index], 1e-12))),
               'probability': float(probabilities[index])} for index in order[:top_k]]
    return {'predicted_key': ranked[0]['key'], 'confidence': ranked[0]['probability'],
            'confidence_type': 'Uncalibrated hybrid probability; not measured accuracy',
            'ranked_keys': ranked,
            'margin': float(np.log(max(probabilities[order[0]], 1e-12)) - np.log(max(probabilities[order[1]], 1e-12)))}
