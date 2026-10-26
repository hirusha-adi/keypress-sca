"""Fit acoustic experts on all clips and combine them with trained transformers."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np


def train_hybrid(dataset: Path, output: Path, *, samples, provenance,
                 epochs=80, batch_size=8, device='auto') -> dict:
    from deep_keypress import DeepConfig, prepare_waveform
    from deep_training import train_deep, _sample_metadata, fingerprint, _archive_previous_run
    from hybrid_keypress import NEURAL_WEIGHT, validate_artifact
    from keypress_model import decode_audio
    from spectral_expert import fit_expert
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    samples = list(samples)
    metadata = _sample_metadata(dataset, samples)
    dataset_hash = fingerprint(metadata)
    config = DeepConfig()
    current = output / 'keypress_model.joblib'
    neural = None
    if current.is_file():
        previous = joblib.load(current)
        candidate = previous.get('neural_model', previous)
        trained = candidate.get('training_metadata', {})
        if (candidate.get('backend') == 'torch_spectrogram_ensemble'
                and trained.get('dataset_fingerprint') == dataset_hash
                and trained.get('epochs_per_member') == epochs
                and candidate.get('deep_config') == asdict(config)):
            neural = candidate
            print(f'Reusing completed {epochs}-epoch transformer members trained on the same {len(samples)} clips', flush=True)
    if neural is None:
        stage = output / 'neural_training'
        train_deep(dataset, stage, samples=samples, provenance=provenance,
                   epochs=epochs, batch_size=batch_size, device=device)
        neural = joblib.load(stage / 'keypress_model.joblib')

    def prepare(sample):
        return prepare_waveform(decode_audio(sample.path, config.sample_rate), config)

    with ThreadPoolExecutor(max_workers=4) as pool:
        waveforms = np.stack(list(pool.map(prepare, samples)))
    labels = np.asarray([sample.label for sample in samples])
    print(f'Fitting spectral experts on all {len(samples)} original clips plus training-only augmentations', flush=True)
    expert = fit_expert(waveforms, labels)
    for sample, item in zip(samples, metadata):
        if hashlib.sha256(Path(sample.path).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Training data changed; no hybrid was published')
    training = {
        **neural['training_metadata'],
        'model_kind': 'Pretrained Swin ensemble plus invariant acoustic experts',
        'training_sample_count': len(samples), 'withheld_sample_count': 0,
        'dataset_fingerprint': dataset_hash,
        'neural_weight': NEURAL_WEIGHT, 'spectral_weight': 1 - NEURAL_WEIGHT,
        'spectral_expert_training': expert['training_metadata'],
        'neural_training_implementation_sha256': neural['training_metadata']['implementation_sha256'],
        'implementation_sha256': {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('hybrid_training.py', 'hybrid_keypress.py', 'spectral_expert.py', 'keypress_model.py', '2_train.py')
        },
        'benchmark_used_for_development': True,
        'external_test_data_used': True,
        'external_test_data_used_for_fitting': False,
        'evaluation_note': 'The existing 20-press recording guided architecture development. '
        'Its reported score is a development benchmark, not an independent estimate of live accuracy.',
    }
    artifact = {'schema_version': 2, 'backend': 'neural_spectral_hybrid',
                'neural_model': neural, 'spectral_expert': expert,
                'neural_weight': NEURAL_WEIGHT, 'classes': neural['classes'],
                'feature_config': neural['feature_config'], 'training_metadata': training}
    validate_artifact(artifact)
    report = {'schema_version': 2, 'backend': artifact['backend'], **training,
              'classes': artifact['classes'], 'sample_count': len(samples),
              'class_counts': {key: int(np.sum(labels == key)) for key in artifact['classes']},
              'samples': metadata, 'provenance': provenance, 'feature_config': artifact['feature_config']}
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.hybrid-training-', dir=output) as temporary:
        staging = Path(temporary)
        joblib.dump(artifact, staging / 'keypress_model.joblib', compress=3)
        report['model_sha256'] = hashlib.sha256((staging / 'keypress_model.joblib').read_bytes()).hexdigest()
        archive = _archive_previous_run(output)
        if archive:
            report['previous_model_archive'] = archive.relative_to(output).as_posix()
        (staging / 'training.json').write_text(json.dumps(report, indent=2) + '\n')
        summary = {key: report[key] for key in ('backend', 'classes', 'training_sample_count', 'withheld_sample_count',
                                               'parameter_count', 'model_sha256', 'benchmark_used_for_development')}
        summary['separate_test'] = None
        (staging / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (staging / 'training.json').replace(output / 'training.json')
        (staging / 'summary.json').replace(output / 'summary.json')
        (staging / 'keypress_model.joblib').replace(current)
    print(f"Saved neural/spectral hybrid: {training['parameter_count']:,} neural parameters; all {len(samples)} clips; withheld=0", flush=True)
    return report
