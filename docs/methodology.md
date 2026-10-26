# Methodology and experiment notes

This document describes the implementation in this repository. It is an exploratory application of established acoustic keystroke recognition methods, with no claim of a new attack or model architecture.

## Recordings and labels

Experiment 1 uses phone-recorded M4A files for numpad keys 0–9. There are 25 retained presses per key except key 6, which has 26, for 251 training clips. A separate recording spells `0789123456`, giving one example per key.

Experiment 2 uses recordings from the same LCS USB Audio Mono microphone setup intended for live prediction. Keys 0, 1, 2, and 3 have 176, 175, 178, and 177 retained detections respectively: 706 clips. A separate sequence spells `02310231023102310231`, giving five examples per class. The original 48 kHz stereo channels are identical, so the recording is effectively mono.

The public experiment-2 source files are audio-only M4A copies of the AAC streams in the original OBS MP4 files. Conversion copies the compressed stream without re-encoding. [Audio provenance](../datasets/audio_provenance.json) records original-container hashes, public-container hashes, and decoded-audio equality. Split MP3s are unchanged. Historical reports may therefore refer to original MP4 names/hashes; those are historical container identities, not the hashes of the public M4As.

Training labels come from directory names. Sequence ground truth comes from the recording filename and the ordered clip filenames. Neither source names nor expected sequences are input features.

## Detection and conservative splitting

The splitter computes a 5 ms RMS envelope, estimates the background, and identifies transient candidates. A 350 ms minimum interval groups press/release sounds into one event. The second experiment adds peak-width and local-background checks to suppress broad noise bumps.

A boundary must lie in a quiet interval of at least 50 ms, after a minimum 200 ms context from the previous impact and before a minimum 100 ms context preceding the next. If a safe boundary cannot be found, export fails rather than trimming an event to force a split. Filename-labelled sequence recordings must have exactly the expected number of detections.

The intervals cover the entire decoded recording with no gaps or overlap. Leading silence, trailing silence, and between-press background remain assigned to clips. Exports keep the source sample rate and channel count and use LAME's highest VBR quality. Shared headroom gain is computed separately for training and sequence recordings. MP3 encoding is lossy; retain the source audio for future preprocessing.

Detection counts do not independently establish that every faint candidate was a physical keypress. Interpret the retained events alongside the peak plots and representative listening checks.

## Experiment 1: feature-based classifiers

Audio is decoded to mono at 24 kHz, centered on the strongest 5 ms RMS frame, and aligned into 320 ms with 40 ms pre-peak context. The window is DC-centered and peak-normalized. A 512-sample Hann FFT and 128-sample hop feed 64 mel bands.

The search compares 72 fixed configurations spanning six feature profiles and RBF SVM, linear SVM, and shrinkage LDA settings. The selected representation has weighted means and standard deviations of 40 MFCCs, for 80 features, with LDA automatic covariance shrinkage. Frame weights are proportional to the square root of mel-frame power. Scaling is fitted inside the classifier pipeline.

Selection uses the first 19 clips per key: 190 unique development clips. Repeated chronological partitions with 3, 4, and 5 folds give 12 folds and 570 predictions. Neighboring clips are purged around validation blocks. The denominator is repeated predictions on the same 190 examples, not 570 independent clips.

Clip 20 per class supplies a boundary gap. Clips 21 onward give the 51-clip chronological benchmark, evaluated after selection. The final model is then fitted on all 251 clips. The saved historical evaluation describes the 190-clip model; the final artifact includes the former holdout. Both the chronological and separate sequence benchmarks had already been observed, which limits claims of an untouched final evaluation.

## Experiment 2: neural branch

The frontend resamples to 32 kHz and aligns 320 ms with 80 ms of pre-peak context. It creates a 224-band mel spectrogram using FFT 1024, hop 128, and an upper frequency of 15.5 kHz. The log-mel image is resized to 224 × 224 and represented as three identical channels with ImageNet normalization.

One torchvision Swin-B and two independently seeded Swin-T models are initialized with ImageNet-1K weights. Their four-class versions contain 86,747,324, 27,522,430, and 27,522,430 parameters: 141,792,184 total. The architecture family is described by the [Swin Transformer paper](https://arxiv.org/abs/2103.14030); pretrained implementations are provided by [torchvision](https://docs.pytorch.org/vision/stable/models.html).

| Training setting | Value |
| --- | --- |
| Original training clips | All 706, zero withheld |
| Epochs | 80 per member |
| Seeds | 42, 43, 44 |
| Optimizer | AdamW |
| Backbone / head learning rates | 5e-5 / 5e-4 |
| Weight decay / label smoothing | 0.01 / 0.03 |
| Schedule | Five warmup epochs, then cosine |
| Physical / effective batch size | 8 / 32, using gradient accumulation |
| Gradient clipping | Norm 1.0 |
| Recorded training device | RTX 5060, 8 GB VRAM |
| Precision on the recorded run | BF16 training, compressed FP16 stored weights |

Training-only augmentations include up to ±20 ms alignment shift, additive noise at 20–40 dB SNR, smooth random EQ, and small time/frequency masks. Pitch shifting is avoided because the spectral differences may encode the key identity. Inference averages the three members' softmax probability vectors.

Training epoch accuracy is measured on augmented training examples. There is no internal validation split in the default experiment-2 training path, so those curves describe optimization rather than generalization.

## Experiment 2: acoustic experts and fusion

The acoustic branch uses the same prepared 320 ms waveform and three feature views:

| View | Representation | Features |
| --- | --- | ---: |
| Global spectrum | Energy-weighted mean and standard deviation of 96 log-mel bands | 192 |
| Temporal MFCC | 40 coefficients in intervals −20–25, 25–75, and 75–180 ms relative to impact | 120 |
| Attack spectrum | Weighted statistics of 128 bands over 160 ms, starting 8 ms before the peak; FFT 2048 | 256 |

Each view fits a standardized linear SVM with C=0.01 and a shrinkage LDA with shrinkage 0.1. Three EQ/noise variants supplement every original clip, giving 2,824 fitting rows for each of six heads. These are augmented versions of 706 recordings, not 2,824 independent captured examples.

The six acoustic-head probability vectors are averaged. Linear SVM scores are converted by softmax; this does not calibrate them. Final fusion is:

```text
P(key | audio) = 0.80 × mean(acoustic-head probabilities)
               + 0.20 × mean(Swin probabilities)
```

The highest value selects the key. Confidence is the corresponding fused value; the hybrid margin is the gap between the two largest log probabilities. Both are model outputs, not measured accuracy. Feature design and fusion were developed while consulting the existing twenty-press benchmark.

## File inference, live capture, and replay

Single-file inference expects one complete isolated press. Live capture operates at the microphone's native rate, estimates background during startup, and buffers event context. Complete events are resampled and passed through the same model preprocessing. The large networks are loaded and warmed before capture begins to avoid startup delays overflowing the input queue.

Streaming quality checks reject some incomplete or ambiguous events. They do not constitute a trained unknown-key detector. A clean transient from another key or another object can still be assigned a supported class.

Replay feeds an audio file through the streaming detector without opening a microphone. The original twenty-press sequence produced 20 clean events, all correctly classified by the final hybrid. It is still the existing development recording, so replay is a pipeline check rather than a new data source.

With `--expected-sequence`, scoring compares detected events to labels by position and counts rejected, extra, and missing events as errors. An insertion or deletion can shift later comparisons. Independently timestamped labels would be stronger for rapid or ambiguous sequences. The expected sequence is used for scoring only.

## Evidence and reproducibility limits

The [results directory](../results/README.md) contains selected reports, predictions, figures, and source-file hashes. Models and scratch outputs stay local in `work/`. Scripts resolve their default paths from the repository location, and dependency files record the original environment.

Public audio-only remuxing changes container hashes even when decoded samples are identical. The provenance report explains that mapping. Model code hashes in historical reports describe the recorded run, and are not rewritten to pretend that later documentation or source edits were part of that run.

GPU kernels, dependency changes, and pretrained-weight availability may affect retraining. Re-running a stochastic training recipe is not a promise of byte-identical artifacts or a guaranteed 20/20 result.

The strongest current limitation is the data collection design: one source recording per key. A better assessment would randomize all keys within several independent sessions, separate fitting/calibration/final test sessions, and freeze feature and fusion decisions before the final test.
