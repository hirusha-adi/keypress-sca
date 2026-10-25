# Numpad audio: visualize, split, train, and predict

Install FFmpeg (including `ffprobe` and the `libmp3lame` encoder). The prepared Python environment is `.venv`; activate it with `source .venv/bin/activate`, or use `.venv/bin/python` directly. To set up another checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/1_numpad/requirements-lock.txt
```

Python 3.12 was used for the saved models. `requirements-lock.txt` records the exact dependency versions; `requirements.txt` lists direct dependencies. Use the locked versions when loading the saved models, or retrain after changing dependencies.

From the repository root, with the environment active:

```bash
python scripts/1_numpad/0_visualize.py
python scripts/1_numpad/1_keypresses.py --dry-run
python scripts/1_numpad/1_keypresses.py
python scripts/1_numpad/2_train.py
python scripts/1_numpad/3_predict.py path/to/new_keypress.m4a
```

## Visualize recordings

`0_visualize.py` saves `work/1_numpad/rms_peaks.png`: one RMS envelope per key, numbered detected presses, detection threshold and P20 noise estimate lines, and P10/P20/P50/P80/P95 RMS percentile summaries. All ten plots share a linear amplitude scale. Analysis uses the original recordings before the splitter's output gain.

The visualizer shares the splitter's detector and remains usable when a recording has an unexpected count or no safe split gaps. This makes it useful for inspecting detection settings before splitting. Both scripts accept `--threshold-db` to override the adaptive detection threshold.

Paths default to the repository's dataset and work directories, regardless of the current working directory. Use `--dataset-dir PATH`, `--output PATH.png`, or `--dpi 240` to change the input, image location, or resolution. Re-running the visualizer replaces its output image.

## Split recordings

The script reads `datasets/1_numpad/0.m4a` through `9.m4a` and writes numbered clips such as `datasets/1_numpad/0/1.mp3`. The default input location works from any working directory. Use `--dataset-dir PATH` for a different directory containing the same filenames, or `--overwrite` to regenerate existing clips. Overwriting also removes obsolete numbered MP3s in those output folders.

## How it avoids cutting keystrokes

- Detects transients using 5 ms RMS energy across both channels, with a threshold estimated separately for each recording.
- Groups the press and release transients using a 350 ms minimum interval between detected presses. These recordings contain slow, separated presses; this setting is unsuitable for rapid typing.
- Places each cut inside a quiet window of at least 50 ms, at least 200 ms after the preceding press peak and 100 ms before the next. It refuses to export if no safe gap exists.
- Retains every decoded sample, including leading/trailing quiet and the silence between presses. Clips therefore have variable durations. No onset, release, or decay is trimmed away; no denoising, fades, channel mixing, or resampling are applied to the exported audio.
- Validates that each recording contains 25 or 26 detections before writing anything. It never forces a recording to have that count. If inspection justifies another count, use `--expected-counts`; `--threshold-db` overrides detection sensitivity.

MP3 uses the original sample rate and channel count with LAME's highest VBR quality (`-q:a 0`). MP3 is lossy; retain the original M4As for research. A single shared gain across all recordings leaves encoding headroom when decoded AAC peaks exceed full scale, preserving relative amplitudes between keys and clips.

All clips are encoded in a temporary directory before installation. Original recordings are never changed. `datasets/1_numpad/split_manifest.json` records source hashes, thresholds, the shared gain, and the exact source sample boundaries of every clip.

For the supplied recordings, detection finds **25 presses per key except 6, which has 26: 251 clips total**. Automatic counts and quiet boundaries are safeguards; listen to representative outputs before using the dataset for training.

## Train the classifier

```bash
python scripts/1_numpad/2_train.py
```

`2_train.py` reads the numbered MP3s in `datasets/1_numpad/0/` through `9/`. Folder names are the class labels. It compares four feature representations (MFCC statistics, log-mel statistics, a log-mel patch, and temporal log-mel summaries) with an RBF support vector classifier using `C=1` or `C=10`. The selected model uses **40 MFCC features and C=10**.

Training and prediction share the same preprocessing in `keypress_model.py`: decode to mono at 24 kHz, identify one isolated press, and align a 320 ms window beginning 40 ms before the strongest 5 ms RMS frame. Short context is padded. The window is DC-centered and peak-normalized, then converted to a 64-band mel spectrogram using a 512-sample Hann window and 128-sample hop. The selected representation contains the temporal means and standard deviations of 20 MFCCs. This avoids using clip length or long leading silence as input features.

Feature scaling is fitted inside each classifier pipeline. Model selection uses only the first 19 presses per key: **190 development clips**. Four contiguous chronological validation blocks per key are used, with the immediately neighboring press on either side excluded from that fold's training set. Clip 20 from each key is excluded as a boundary gap. The last 20% of each key's clips (21 onward: **51 clips**) are evaluated only after selecting the feature representation and C.

Results for the supplied dataset:

| Measure | Result |
| --- | --- |
| Development validation accuracy | 169/190 (88.95%) |
| Chronological holdout accuracy | 47/51 (92.16%) |
| Holdout top-3 accuracy | 50/51 (98.04%) |
| Holdout macro-F1 | 0.918 |

Each key currently has one source recording, so recording conditions and key labels are confounded. These results measure later presses from the same recordings. Performance on a new recording session, microphone position, phone, or keyboard has not been measured. For that assessment, collect several sessions containing all ten keys and reserve complete sessions for evaluation. The evaluation design follows scikit-learn's guidance on [preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html) and [correlated observations in validation](https://scikit-learn.org/stable/modules/cross_validation.html).

Training writes the following files to `work/1_numpad/`:

| File | Purpose |
| --- | --- |
| `keypress_model.joblib` | Model for new predictions, refitted on all 251 clips after evaluation |
| `evaluation_model.joblib` | Model fitted on 190 development clips, for reproducing the held-out results |
| `evaluation.json` | Metrics, selected settings, candidate scores, exact split membership, source/clip hashes, and library versions |
| `heldout_predictions.csv` | Actual and predicted keys, plus top-three ranking scores for each reserved clip |
| `confusion_matrix.png` | Actual versus predicted key counts on the 51 reserved clips |

The final model includes the former holdout clips in its training set. To reproduce the reported evaluation, use `evaluation_model.joblib`. Re-running training replaces these five artifacts. `--dataset-dir PATH` and `--output-dir PATH` override the defaults, which otherwise work from any working directory.

## Predict a key from audio

```bash
python scripts/1_numpad/3_predict.py path/to/new_keypress.m4a
python scripts/1_numpad/3_predict.py path/to/new_keypress.wav --json
```

The input must contain one isolated keypress, including its release. MP3, M4A, WAV, and other FFmpeg-readable formats are supported; input sample rate and duration may differ from the training clips. The command prints the predicted key and the top three ranked SVM scores. These scores are not probabilities. Use `--top-k 1` for only the highest-ranked key, or `--model PATH` to load a different trained artifact.

Silence, stationary background noise, invalid audio, and multiple well-separated presses are rejected. The transient check groups press/release sounds within 350 ms, matching these slow recordings; it is not a rapid-typing segmenter. The classifier chooses among `0` through `9` only and has no learned class for other keys or unrelated transient sounds.

## Verify the pipeline

```bash
python -m unittest discover -s scripts/1_numpad -p 'test_*.py' -v
```

The 16 tests cover safe segmentation, chronological split isolation, numeric clip ordering, fixed feature dimensions, level/leading-silence invariance, invalid input rejection, decoding parity, and persisted model/configuration predictions.
