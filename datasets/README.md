# Recordings and labels

The data contains my own numpad recordings for an exploratory sound-classification project. The sources, prepared clips, and split manifests are included so the experiments can be inspected and rerun. Generated model weights and scratch output are kept outside this directory.

| Directory | Contents | Training labels or expected sequence | Clip count |
| --- | --- | --- | ---: |
| `1_numpad/` | Original phone M4A files and numbered training clips | Numpad 0–9 | 251 |
| `1_tests/` | Separate phone sequence recording and labeled clips | `0789123456` | 10 |
| `2_numpad/` | Audio-only M4A recordings and numbered training clips | Numpad 0–3 | 706 |
| `2_tests/` | Separate audio-only M4A sequence recording and labeled clips | `02310231023102310231` | 20 |

Dataset 1 has 25 clips per key, except key 6 with 26. Dataset 2 has 176 clips for 0, 175 for 1, 178 for 2, and 177 for 3. Its recordings used the same LCS microphone setup as the live demonstration. Exact keyboard make/model and recording-session dates have not been documented here.

Dataset 2 was originally captured in MP4 containers that also contained screen video. Only audio is needed for this project. The public files were remuxed to M4A without re-encoding the audio; the private video originals stay local under the ignored `original_video/` directory. [`audio_provenance.json`](audio_provenance.json) records original/published source hashes and matching decoded-audio hashes. Historical experiment reports retain their original MP4 filenames and hashes, while the current split manifests identify the published M4A sources.

## File layout

Training labels come from directory names:

```text
2_numpad/
├── 0.m4a                 # audio-only source recording of key 0
├── 0/
│   ├── 1.mp3            # first detected press of key 0
│   └── 2.mp3
├── 1.m4a
├── 1/
└── split_manifest.json
```

Sequence labels come from the original filename. Each split clip is named `PRESS_NUMBER-ACTUAL_KEY.mp3`, with a one-based global press number:

```text
2_tests/
├── 02310231023102310231.m4a
├── 02310231023102310231/
│   ├── 1-0.mp3
│   ├── 2-2.mp3
│   ├── 3-3.mp3
│   └── ...
└── split_manifest.json
```

The filename supplies ground truth for evaluation. The classifier predicts from the audio; it does not receive the expected key.

## Segmentation and provenance

The splitter uses RMS peaks to detect slow, separated presses and places cuts in quiet gaps. It retains every decoded source sample in contiguous clip ranges, including background before and after the presses. It refuses an unsafe cut or a sequence detection count that disagrees with the filename. These safeguards do not prove that every detected training transient is a real press; listen to representative clips and inspect the plots when changing settings.

Each `split_manifest.json` records source SHA-256 hashes, sample rates, channels, detection settings, gain, labels, and exact source-sample boundaries. MP3 clips are lossy exports, so retain the source recordings when preparing new data. The redundant source ZIP archive is excluded from Git.

Prepared clips are already included. To inspect or regenerate them, follow the [dataset 1](../scripts/1_numpad/README.md#split-recordings) or [dataset 2](../scripts/2_numpad/README.md#sources-and-segmentation) instructions. `--dry-run` analyzes without writing. `--overwrite` replaces generated clips and removes stale generated filenames in the selected output folders.

## Evaluation limits

Each training key has one source recording. Key identity can therefore be confounded with recording conditions; many clips from one recording are not many independent recording sessions. These recordings do not establish transfer to a different keyboard, typist, microphone, or room.

The `*_tests` names describe their role in the original workflow. The supplied sequences have been inspected during development. In particular, the 20-event dataset 2 sequence informed model and fusion choices, so its final 20/20 score is a development benchmark. It is not a fresh test of live accuracy. Dataset 1's existing benchmarks were also observed before the improvement run. See the [curated results](../results/README.md) for the exact protocols and counts.
