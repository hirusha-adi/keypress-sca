# Related work and project scope

I started this repository because I wanted to play around with recognizing keypress sounds. The underlying idea is already well established. This implementation is a learning project and practical experiment, with no claim of research novelty, a new attack, or state-of-the-art accuracy.

## Papers worth reading

| Paper | Relevance |
| --- | --- |
| Asonov and Agrawal, **Keyboard Acoustic Emanations**, IEEE Symposium on Security and Privacy, 2004. [IBM Research](https://research.ibm.com/publications/keyboard-acoustic-emanations) | Early demonstration that key sounds can reveal key identity, including variation in capture conditions. |
| Zhuang, Zhou, and Tygar, **Keyboard Acoustic Emanations Revisited**, ACM CCS, 2005. [Author-hosted paper](https://people.eecs.berkeley.edu/~tygar/papers/Keyboard_Acoustic_Emanations_Revisited/preprint.pdf) | Combines acoustic features, statistical learning, and language constraints; demonstrates that labelled training is not required in every attack setting. |
| Harrison, Toreini, and Mehrnezhad, **A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards**, IEEE EuroS&P Workshops, 2023. [Paper](https://arxiv.org/abs/2308.01074) | Modern spectrogram-based deep learning with phone and Zoom recordings. Reports 95% and 93% in its own experimental settings. |
| Ayati, Park, Cai, and Botacin, **Making Acoustic Side-Channel Attacks on Noisy Keyboards Viable with LLM-Assisted Spectrograms' “Typo” Correction**, USENIX WOOT, 2025. [Proceedings and paper](https://www.usenix.org/conference/woot25/presentation/ayati) | Explores pretrained vision transformers for key classification and language-model correction of sequences under noise. |
| Maurya, Choudhury, Agarwal, and Buduru, **DECKER: Domain-invariant Embedding for Cross-Keyboard Extraction and Recognition**, AsiaCCS, 2026. [Paper](https://arxiv.org/abs/2605.03384) | Studies generalization across keyboards, users, and capture conditions, using data from 53 participants and 37 laptop keyboards. |

These papers address substantially broader questions than this four-key prototype. Their datasets, class counts, capture conditions, and evaluation methods differ. The local 20/20 score is not a direct comparison with their reported accuracy.

## What this repository explores

The practical exercise here is connecting recording inspection, conservative segmentation, feature extraction, model training, frozen evaluation, and microphone streaming into a working pipeline. It compares a small conventional model with a much larger pretrained ensemble and a hybrid on a small personal dataset.

The model does not use language modelling or text correction. It is currently restricted to the trained keyboard setup and the supported class set. This is a limitation of this implementation; the literature already explores transfer and more general attack settings.

## Possible follow-ups

The most useful next step would be independently recorded, interleaved-key sessions with final test sessions reserved before model changes. Confidence calibration and selective prediction would then be interesting to measure: when does the recognizer make a confident mistake, and how much accuracy can be gained by abstaining on uncertain events?

Calibration and rejection are established methods. A basic reference is Guo et al., **On Calibration of Modern Neural Networks**, ICML 2017 ([paper](https://proceedings.mlr.press/v70/guo17a.html)). Applying temperature scaling does not automatically detect unfamiliar keys or unrelated sounds. Those need their own evaluation examples and metrics.

No calibration, cross-session generalization, or unknown-key detection result is claimed here. These are ideas for further experiments, and any publication proposal would need a focused literature review and a clearer contribution.
