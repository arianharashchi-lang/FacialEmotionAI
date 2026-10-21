# Release Notes

## Unreleased — Multilingual web interface

### Web demo
- Added a language selector for 12 interface languages: English, Persian, Arabic, Spanish, French, German, Chinese, Japanese, Portuguese, Hindi, Turkish, and Russian.
- English is selected by default on every visit.
- Language choice is session-only and is not persisted for future visits.
- Persian and Arabic use right-to-left layout; other languages use left-to-right layout.
- Translated interface labels, status messages, and emotion names.

### Compatibility
- No intended changes to the trained emotion models or inference API.


## v1.0.0 — First stable portfolio release

### Runtime

- Real-time OpenCV face detection.
- Real/Fake anti-spoofing gate.
- MobileNetV2 CNN feature extraction.
- 20-frame temporal feature buffer.
- BiLSTM + Temporal Attention.
- Final five-class live output:
  - Angry
  - Happy
  - Surprise
  - Sad
  - Neutral
- Fear is merged into Surprise.
- Disgust is merged into Sad.
- Confidence-gated Sad/Neutral adapter.
- SQLite runtime logging.

### Final model artifacts

- `cnn_feature_extractor.keras`
- `fake_discriminator.keras`
- `lstm_attention_v1_best.keras`
- `lstm_attention_v1_classes.json`
- `lstm_v1_feature_mean.npy`
- `lstm_v1_feature_std.npy`
- `sad_neutral_adapter.keras`
- `sad_neutral_adapter_classes.json`

### Core evaluation

- CNN test accuracy: 41.49%
- CNN Macro F1: 40.00%
- LSTM V1 test accuracy: 45.91%
- LSTM V1 Macro F1: 43.92%

### Repository quality

- Final models stored through Git LFS.
- Raw datasets and private webcam media excluded.
- Legacy runtime prototypes archived.
- Results organized by experiment.
- Reproducibility and limitations documented.
- CI syntax smoke test added.
- MIT license added.

### Known limitations

See [RESULTS.md](RESULTS.md) and [README.md](README.md).
