# FacialEmotionAI

Real-time facial emotion recognition with **OpenCV**, **MobileNetV2 CNN feature extraction**, **BiLSTM + Temporal Attention**, a **Real/Fake anti-spoofing gate**, and a confidence-gated **Sad/Neutral webcam adapter**.

## Final emotion classes

The deployed system outputs five classes:

- Angry
- Happy
- Surprise = original Surprise + Fear
- Sad = original Sad + Disgust
- Neutral

The class fusion was introduced after real-time testing showed strong visual overlap between **Fear/Surprise** and **Disgust/Sad** in the original 7-class setup.

## Final pipeline

```text
Webcam
  ↓
OpenCV Face Detection
  ↓
Real / Fake Gate
  ↓
CNN Feature Extractor (MobileNetV2)
  ↓
256-D feature vector per sampled frame
  ↓
Rolling Buffer: 20 × 256
  ↓
BiLSTM
  ↓
Temporal Attention
  ↓
Original 7-class probabilities
  ↓
Class Fusion
  ├── Fear + Surprise  → Surprise
  └── Disgust + Sad    → Sad
  ↓
5-class output
  ↓
Sad / Neutral Adapter
  └── used only when adapter confidence ≥ 70%
```

## Architecture

![System architecture](docs/architecture.png)

## Dataset and evaluation protocol

The temporal emotion model was trained using **RAVDESS** sequences.

Actor-wise split:

```text
Actors 01–16 → Train
Actors 17–20 → Validation
Actors 21–24 → Test
```

Each temporal sample contains 20 frames at 160 × 160 × 3. After CNN feature extraction, each sequence is represented as **20 × 256**.

## Main results

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| CNN baseline | 41.49% | 40.00% |
| LSTM V1 Rebuild | **45.91%** | **43.92%** |
| LSTM V2 | 43.27% | 36.18% |
| LSTM V3 | 44.23% | 41.95% |
| LSTM V4 | 43.27% | 41.82% |

The final temporal backbone is **LSTM V1 Rebuild**.

### Sad / Neutral webcam adapter

A small person-wise webcam adaptation dataset was collected from two people. The final binary adapter was trained only for **Sad vs Neutral**.

```text
Mean Accuracy   : 63.75%
Mean Macro F1   : 57.22%
Mean Neutral F1 : 69.90%
Mean Sad F1     : 44.55%
```

These adapter metrics come from only two participants and should **not** be interpreted as general population performance.

## Runtime behavior

- Emotion inference runs only for faces classified as REAL.
- A rolling temporal buffer stores the latest 20 CNN feature vectors.
- The final output contains five emotion classes.
- The Sad/Neutral adapter is used only when its confidence is at least 70%.
- Results are logged to SQLite.
- Captured face images can be stored locally for debugging/evaluation.

## Final runtime files

```text
main_5class.py
emotion_detector_5class.py
fake_detector.py

models/
├── cnn_feature_extractor.keras
├── lstm_attention_v1_best.keras
├── sad_neutral_adapter.keras
├── fake_discriminator.keras
├── lstm_v1_feature_mean.npy
├── lstm_v1_feature_std.npy
├── lstm_attention_v1_classes.json
└── sad_neutral_adapter_classes.json
```

## Installation

Recommended Python version: **Python 3.11**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
python main_5class.py
```

Controls:

```text
R = Reset temporal buffer
Q = Quit
```

## Project structure

```text
FacialEmotionAI/
├── main_5class.py
├── emotion_detector_5class.py
├── fake_detector.py
├── models/
├── data/
├── database/
├── results/
├── docs/
│   ├── architecture.png
│   └── screenshots/
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

## Notes

- RAVDESS data is not included in this repository.
- Raw webcam adaptation data should not be committed.
- Large model files should be stored with **Git LFS** when necessary.
- TensorFlow native Windows builds may run inference on CPU depending on version and hardware.
- Softmax values are model confidence scores, not literal psychological certainty.

## Limitations

- RAVDESS contains acted emotions and a limited number of identities.
- Real-world lighting, pose, camera quality and identity variation can reduce performance.
- Sad/Neutral adaptation was evaluated on only two participants.
- This is a research / portfolio project, not a clinical or psychological diagnostic system.

## Future work

- Collect a larger multi-person real-world dataset.
- Evaluate the final 5-class system on a separate unseen-person test set.
- Improve anti-spoofing robustness.
- Add face tracking for more stable identity continuity.
- Calibrate probabilities and thresholds on a larger validation set.
