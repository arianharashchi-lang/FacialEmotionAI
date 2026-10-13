# Reproducibility

This repository includes the code required to rebuild the main training pipeline while intentionally excluding raw datasets and private webcam captures.

## Environment

Recommended:

- Python 3.11
- TensorFlow 2.x
- OpenCV
- NumPy
- scikit-learn

Install:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Prepare RAVDESS

The project uses the official RAVDESS video speech data.

```powershell
python download_prepare_ravdess.py
python prepare_ravdess_sequences.py
```

The preparation code maps the seven used RAVDESS emotion labels and excludes the Calm class.

## 2. Create the actor-wise CNN split

```powershell
python prepare_cnn_actor_split.py
```

Split:

```text
Actors 01–16 → train
Actors 17–20 → validation
Actors 21–24 → test
```

## 3. Train the CNN

```powershell
python train_emotion_cnn.py
```

The CNN script supports two MobileNetV2 weight paths:

1. If the local `models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5` file exists, it is used.
2. Otherwise Keras attempts to load the standard ImageNet weights.

The local weight file is intentionally ignored by Git.

The training script saves the 256-D feature extractor used by the temporal model.

## 4. Extract temporal features

```powershell
python extract_lstm_features.py
```

Expected feature shape per sequence:

```text
20 × 256
```

## 5. Train the selected temporal model

```powershell
python train_lstm_attention_v1_rebuild.py
```

The selected runtime checkpoint is:

```text
models/lstm_attention_v1_best.keras
```

V2/V3/V4 are retained as research experiments, not runtime dependencies.

## 6. Optional webcam adaptation

Private adaptation media is not committed.

Collection:

```powershell
python collect_webcam_emotions.py --person p01
python collect_webcam_emotions.py --person p02
```

Feature extraction:

```powershell
python extract_webcam_features.py
```

Final binary adapter:

```powershell
python train_sad_neutral_adapter.py
```

Do not publish participant face data without explicit consent.

## 7. Anti-spoofing

Local Real/Fake examples are stored under ignored data directories. The released runtime uses the trained discriminator model:

```text
models/fake_discriminator.keras
```

The current released training path is supervised Real/Fake discriminator training.

## 8. Run the final application

```powershell
python main_5class.py
```

## Reproducibility caveats

- RAVDESS is acted and identity-limited.
- The webcam adapter dataset is intentionally private and small.
- Random seeds are set in the main training scripts where practical, but exact reproducibility can still vary across TensorFlow versions and hardware.
- The final class-fusion deployment has not been independently re-benchmarked on a new held-out five-class test set.
