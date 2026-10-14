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

## 6. Sad / Neutral webcam adaptation

Private participant media is not committed.

The released adapter originally used two participants. For the expanded workflow, collect additional **consented, pseudonymous participants** and keep person identities separated. A minimum of four participants is enforced by the new trainer; **six participants (p01–p06) with at least 20 Neutral and 20 Sad sequences per person are recommended** for the next benchmark.

Example collection for four new participants:

```powershell
python collect_webcam_emotions.py --person p03 --target-per-class 20
python collect_webcam_emotions.py --person p04 --target-per-class 20
python collect_webcam_emotions.py --person p05 --target-per-class 20
python collect_webcam_emotions.py --person p06 --target-per-class 20
```

The final adapter is binary, so this collector now records only **Neutral** and **Sad**.

Feature extraction automatically discovers every participant folder:

```powershell
python extract_webcam_features.py
```

Retrain with leave-one-person-out cross-validation:

```powershell
python train_sad_neutral_adapter.py --min-persons 4 --min-sequences-per-class 20
```

The trainer now:
- evaluates every participant as an unseen validation person once;
- aggregates out-of-fold predictions across participants;
- fits temperature scaling from out-of-fold probabilities;
- reports calibration error before and after scaling;
- selects a confidence threshold from out-of-fold predictions;
- saves the chosen temperature and threshold to `models/sad_neutral_adapter_calibration.json`;
- trains the final adapter on all eligible participants only after evaluation/calibration is complete.

At runtime, `emotion_detector_5class.py` automatically loads the saved calibration metadata. If the calibration file is absent, it safely falls back to temperature `1.0` and confidence threshold `0.70`.

Do not publish participant face data without explicit consent. Increasing participant count can improve generalization, but an accuracy increase must be measured after retraining rather than assumed.

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

## 9. Benchmark the final five-class deployment

After the actor-wise test features exist under `data/lstm_features/test/`, run:

```powershell
python benchmark_final_5class.py
```

This evaluates the existing unseen-actor test split (Actors 21–24) in two modes:

1. **Fusion only** — Fear is merged into Surprise and Disgust into Sad.
2. **Adapter assisted** — the same five-class fusion plus the confidence-gated Sad/Neutral adapter.

Outputs are written to `results/final_5class/` and include text reports, confusion matrices, JSON metrics, and per-sequence predictions.

The benchmark is sequence-level and intentionally does **not** apply live webcam smoothing across independent RAVDESS sequences. Actors 21–24 were unseen during model training, but the repository also notes that this test split was used during earlier model comparison, so it should not be described as a pristine post-selection benchmark.
