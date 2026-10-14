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

## 10. Anti-spoofing robustness evaluation

The original anti-spoofing training script uses a seeded random image-level validation split. The committed perfect validation score should therefore be treated only as a project-specific development result, not as evidence of robustness to new presentation attacks.

For a separate robustness check, collect fresh local evaluation sessions by category. Example:

```powershell
python collect_anti_spoof_eval.py --category real --session daylight01
python collect_anti_spoof_eval.py --category print --session print01
python collect_anti_spoof_eval.py --category phone --session phone01
python collect_anti_spoof_eval.py --category monitor --session monitor01
python collect_anti_spoof_eval.py --category replay --session replay01
```

The collector stores images under the ignored `data/anti_spoof_eval/` tree. Collect multiple short sessions with different angles, distances, and lighting conditions rather than a large number of nearly identical adjacent frames.

Then run:

```powershell
python benchmark_anti_spoof.py
```

The benchmark uses the deployed runtime threshold (`real_threshold = 0.65`) and reports overall accuracy, per-category spoof rejection, and PAD-style APCER/BPCER/ACER values. Outputs are written to `results/anti_spoofing_robustness/`.

These metrics describe only the locally collected evaluation set and are **not** a certified or universal face presentation-attack benchmark. Raw face media should remain private unless explicit consent permits publication.
