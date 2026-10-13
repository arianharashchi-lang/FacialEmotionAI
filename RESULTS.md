# Results

This page summarizes the evaluation artifacts committed with FacialEmotionAI.

## Evaluation protocol

The core CNN/LSTM emotion results use an **actor-wise RAVDESS split**:

- Train: Actors 01–16
- Validation: Actors 17–20
- Test: Actors 21–24

This is stricter than a random frame split because identities in the test set are unseen during training.

## CNN baseline

- Test Accuracy: **41.49%**
- Macro F1: **40.00%**
- Weighted F1: **40.22%**

![CNN confusion matrix](results/cnn/cnn_test_confusion_matrix.png)

Raw report: [results/cnn/cnn_test_report.txt](results/cnn/cnn_test_report.txt)

## LSTM V1 Rebuild

- Test Accuracy: **45.91%**
- Macro F1: **43.92%**
- Weighted F1: **43.38%**

![LSTM V1 confusion matrix](results/lstm_v1/lstm_attention_v1_confusion_matrix.png)

Raw report: [results/lstm_v1/lstm_attention_v1_test_report.txt](results/lstm_v1/lstm_attention_v1_test_report.txt)

The temporal model improved test accuracy by about **4.42 percentage points** over the static CNN baseline.

## Temporal experiments

| Model | Test Accuracy | Macro F1 | Status |
|---|---:|---:|---|
| V1 Rebuild | **45.91%** | **43.92%** | Selected |
| V2 | 43.27% | 36.18% | Experiment |
| V3 | 44.23% | 41.95% | Experiment |
| V4 | 43.27% | 41.82% | Experiment |

Experimental artifacts are kept under `results/experiments/` for transparency.

## Sad / Neutral adapter

Person-wise cross-validation across two participants:

- Mean Accuracy: **63.75%**
- Mean Macro F1: **57.22%**
- Mean Neutral F1: **69.90%**
- Mean Sad F1: **44.55%**

Raw metrics:
- [results/adapter/sad_neutral_adapter_metrics.json](results/adapter/sad_neutral_adapter_metrics.json)
- [results/adapter/sad_neutral_adapter_cv_report.txt](results/adapter/sad_neutral_adapter_cv_report.txt)

### Important limitation

This adapter was evaluated on only two participants. The score is useful as a development signal, but it is **not a population-level estimate**.

## Anti-spoofing

The committed anti-spoofing metric file reports perfect performance on the project-specific evaluation split. That result should be interpreted only in the context of the collected dataset and split, not as universal spoof-detection accuracy.

See:
- [results/anti_spoofing/fake_detector_metrics.json](results/anti_spoofing/fake_detector_metrics.json)
- [results/anti_spoofing/fake_detector_confusion_matrix.png](results/anti_spoofing/fake_detector_confusion_matrix.png)

## Final five-class deployment

The live system merges:
- Fear into Surprise
- Disgust into Sad

and optionally refines Sad/Neutral using the confidence-gated adapter.

The reported CNN/LSTM test metrics above are the original **7-class actor-wise evaluation**. A separate held-out benchmark of the final fused five-class deployment has not yet been performed, so this repository does not claim a new five-class test accuracy.
