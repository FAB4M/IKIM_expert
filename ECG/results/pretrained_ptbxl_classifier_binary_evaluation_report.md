# Pretrained PTB-XL Classifier Binary Evaluation

A pretrained PTB-XL signal classifier was evaluated as the main pathology-detection expert within the ECG tool-agent architecture.

## Task

Binary ECG classification: normal vs. pathological.

## Dataset

- Evaluation examples: 1000
- Normal examples: 500
- Pathological examples: 500

## Results

- Accuracy: 0.921
- Macro F1-score: 0.921
- Weighted F1-score: 0.921

## Class-wise performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| normal | 0.907 | 0.938 | 0.922 | 500 |
| pathological | 0.936 | 0.904 | 0.920 | 500 |

## Confusion matrix

| True / Predicted | normal | pathological |
|---|---:|---:|
| normal | 469 | 31 |
| pathological | 48 | 452 |

## Interpretation

The pretrained PTB-XL signal classifier achieved strong binary performance and is suitable as the main pathology-detection expert in the ECG agent.
Compared with the previously fine-tuned Qwen-VL image model, the signal-based expert performed substantially better for normal/pathological classification.
Therefore, Qwen is used as the agentic reasoning and explanation layer rather than as the primary pathology detector.

## Limitations

- The classifier was evaluated on PTB-XL-derived data and should not be interpreted as externally clinically validated.
- False negatives remain clinically relevant, with 48 pathological ECGs classified as normal.
- The system is a research prototype and not a clinical diagnostic tool.