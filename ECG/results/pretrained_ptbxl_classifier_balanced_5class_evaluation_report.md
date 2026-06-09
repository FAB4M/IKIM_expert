# Pretrained PTB-XL Classifier Balanced 5-Class Evaluation

A pretrained PTB-XL signal classifier was evaluated on a balanced 5-class subset with 200 examples per class.

## Task

Top-class prediction among the five PTB-XL diagnostic superclasses: NORM, MI, STTC, CD, and HYP.

## Dataset

- Total examples: 1000
- NORM: 200
- MI: 200
- STTC: 200
- CD: 200
- HYP: 200

## Results

- Balanced top-class accuracy: 0.682
- Macro F1-score: 0.664
- Weighted F1-score: 0.664

## Class-wise performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| NORM | 0.667 | 0.950 | 0.784 | 200 |
| MI | 0.648 | 0.800 | 0.716 | 200 |
| STTC | 0.578 | 0.800 | 0.671 | 200 |
| CD | 0.915 | 0.535 | 0.675 | 200 |
| HYP | 0.878 | 0.325 | 0.474 | 200 |

## Interpretation

The pretrained PTB-XL classifier showed moderate 5-class top-class performance on the balanced evaluation set.
NORM, MI, and STTC showed high recall, while CD and especially HYP showed lower recall.
Therefore, the model is suitable as a diagnostic superclass expert, but its outputs should be interpreted as probabilistic tool outputs rather than definitive diagnoses.

## Role in the ECG agent

The model is used as the main signal-based expert for PTB-XL superclass probabilities.
Qwen acts as the agentic explanation layer and integrates the classifier output with rhythm features from NeuroKit2.