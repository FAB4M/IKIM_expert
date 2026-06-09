# Quantitative Evaluation of Clean Qwen-VL ECG Pathology Model

A Qwen2.5-VL-3B-Instruct model was fine-tuned with LoRA on a clean binary ECG pathology dataset.

## Task

Input: rendered 12-lead ECG image + classification question.
Output: structured binary classification as normal or pathological with PTB-XL label explanation.

## Dataset

- Clean binary pathology dataset
- Normal examples: diagnostic_classes == ['NORM']
- Pathological examples: contain MI, STTC, CD, or HYP and do not contain NORM
- Evaluation set size: 100 examples

## Results

- Accuracy: 0.58
- Macro F1-score: 0.576
- Weighted F1-score: 0.580

## Class-wise performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| normal | 0.618 | 0.618 | 0.618 | 55 |
| pathological | 0.533 | 0.533 | 0.533 | 45 |

## Confusion matrix

| True / Predicted | normal | pathological |
|---|---:|---:|
| normal | 34 | 21 |
| pathological | 21 | 24 |

## Interpretation

The clean Qwen-VL model performed above chance on binary ECG pathology recognition and learned the desired structured output format.
However, the model still missed a substantial number of pathological ECGs, with 21 pathological examples classified as normal.
This indicates that the current approach is a proof of concept rather than a clinically reliable model.

## Methodological conclusion

The experiment demonstrates that multimodal Qwen-VL LoRA fine-tuning on rendered ECG images is technically feasible.
Further improvements should focus on larger and cleaner training datasets, higher-resolution ECG image inputs, stronger answer-only loss masking, and potentially raw ECG signal models.