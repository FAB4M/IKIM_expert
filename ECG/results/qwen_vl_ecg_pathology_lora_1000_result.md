# Qwen-VL ECG Pathology Fine-Tuning Result

A Qwen2.5-VL-3B-Instruct model was fine-tuned with LoRA on a balanced ECG image-question-answer dataset for binary pathology recognition.

## Task

Input: rendered 12-lead ECG image + question.
Output: classification as normal or pathological, PTB-XL label(s), and short interpretation.

## Dataset

- 1,000 ECG image-question-answer examples
- 500 normal examples
- 500 pathological examples
- Pathological examples sampled from MI, STTC, CD, and HYP

## Normal example

ECG ID: 7609
True answer: Classification: normal. PTB-XL label: NORM (normal ECG). Interpretation: This ECG is most consistent with a normal ECG.

Model output:
Classification: normal. PTB-XL label: NORM (normal ECG). Interpretation: This ECG is most consistent with a normal ECG.

## Pathological example

ECG ID: 6869
True answer: Classification: pathological. PTB-XL label(s): CD (conduction disturbance), NORM (normal ECG). Interpretation: This ECG shows pathological PTB-XL diagnostic superclass label(s).

Model output:
Classification: normal. PTB-XL label: NORM (normal ECG). Interpretation: This ECG is most consistent with a normal ECG.

## Interpretation

This experiment represents the main multimodal Qwen-based pathology-recognition component of the project. It uses ECG images directly as visual input and trains Qwen-VL to generate structured normal/pathological interpretations.