# Clean Qwen-VL ECG Pathology Fine-Tuning Result

A clean binary ECG pathology dataset was created for Qwen2.5-VL fine-tuning.
Ambiguous multi-label cases containing both NORM and pathological labels were excluded.

## Task

Input: rendered 12-lead ECG image + classification question.
Output: structured classification as normal or pathological, PTB-XL label(s), and short interpretation.

## Dataset Definition

- Normal examples: diagnostic_classes == ['NORM']
- Pathological examples: contain MI, STTC, CD, or HYP and do not contain NORM

## Model

- Base model: Qwen/Qwen2.5-VL-3B-Instruct
- Fine-tuning method: LoRA
- Quantization: 4-bit
- Task: multimodal ECG image-question-answer fine-tuning

## Qualitative Evaluation

### Pathological example

ECG ID: 6469
True answer: Classification: pathological. PTB-XL label(s): CD (conduction disturbance), MI (myocardial infarction). Interpretation: This ECG shows pathological PTB-XL diagnostic superclass label(s).

Model output: Classification: pathological. PTB-XL label(s): CD (conduction disturbance), MI (myocardial infarction). Interpretation: This ECG shows pathological PTB-XL diagnostic superclass label(s).

### Normal example

ECG ID: 7609
True answer: Classification: normal. PTB-XL label(s): NORM (normal ECG). Interpretation: This ECG is most consistent with a normal ECG.

Model output: Classification: normal. PTB-XL label(s): NORM (normal ECG). Interpretation: This ECG is most consistent with a normal ECG.

## Interpretation

The clean Qwen-VL pathology fine-tuning run successfully learned the structured output format and correctly classified one normal and one clearly pathological ECG example.
Compared with the previous ambiguous dataset, removing NORM-plus-pathology cases improved qualitative behavior.
This result represents the main multimodal Qwen-based pathology-recognition component of the project.