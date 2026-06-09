# Qwen ECG Tool-Agent Fine-Tuning

A tool-augmented ECG agent was implemented using specialized ECG tools and a supervised Qwen explanation model.

## Tools

1. NeuroKit2 rhythm feature tool
- Input: PTB-XL raw ECG signal
- Output: estimated heart rate, number of detected R-peaks, RR regularity
- Scope: rhythm feature extraction only, not full arrhythmia diagnosis

2. Pretrained PTB-XL signal classifier
- Input: PTB-XL raw ECG signal
- Output: probabilities for NORM, MI, STTC, CD, and HYP
- Role: diagnostic superclass prediction

3. Qwen ECG agent
- Base model: Qwen/Qwen2.5-0.5B-Instruct
- Fine-tuning method: LoRA
- Input: structured tool outputs
- Output: cautious structured ECG explanation

## Training

- Training examples: 1000 ECG tool-output examples
- Train/evaluation split: 90/10
- Epochs: 2
- Final training loss: 0.5204
- Validation loss after epoch 2: 0.3941

## Qualitative Evaluation Example

For ECG ID 8609, the Qwen agent correctly reproduced the rhythm features, PTB-XL classifier outputs, predicted labels, label explanations, and a cautious integrated interpretation.

## Interpretation

This agent follows a MedRAX-inspired architecture: specialized ECG tools perform signal analysis, while Qwen acts as an integration and explanation layer.
Qwen is not used as the primary pathology detector. Instead, it translates rhythm features and PTB-XL classifier outputs into a structured, cautious explanation.

## Limitations

- The rhythm tool provides feature extraction, not full arrhythmia diagnosis.
- The pretrained PTB-XL classifier is an external expert model and should be evaluated separately.
- The system is a research prototype and not a clinical diagnostic tool.