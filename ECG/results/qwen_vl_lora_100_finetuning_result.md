# Qwen-VL ECG LoRA Fine-Tuning Test

A small multimodal LoRA fine-tuning experiment was performed using Qwen2.5-VL-3B-Instruct on 100 ECG image-question-answer examples.

## Test Example

- ECG ID: 11147
- True answer: This ECG is most consistent with a normal ECG.
- Image: /content/drive/MyDrive/ecg-qwen-modality-expert/results/ecg_images_1000/ecg_11147.png

## Fine-tuned Qwen-VL Output

The provided ECG waveform is from the PTB-XL dataset. The model generated a general visual ECG description mentioning baseline, P waves, and QRS complexes, but did not reproduce the expected concise normal/abnormal answer format.

## Interpretation

The experiment confirms that multimodal Qwen-VL fine-tuning is technically feasible in this setup. However, the qualitative result shows that 100 examples and one epoch are not sufficient to reliably align the model with the desired ECG classification answer format.

## Next Improvement

The next step should be to improve the supervised multimodal fine-tuning setup by using more examples, more specific prompts, and answer-only loss masking so that the model learns primarily from the assistant response.