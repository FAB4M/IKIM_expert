# One-page summary

## ECG-Qwen Tool-Agent

This project developed a MedRAX-inspired ECG tool-agent using the Qwen model family. The selected modality was 12-lead electrocardiography. The system was designed so that Qwen does not act as the primary ECG pathology detector. Instead, specialized external ECG tools analyze the raw ECG signal, and Qwen acts as an agentic explanation layer that integrates and explains the tool outputs in a structured and cautious way.

## Research and data acquisition

The research focused on publicly available ECG datasets, pretrained ECG models, and tools that can be integrated into an agentic system. PTB-XL was selected as the primary dataset because it is openly available, well documented, contains 10-second 12-lead ECG raw signals, and provides diagnostic superclass labels. The prototype used PTB-XL records100, sampled at 100 Hz. The relevant diagnostic superclasses were NORM, MI, STTC, CD, and HYP, corresponding to normal ECG, myocardial infarction, ST/T changes, conduction disturbance, and hypertrophy.

Existing tools and models were also reviewed. NeuroKit2 was selected as an external rhythm feature extraction tool. It was used to estimate heart rate, detect R-peaks, and assess RR regularity. A publicly available pretrained PTB-XL classifier from Hugging Face was selected as the main signal-based classification expert. This model processes raw 12-lead ECG signals and outputs probabilities for the five PTB-XL diagnostic superclasses.

## Agent architecture

The final architecture consists of three components. First, NeuroKit2 extracts rhythm-related features from the raw ECG signal. Second, the pretrained PTB-XL classifier predicts probabilities for NORM, MI, STTC, CD, and HYP. Third, Qwen/Qwen2.5-0.5B-Instruct integrates the structured outputs from both tools and generates a cautious ECG agent summary. The summary contains a rhythm assessment, diagnostic superclass classification, integrated interpretation, and a limitation statement.

## Qwen training

Qwen/Qwen2.5-0.5B-Instruct was fine-tuned with LoRA as an ECG tool-agent. The training data consisted of 1000 structured ECG tool-output examples generated from the NeuroKit2 rhythm module and the pretrained PTB-XL classifier. Qwen was trained in a supervised manner to translate these tool outputs into structured ECG agent summaries. It was not trained to directly classify raw ECG signals or ECG images.

Training setup:

- Base model: Qwen/Qwen2.5-0.5B-Instruct
- Fine-tuning method: LoRA
- Training examples: 1000
- Train/evaluation split: 90/10
- Epochs: 2
- Validation loss after epoch 2: 0.3941

## Results

The pretrained PTB-XL classifier achieved strong binary normal/pathological performance with an accuracy of 0.921. The normal class reached an F1-score of 0.922, and the pathological class reached an F1-score of 0.920. In the confusion matrix, 469 of 500 normal ECGs and 452 of 500 pathological ECGs were correctly classified.

For balanced 5-class top-class classification across NORM, MI, STTC, CD, and HYP, the pretrained classifier achieved an accuracy of 0.682 and a macro F1-score of 0.664. This indicates that the model is strong for binary pathology detection and moderately useful for diagnostic superclass classification.

Qualitative evaluation of the fine-tuned Qwen agent showed that it could reproduce rhythm features, classifier probabilities, predicted PTB-XL labels, label meanings, and cautious limitations in a structured format. This supports the intended role of Qwen as an explanation and integration layer rather than as the primary diagnostic model.

## Limitations

This system is a research prototype and not a clinical diagnostic tool. NeuroKit2 provides rhythm feature extraction only and does not perform full arrhythmia diagnosis. The pretrained PTB-XL classifier was evaluated on PTB-XL-derived data and should not be considered externally clinically validated. Future work could integrate additional ECG expert models, dedicated arrhythmia classifiers, multimodal ECG image datasets, or ECG foundation models such as DeepECG-SSL or ECG-Mamba.
