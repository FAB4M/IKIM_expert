# Research Notes

## Search Strategy

The modality selected for this project was electrocardiography (ECG).
The search focused on open ECG repositories, publicly available benchmark datasets, ECG classification papers, and existing ECG-related foundation model or LLM-based approaches.

Search terms included:

- open ECG dataset
- PTB-XL ECG dataset
- ECG classification benchmark
- ECG question answering dataset
- ECG foundation model
- ECG multimodal large language model
- Qwen ECG fine-tuning

Repositories and sources considered:

- PhysioNet
- Papers with Code
- Hugging Face
- arXiv
- PubMed / Google Scholar

## Main Dataset: PTB-XL

PTB-XL was selected as the main dataset because it is a large, publicly available 12-lead ECG dataset with clinical diagnostic annotations.
It provides ECG signal files, metadata, and diagnostic SCP codes that can be mapped to diagnostic superclasses.

Relevant PTB-XL diagnostic superclasses used in this project:

- NORM: normal ECG
- MI: myocardial infarction
- STTC: ST/T changes
- CD: conduction disturbance
- HYP: hypertrophy

## Data Subset Used

For this proof of concept, a balanced subset of 1,000 ECGs was constructed:

- 200 NORM
- 200 MI
- 200 STTC
- 200 CD
- 200 HYP

The corresponding low-resolution 100 Hz ECG signal files from PTB-XL records100 were downloaded.
Each ECG was rendered as a 12-lead PNG image.

## Other Potential ECG Datasets

Other ECG datasets that could be considered in future work include:

- MIT-BIH Arrhythmia Database
- Chapman-Shaoxing ECG dataset
- CPSC ECG datasets
- ECG-QA or other ECG question-answering resources

These datasets could be used to extend the model beyond PTB-XL labels and evaluate generalization.

## Existing Models and Benchmarks

Existing ECG-related model families and benchmarks include classical deep learning ECG classifiers, transformer-based ECG models, and recent ECG foundation model or ECG question-answering approaches.
For this proof of concept, the focus was not on outperforming existing ECG classifiers, but on demonstrating how a small Qwen model can be adapted to ECG-specific label interpretation.

## Additional Data Acquisition Strategies

Possible additional strategies include:

- Extending from 1,000 ECGs to the full PTB-XL dataset.
- Adding ECG report text if available.
- Creating richer question-answer pairs from PTB-XL diagnostic labels and metadata.
- Combining PTB-XL with MIT-BIH or Chapman-Shaoxing for external validation.
- Using expert-curated ECG interpretation templates.
- Generating multimodal ECG image-question-answer pairs for Qwen-VL fine-tuning.

## Rationale for Choosing ECG

ECG was chosen because it is clinically relevant, comparatively lightweight, standardized, and feasible for a short proof-of-concept project.
Unlike CT or MRI, ECG data are smaller and easier to process on limited hardware.
In addition, ECGs can be represented both as time-series data and as image-like 12-lead plots, making them suitable for both text-only and multimodal model development.