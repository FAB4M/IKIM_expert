Generated using AI; Bitte mit Vorsicht genießen; Soll die Struktur ungefähr darstellen


# KIMI_CXR — Chest-X-Ray-Agent

Lokal und in Colab lauffähig: ein LLM-Agent (Qwen) orchestriert CXR-Tools. Der Classifier
ist selbst gebaut und auf VinBigData-15 trainiert; Anatomie, View und Support-Device kommen
aus eingebundenen Modellen.


Eingebundene Modelle zusätzlich:
- TorchXRayVision (Anatomie-Segmenter): https://github.com/mlmed/torchxrayvision
- ianpan/chest-x-ray-basic (View + Lunge/Herz): https://huggingface.co/ianpan/chest-x-ray-basic
- itsomk/chexpert-densenet121 (Support-Device): https://huggingface.co/itsomk/chexpert-densenet121

---

## Pipeline

Bild (PNG/JPG/DICOM) → Classifier → 15 Pathologie-Wahrscheinlichkeiten + Grad-CAM →
Anatomie-Segmenter (grobe Lokalisation, z. B. „rechtes Oberfeld") → View (AP/PA/lateral) →
Support-Device-Check → Qwen formuliert eine Fließtext-Befundung.

**Core-Tools:**
- `MyChestXRayClassifierTool` – VinBigData-15 Multi-Label-Classifier + Grad-CAM
- `DicomProcessorTool` – DICOM → PNG (Rescale/Window/CLAHE, JPEG2000)
- `ImageVisualizerTool` – Vorschau/Overlays
- `XRVAnatomySegmenterTool` – rechte/linke Lunge, obere/mittlere/untere Zone, Herz, Mediastinum
- `ChestXrayBasicTool` – View (AP/PA/lateral), Lunge re/li, Herz, CTR
- `SupportDeviceClassifierTool` – Support Device Ja/Nein + Confidence

Die Anatomie-/Orientierungs-Modelle laden ihre Gewichte lazy beim ersten Lauf. Fehlen die Pakete,
läuft die Kernpipeline weiter; die Tools melden sich ab.

Deaktivierte Stubs (nicht ausgebaut): `GroundingTool`, `LlavaMedVQATool`, `ReportGenerationTool`,
`ImageGenerationTool`.

Die 18 TorchXRayVision-Pathologien liegen in `medrax/training/labels.py` nur als Referenz-Mapping.

---

## Projektstruktur

```
KIMI_CXR/
├── config.py                # Pfade + Colab-Switch + LLM-Config
├── .env.example / .env      # Ollama-Defaults
├── requirements.txt, pyproject.toml
├── medrax/
│   ├── agent/               # qwen_client, prompts, initialize_agent
│   ├── tools/               # base, my_classifier_tool, dicom, visualization,
│   │                        # anatomy_segmentation, view_anatomy, support_devices,
│   │                        # localization, Stubs, Registry
│   ├── training/            # labels, dataset, model, gradcam, losses, metrics,
│   │                        # train_classifier, evaluate_classifier
│   ├── llm/                 # prepare_reports, train_lm_unsupervised, serve_qwen
│   └── utils/               # dicom_utils, image_utils, paths, logging, seed
├── scripts/                 # test_* + convert_dicom_to_png_cache, build_vinbigdata_labels
├── data/                    # VinBigData/ + chexpert/
├── weights/                 # classifier/ + llm/
├── outputs/  logs/  notebooks/
```

---

## Setup (lokal, Windows)

> Python liegt in Anaconda; Aufruf über `py` bzw. `C:\Users\fabio\anaconda3\python.exe`
> (das blanke `python` ist ein Windows-Store-Stub).

```powershell
py -m pip install -r requirements.txt
py -m pip install -e .        # optional, editierbar
py config.py                  # legt Ordner an, zeigt Config
```

Core: torch, torchvision, pydicom, opencv, scikit-learn, scikit-image, pandas, numpy, pillow,
matplotlib, tqdm, python-dotenv, requests.
Anatomie-/Orientierungs-Tools: `torchxrayvision`, `transformers<5`, `timm`, `einops`,
`huggingface_hub`, `safetensors`.

> `config.py` setzt `KMP_DUPLICATE_LIB_OK=TRUE` (OpenMP-Konflikt torch ↔ skimage/torchxrayvision).

---

## Qwen (Ollama)

`.env`:
```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen3:1.7b
```
```powershell
ollama serve
ollama pull qwen3:1.7b
py scripts/test_agent_qwen.py
```
Ohne laufenden Server brechen die Tests nicht ab (strukturierter Fallback).
Alternativ vLLM/LM Studio: `OPENAI_BASE_URL=http://localhost:8000/v1`.

---

## Tests

```powershell
py scripts/test_classifier_10_images.py   # Classifier-Smoke (DUMMY-Labels)
py scripts/test_agent_qwen.py             # Qwen/Ollama erreichbar?
py scripts/test_anatomy_tools.py          # XRV / ianpan / Support-Device (lädt Gewichte)
py scripts/test_full_pipeline.py          # komplette Pipeline
```

> Die 108 lokalen Test-DICOMs haben keine Labels → lokales „Training" nutzt DUMMY-Labels.
> Echtes Training läuft in Colab mit den 15.000 gelabelten Bildern.

DICOMs vorab cachen:
```powershell
py scripts/convert_dicom_to_png_cache.py --src data/VinBigData/test
```

---

## Classifier trainieren

```powershell
py -m medrax.training.train_classifier --max-samples 10 --epochs 1 --batch-size 2   # Smoke
py -m medrax.training.train_classifier --epochs 10 --batch-size 16 --backbone resnet18
py -m medrax.training.train_classifier --resume weights/classifier/last_model.pt --epochs 5
py -m medrax.training.evaluate_classifier --checkpoint weights/classifier/best_model.pt
py scripts/build_vinbigdata_labels.py     # Detection-CSV -> Multi-Label-CSV
```

---

## config.py

`BASE_DIR` wird automatisch bestimmt: env `KIMI_CXR_BASE_DIR` > Colab (`_DEFAULT_COLAB`) >
lokal (`_DEFAULT_LOCAL`). Für Colab nur `_DEFAULT_COLAB` setzen (siehe `notebooks/colab_notes.md`).
Weitere Schalter: `BACKBONE`, `IMG_SIZE`, `LOC_*`, LLM-Variablen.

---

Nur Forschung & Entwicklung. Nicht klinisch validiert, keine medizinische Diagnose.
