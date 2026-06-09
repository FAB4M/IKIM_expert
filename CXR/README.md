# IKIM_CXR — Chest-X-Ray-Agent

Ich hoffe der Agent gefällt euch. Idee war ein Agent auf aBsis von Qwen, der von Subprogrammen wie einen selbst trainierten Classifier, outputs bekommt. Qwen selbst ist unsupervised auf 20k Befundberichten trainiert. Der Classifier sup. auf 15k gelabelten CXR. In einem weiteren Schritt könnte man den Agent in Loop laufen lassen und die Subprogramm je nach Aufgabenstellung (z.B. Wo ist das Inflitrat? -> localizer), selbst aussuchen lassen. 
AI generated summary, pls take with a grain of salt. 
---

## Idee

Ein Thoraxbild wird durch mehrere spezialisierte Tools verarbeitet.

Die Pipeline ist:

```text
Bild (PNG/JPG/DICOM)
→ Pathologie-Classifier
→ Grad-CAM
→ Anatomie-Segmentierung
→ View-Erkennung
→ Support-Device-Check
→ Qwen erstellt Befundtext
```

Das LLM trifft dabei keine Diagnose allein.
Es fasst die Ergebnisse der Tools zusammen und formuliert daraus einen radiologisch orientierten Text.

---

## Eingebundene Modelle

Zusätzlich zum eigenen Classifier werden folgende Modelle genutzt:

* TorchXRayVision
  Anatomie-Segmentierung
  https://github.com/mlmed/torchxrayvision

* ianpan/chest-x-ray-basic
  View-Erkennung und Basis-Anatomie
  https://huggingface.co/ianpan/chest-x-ray-basic

* itsomk/chexpert-densenet121
  Support-Device-Klassifikation
  https://huggingface.co/itsomk/chexpert-densenet121

Die Modelle laden ihre Gewichte erst beim ersten Aufruf.
Wenn optionale Pakete fehlen, läuft die Kernpipeline weiter. Die jeweiligen Tools melden sich dann ab.

---

## Core-Tools

* `MyChestXRayClassifierTool`
  Eigener Multi-Label-Classifier für VinBigData-15 mit Grad-CAM.

* `DicomProcessorTool`
  Wandelt DICOMs in PNGs um. Unterstützt Rescale, Windowing, CLAHE und JPEG2000.

* `ImageVisualizerTool`
  Erstellt Vorschauen und Overlays.

* `XRVAnatomySegmenterTool`
  Segmentiert grob Lunge, Herz und Mediastinum. Zusätzlich werden obere, mittlere und untere Lungenzonen abgeleitet.

* `ChestXrayBasicTool`
  Erkennt AP, PA oder lateral. Liefert außerdem grobe Informationen zu rechter/linker Lunge, Herz und CTR.

* `SupportDeviceClassifierTool`
  Prüft, ob Support-Devices sichtbar sind, inklusive Confidence.

---

## Deaktivierte Stubs

Einige Tools sind vorbereitet, aber aktuell nicht ausgebaut:

* `GroundingTool`
* `LlavaMedVQATool`
* `ReportGenerationTool`
* `ImageGenerationTool`

Die 18 TorchXRayVision-Pathologien liegen nur als Referenz-Mapping in:

```text
medrax/training/labels.py
```

---

## Projektstruktur

```text
IKIM_CXR/
├── config.py
├── .env.example / .env
├── requirements.txt
├── pyproject.toml
├── medrax/
│   ├── agent/
│   ├── tools/
│   ├── training/
│   ├── llm/
│   └── utils/
├── scripts/
├── data/
├── weights/
├── outputs/
├── logs/
└── notebooks/
```

Wichtige Bereiche:

* `medrax/agent/`
  Qwen-Client, Prompts und Agent-Initialisierung.

* `medrax/tools/`
  Alle Bild-, Klassifikations-, Segmentierungs- und Hilfstools.

* `medrax/training/`
  Dataset, Modell, Losses, Metriken, Grad-CAM, Training und Evaluation.

* `medrax/llm/`
  Vorbereitung von Berichtsdaten und optionales Qwen-Weitertraining.

* `scripts/`
  Tests, Hilfsskripte und DICOM-Caching.

---

## Lokales Setup unter Windows

Python läuft über Anaconda.
Der Aufruf erfolgt über `py` oder direkt über den Anaconda-Python-Pfad.

```powershell
py -m pip install -r requirements.txt
py -m pip install -e .
py config.py
```

`config.py` legt benötigte Ordner an und zeigt die aktuelle Konfiguration.

Wichtige Pakete:

```text
torch
torchvision
pydicom
opencv
scikit-learn
scikit-image
pandas
numpy
pillow
matplotlib
tqdm
python-dotenv
requests
```

Für die optionalen Anatomie- und Orientierungs-Tools zusätzlich:

```text
torchxrayvision
transformers<5
timm
einops
huggingface_hub
safetensors
```

Unter Windows setzt `config.py` außerdem:

```text
KMP_DUPLICATE_LIB_OK=TRUE
```

Das verhindert typische OpenMP-Konflikte zwischen Torch, scikit-image und TorchXRayVision.

---

## Qwen mit Ollama

Die lokale LLM-Anbindung läuft über eine OpenAI-kompatible Schnittstelle.

Beispiel für `.env`:

```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen3:1.7b
```

Start mit Ollama:

```powershell
ollama serve
ollama pull qwen3:1.7b
py scripts/test_agent_qwen.py
```

Wenn Ollama nicht läuft, bricht die Pipeline nicht hart ab.
Es gibt einen strukturierten Fallback.

Alternativ können auch vLLM oder LM Studio genutzt werden:

```env
OPENAI_BASE_URL=http://localhost:8000/v1
```

---

## Tests

```powershell
py scripts/test_classifier_10_images.py
py scripts/test_agent_qwen.py
py scripts/test_anatomy_tools.py
py scripts/test_full_pipeline.py
```

Die lokalen 108 Test-DICOMs besitzen keine echten Labels.
Lokales Training nutzt deshalb nur Dummy-Labels.

Das echte Training ist für Colab mit den gelabelten VinBigData-Daten vorgesehen.

DICOMs können vorher gecacht werden:

```powershell
py scripts/convert_dicom_to_png_cache.py --src data/VinBigData/test
```

---

## Classifier-Training

Smoke-Test:

```powershell
py -m medrax.training.train_classifier --max-samples 10 --epochs 1 --batch-size 2
```

Normales Training:

```powershell
py -m medrax.training.train_classifier --epochs 10 --batch-size 16 --backbone resnet18
```

Training fortsetzen:

```powershell
py -m medrax.training.train_classifier --resume weights/classifier/last_model.pt --epochs 5
```

Evaluation:

```powershell
py -m medrax.training.evaluate_classifier --checkpoint weights/classifier/best_model.pt
```

Labels aus VinBigData-Detection-CSV bauen:

```powershell
py scripts/build_vinbigdata_labels.py
```

---

## Konfiguration

`config.py` bestimmt automatisch das Basisverzeichnis.

Priorität:

```text
KIMI_CXR_BASE_DIR
→ Colab-Pfad
→ lokaler Standardpfad
```

Weitere wichtige Schalter:

* `BACKBONE`
* `IMG_SIZE`
* `LOC_*`
* LLM-Variablen
* Colab-/Local-Switches

Für Colab muss nur der passende Colab-Pfad gesetzt werden.
Details stehen in:

```text
notebooks/colab_notes.md
```

---

## Aktueller Stand

Funktional vorhanden:

* DICOM-Verarbeitung
* eigener CXR-Classifier
* Grad-CAM
* optionale Anatomie-Segmentierung
* optionale View-Erkennung
* optionaler Support-Device-Check
* Qwen-basierte Befundformulierung
* lokale Tests
* Colab-Training

Noch offen:

* klinische Validierung
* robustere Evaluation auf externen Daten
* bessere Pathologie-Lokalisation
* Ausbau der Berichtsgenerierung
* systematische Fehleranalyse

---

## Hinweis

Dieses Projekt ist ein technischer Prototyp.
Es ist nicht für den klinischen Einsatz bestimmt.
Die Ausgaben dürfen nicht als medizinische Diagnose verwendet werden.
