# KIMI_CXR in Google Colab

Kurzanleitung, um das Projekt in Colab mit GPU zu betreiben. Lokal heißt der
Projektordner `KIMI_CXR`, in Colab wird `IKIM_CXR` auf Google Drive verwendet.

---

## 1) BASE_DIR setzen

Es gibt zwei Wege – einer reicht:

**A) `config.py` anpassen** (eine Zeile):
```python
_DEFAULT_COLAB = "/content/drive/MyDrive/Colab/IKIM_CXR"
```
`config.py` erkennt Colab automatisch (Existenz von `/content`) und nutzt dann diesen Pfad.

**B) Umgebungsvariable** (überschreibt alles, ohne Code-Änderung):
```python
import os
os.environ["KIMI_CXR_BASE_DIR"] = "/content/drive/MyDrive/Colab/IKIM_CXR"
```

---

## 2) Google Drive mounten

```python
from google.colab import drive
drive.mount("/content/drive")
```

---

## 3) Erwartete Ordnerstruktur auf Drive

```
/content/drive/MyDrive/Colab/IKIM_CXR/
├── config.py, medrax/, scripts/, requirements.txt
└── data/VinBigData/
    ├── train.csv
    ├── train/      # 15.000 DICOMs (für echtes Training)
    └── test/       # optional
```
Projektcode nach Drive kopieren (oder `git clone`), dann ins Projekt wechseln:
```python
%cd /content/drive/MyDrive/Colab/IKIM_CXR
```

---

## 4) Abhängigkeiten installieren

```python
!pip install -r requirements.txt
# Für komprimierte DICOMs (JPEG2000) in Colab zusätzlich empfehlenswert:
!pip install pylibjpeg pylibjpeg-openjpeg pylibjpeg-libjpeg
# Anatomie-/Orientierungs-Tools (XRV-Segmenter, ianpan-View, Support-Device):
!pip install torchxrayvision "transformers<5" timm einops huggingface_hub safetensors "albumentations<1.4"
# (Gewichte werden beim ersten Lauf von GitHub/HuggingFace geladen.
#  albumentations wird vom ianpan-Modell benötigt.)
```

Konfiguration prüfen:
```python
import config; config.ensure_dirs(); print(config.summary())   # DEVICE sollte 'cuda' sein
```

---

## 5) 10-Bilder-Test (schnell)

```python
!python scripts/test_classifier_10_images.py
```
Liegen lokal/Colab keine gelabelten Bilder, läuft automatisch der DUMMY-Modus
(klar markiert, rein technisch).

---

## 6) Echtes Training mit GPU

Mit vorhandenem `train.csv` + `train/`-Ordner (15.000 Bilder):
```python
!python -m medrax.training.train_classifier --epochs 10 --batch-size 16 \
    --backbone resnet18 --img-size 224
```
Tipps:
- `--backbone efficientnet_b0` oder (mit `pip install timm`) `--backbone convnext_tiny`.
- `--amp` aktiviert Mixed Precision (nur auf GPU sinnvoll).
- DICOMs einmalig cachen: `!python scripts/convert_dicom_to_png_cache.py --src data/VinBigData/train`
- Checkpoints landen unter `weights/classifier/` auf Drive (bleiben erhalten).
- Fortsetzen: `--resume weights/classifier/last_model.pt`.

Evaluieren:
```python
!python -m medrax.training.evaluate_classifier --checkpoint weights/classifier/best_model.pt
```

---

## 7) LLM in Colab

Ollama ist in Colab unüblich. Optionen:
- externen OpenAI-kompatiblen Endpoint in `.env`/Umgebung setzen
  (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`), oder
- ohne LLM arbeiten: die Pipeline liefert dann den strukturierten Fallback-Text.

---

> Hinweis: Forschungs-/Entwicklungsprojekt, nicht klinisch validiert.
