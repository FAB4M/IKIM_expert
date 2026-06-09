"""Zentrale Konfiguration (Pfade + LLM).

BASE_DIR: env KIMI_CXR_BASE_DIR > Colab (_DEFAULT_COLAB) > lokal (_DEFAULT_LOCAL).
Alle übrigen Pfade leiten sich davon ab.
"""

from __future__ import annotations

import os
from pathlib import Path

# Windows/Anaconda: torch (MKL) und skimage/torchxrayvision bringen je eine OpenMP-
# Runtime (libiomp5md.dll) mit -> sonst "OMP Error #15" + Prozessabbruch. Dieser
# für Inferenz übliche Workaround muss gesetzt sein, BEVOR torch geladen wird.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --------------------------------------------------------------------------
# .env laden (LLM-Konfiguration), falls python-dotenv vorhanden ist.
# --------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    # .env liegt neben dieser Datei
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
except Exception:  # python-dotenv optional – Umgebungsvariablen funktionieren trotzdem
    pass


# --------------------------------------------------------------------------
# 1) Basisverzeichnis
# --------------------------------------------------------------------------
# Lokaler Windows-Pfad (Standard für deinen Rechner):
_DEFAULT_LOCAL = r"C:\Users\fabio\OneDrive - Data Moda Digital Engineering GmbH\Claude_Base\KIMI_CXR"

# Colab-Pfad (Google Drive). Bei Bedarf hier anpassen:
_DEFAULT_COLAB = "/content/drive/MyDrive/Colab/IKIM_CXR"


def _detect_base_dir() -> str:
    # (1) Explizite Umgebungsvariable hat Vorrang
    env_base = os.environ.get("KIMI_CXR_BASE_DIR")
    if env_base:
        return env_base

    # (2) Colab automatisch erkennen
    in_colab = False
    try:
        import google.colab  # noqa: F401

        in_colab = True
    except Exception:
        in_colab = os.path.isdir("/content")

    if in_colab:
        return _DEFAULT_COLAB

    # (3) Lokaler Fallback
    return _DEFAULT_LOCAL


BASE_DIR = Path(_detect_base_dir())

# --------------------------------------------------------------------------
# 2) Abgeleitete Verzeichnisse (alle relativ zu BASE_DIR)
# --------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"      # vom Notebook genutzt (Kompatibilität)
WEIGHTS_DIR = BASE_DIR / "weights"   # bevorzugter Ort für Checkpoints
TEMP_DIR = BASE_DIR / "tmp"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "outputs"
CONFIG_DIR = BASE_DIR  # config.py liegt im Projekt-Root

# Datensätze
VINBIGDATA_DIR = DATA_DIR / "VinBigData"
TRAIN_CSV = VINBIGDATA_DIR / "train.csv"
# TRAIN_DIR/TEST_DIR zeigen standardmäßig auf die VinBigData-Bildordner.
# Lokal existiert nur 'test' (108 DICOMs); in Colab zusätzlich 'train' (15.000).
TRAIN_DIR = VINBIGDATA_DIR / "train"
TEST_DIR = VINBIGDATA_DIR / "test"

# Unterordner Daten
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Unterordner Outputs
PRED_DIR = OUTPUT_DIR / "predictions"
VIZ_DIR = OUTPUT_DIR / "visualizations"
REPORT_DIR = OUTPUT_DIR / "reports"

# Checkpoints des eigenen Classifiers
CLASSIFIER_CKPT_DIR = WEIGHTS_DIR / "classifier"
BEST_CKPT = CLASSIFIER_CKPT_DIR / "best_model.pt"
LAST_CKPT = CLASSIFIER_CKPT_DIR / "last_model.pt"

# Externe Tool-Gewichte (z. B. TorchXRayVision)
EXTERNAL_TOOLS_DIR = WEIGHTS_DIR / "external_tools"

# CheXpert Plus – Radiologieberichte für das unsupervised LLM-Training
CHEXPERT_DIR = DATA_DIR / "chexpert"
CHEXPERT_CSV = CHEXPERT_DIR / "metadata" / "table_subsets" / "df_chexpert_plus_240401.csv"
CHEXPERT_PROCESSED = CHEXPERT_DIR / "processed"
REPORTS_CORPUS = CHEXPERT_PROCESSED / "reports_corpus.jsonl"

# LLM-Gewichte (LoRA-Adapter des stil-adaptierten Qwen)
LLM_DIR = WEIGHTS_DIR / "llm"
LLM_ADAPTER_DIR = LLM_DIR / "qwen_reports_lora"

# Liste aller Verzeichnisse, die existieren sollen
ALL_DIRS = [
    DATA_DIR, MODEL_DIR, WEIGHTS_DIR, TEMP_DIR, LOG_DIR, OUTPUT_DIR,
    RAW_DIR, PROCESSED_DIR, CACHE_DIR,
    PRED_DIR, VIZ_DIR, REPORT_DIR,
    CLASSIFIER_CKPT_DIR, EXTERNAL_TOOLS_DIR,
    CHEXPERT_PROCESSED, LLM_DIR,
]

# --------------------------------------------------------------------------
# 3) Modell- / Trainings-Defaults
# --------------------------------------------------------------------------
# Aktives Label-Set kommt aus medrax/training/labels.py (VinBigData-15).
# Hier nur als Komfort-Import; bei Importfehler leer lassen (config bleibt nutzbar).
try:
    from medrax.training.labels import VINBIGDATA_CLASSES as ACTIVE_CLASSES
    from medrax.training.labels import NUM_CLASSES
except Exception:  # z. B. wenn config.py isoliert importiert wird
    ACTIVE_CLASSES = []
    NUM_CLASSES = 15

BACKBONE = os.environ.get("KIMI_CXR_BACKBONE", "resnet18")  # resnet18 | efficientnet_b0 | (timm: convnext_tiny)
IMG_SIZE = int(os.environ.get("KIMI_CXR_IMG_SIZE", "224"))
NORM_MEAN = (0.5,)
NORM_STD = (0.5,)


def get_device(prefer: str | None = None) -> str:
    """Gibt 'cuda' zurück, wenn verfügbar, sonst 'cpu'. Robust ohne torch."""
    if prefer in ("cpu", "cuda"):
        if prefer == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    print("[config] CUDA angefordert, aber nicht verfügbar -> CPU.")
                    return "cpu"
            except Exception:
                return "cpu"
        return prefer
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = get_device()

# --------------------------------------------------------------------------
# 4) Lokalisierungs-Schwellen (Grad-CAM x Anatomie-Zonen)
#    Eine Zone wird nur dann behauptet, wenn alle drei Bedingungen erfüllt sind.
# --------------------------------------------------------------------------
LOC_MIN_PATHOLOGY_PROB = float(os.environ.get("KIMI_CXR_LOC_MIN_PROB", "0.40"))
LOC_MIN_ZONE_DOMINANCE = float(os.environ.get("KIMI_CXR_LOC_MIN_DOM", "0.22"))  # Anteil der Top-Zone
LOC_MIN_ZONE_MARGIN = float(os.environ.get("KIMI_CXR_LOC_MIN_MARGIN", "0.05"))  # Margin Top vs. 2.
LOC_MAX_HEATMAP_ENTROPY = float(os.environ.get("KIMI_CXR_LOC_MAX_ENTROPY", "0.92"))  # >dieser Wert = diffus

# --------------------------------------------------------------------------
# 5) LLM-Konfiguration (Ollama / OpenAI-kompatibel), aus .env / Umgebung
# --------------------------------------------------------------------------
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "qwen3:1.7b")
LLM_TIMEOUT = float(os.environ.get("KIMI_CXR_LLM_TIMEOUT", "60"))

# Basis-LLM für das unsupervised Continued-Pre-Training (Colab/serve_qwen)
BASE_LLM = os.environ.get("KIMI_CXR_BASE_LLM", "Qwen/Qwen2.5-1.5B-Instruct")

# --------------------------------------------------------------------------
# 6) Helfer
# --------------------------------------------------------------------------
def ensure_dirs(verbose: bool = False) -> None:
    """Legt alle Projektverzeichnisse an (idempotent)."""
    for d in ALL_DIRS:
        try:
            d.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"[config] ok: {d}")
        except Exception as e:
            print(f"[config] WARN: Verzeichnis konnte nicht angelegt werden: {d} ({e})")


def check_base_dir() -> None:
    """Verständliche Fehler, wenn die Projektbasis/Daten fehlen."""
    if not BASE_DIR.exists():
        raise FileNotFoundError(
            f"BASE_DIR existiert nicht: {BASE_DIR}\n"
            "-> Passe config.py (_DEFAULT_LOCAL/_DEFAULT_COLAB) oder die Umgebungs-"
            "variable KIMI_CXR_BASE_DIR an."
        )
    if not DATA_DIR.exists():
        print(
            f"[config] Hinweis: DATA_DIR fehlt ({DATA_DIR}). "
            "Lege es an oder rufe config.ensure_dirs() auf."
        )


def summary() -> str:
    lines = [
        "KIMI_CXR Konfiguration",
        f"  BASE_DIR    : {BASE_DIR}",
        f"  DATA_DIR    : {DATA_DIR}",
        f"  WEIGHTS_DIR : {WEIGHTS_DIR}",
        f"  OUTPUT_DIR  : {OUTPUT_DIR}",
        f"  DEVICE      : {DEVICE}",
        f"  BACKBONE    : {BACKBONE}  IMG_SIZE={IMG_SIZE}",
        f"  LLM         : {OPENAI_MODEL} @ {OPENAI_BASE_URL}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ensure_dirs(verbose=True)
    print(summary())
