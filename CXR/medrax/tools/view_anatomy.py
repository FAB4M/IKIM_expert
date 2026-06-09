"""ChestXrayBasicTool – ianpan/chest-x-ray-basic: View (AP/PA/lateral), Lunge re/li, Herz, CTR."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseCXRTool, BaseModel, Field, failed, make_metadata

_VIEW_MAP = {0: "AP", 1: "PA", 2: "lateral"}


def _calculate_ctr(mask: np.ndarray) -> Optional[float]:
    """Kardiothorakaler Quotient aus der Maske (1/2=Lunge, 3=Herz)."""
    try:
        lungs = np.isin(mask, [1, 2])
        heart = mask == 3
        if lungs.sum() == 0 or heart.sum() == 0:
            return None
        xs_l = np.where(lungs.any(axis=0))[0]
        xs_h = np.where(heart.any(axis=0))[0]
        lung_range = xs_l.max() - xs_l.min()
        heart_range = xs_h.max() - xs_h.min()
        if lung_range <= 0:
            return None
        return float(round(heart_range / lung_range, 3))
    except Exception:
        return None


class ViewAnatomyInput(BaseModel):
    image_path: str = Field(..., description="Pfad zu einem CXR-Bild (PNG/JPG/DICOM).")


class ChestXrayBasicTool(BaseCXRTool):
    name = "chest_xray_basic"
    description = (
        "Leichtgewichtiges Orientierungs-Tool (ianpan/chest-x-ray-basic). Bestimmt die "
        "Aufnahme-Projektion (AP/PA/lateral) und segmentiert grob rechte Lunge, linke Lunge "
        "und Herz (inkl. kardiothorakalem Quotienten CTR). NUTZEN zur Orientierung, "
        "besonders rechts/links und View. Eingabe: image_path. Ausgabe: output['view'], "
        "output['right_lung']/['left_lung']/['heart'] (bool), output['cardiothoracic_ratio']. "
        "Kein Pathologie-Classifier."
    )
    args_schema = ViewAnatomyInput

    _MODEL_ID = "ianpan/chest-x-ray-basic"

    def __init__(self, device: Optional[str] = None):
        import config

        self.device = device or config.get_device()
        self._model = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(self._MODEL_ID, trust_remote_code=True)
            self._model = self._model.eval().to(self.device)
            print("[chest-x-ray-basic] Modell geladen.")
            return True
        except Exception as e:
            print("[chest-x-ray-basic] nicht verfügbar:", e)
            self._model = None
            return False

    def _load_input(self, image_path: str):
        """Lädt ein Graustufenbild als numpy (für model.preprocess)."""
        from medrax.utils.dicom_utils import is_dicom, read_dicom_array

        if is_dicom(image_path):
            # bevorzugt die modelleigene DICOM-Routine, sonst unser Reader
            try:
                return self._model.load_image_from_dicom(image_path)
            except Exception:
                return read_dicom_array(image_path)
        import cv2

        img = cv2.imread(str(image_path), 0)
        if img is None:
            from PIL import Image

            img = np.asarray(Image.open(image_path).convert("L"))
        return img

    def analyze(self, image_path: str) -> dict:
        import torch

        img = self._load_input(image_path)
        x = self._model.preprocess(img)
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(self.device)
        with torch.inference_mode():
            out = self._model(x)

        # View
        view_idx = int(torch.argmax(out["view"], dim=1).item())
        view = _VIEW_MAP.get(view_idx, str(view_idx))

        # Maske -> Regionen + CTR
        mask = torch.argmax(out["mask"], dim=1)[0].cpu().numpy()
        ctr = _calculate_ctr(mask)
        result = {
            "view": view,
            "right_lung": bool((mask == 1).sum() > 0),
            "left_lung": bool((mask == 2).sum() > 0),
            "heart": bool((mask == 3).sum() > 0),
            "cardiothoracic_ratio": ctr,
        }
        # Zusatzinfo (optional vorhanden)
        try:
            result["age_est"] = round(float(out["age"].item()), 1)
        except Exception:
            pass
        try:
            result["female_prob"] = round(float(out["female"].item()), 3)
        except Exception:
            pass
        return result

    def _run(self, image_path: str):
        if not Path(image_path).exists():
            return failed(f"Bild nicht gefunden: {image_path}", image_path=image_path)
        if not self._ensure_model():
            return failed(
                "Modell 'ianpan/chest-x-ray-basic' nicht verfügbar. Installiere:  "
                "pip install transformers timm einops huggingface_hub safetensors",
                image_path=image_path, tool="chest_xray_basic",
            )
        try:
            res = self.analyze(image_path)
        except Exception as e:
            return failed(f"chest-x-ray-basic fehlgeschlagen: {e}", image_path=image_path)
        return res, make_metadata("completed", image_path=image_path, model=self._MODEL_ID)
