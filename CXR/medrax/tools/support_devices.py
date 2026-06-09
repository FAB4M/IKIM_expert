"""SupportDeviceClassifierTool – CheXpert-DenseNet121 (itsomk/chexpert-densenet121).
Gibt nur 'Support Devices' aus (Ja/Nein + Confidence).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import BaseCXRTool, BaseModel, Field, failed, make_metadata

# Offizielle Label-Reihenfolge des Modells (14 CheXpert-Beobachtungen)
_CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]


def _build_chexpert_densenet():
    """Repliziert die Modellstruktur des HF-Repos (für passende state_dict-Keys)."""
    import torch.nn as nn
    from torchvision import models

    class DenseNet121_CheXpert(nn.Module):
        def __init__(self, num_labels=14):
            super().__init__()
            self.densenet = models.densenet121(weights=None)
            in_f = self.densenet.classifier.in_features
            self.densenet.classifier = nn.Linear(in_f, num_labels)

        def forward(self, x):
            return self.densenet(x)

    return DenseNet121_CheXpert(num_labels=len(_CHEXPERT_LABELS))


class SupportDeviceInput(BaseModel):
    image_path: str = Field(..., description="Pfad zu einem CXR-Bild (PNG/JPG/DICOM).")


class SupportDeviceClassifierTool(BaseCXRTool):
    name = "support_device_classifier"
    description = (
        "Schmaler Zusatz-Detektor: ist auf dem Röntgen-Thorax wahrscheinlich ein "
        "Support-Device (z. B. Tubus, ZVK, Schrittmacher, Drainage) sichtbar? Basiert auf "
        "einem CheXpert-DenseNet121 und gibt NUR das Label 'Support Devices' aus. "
        "Eingabe: image_path. Ausgabe: output['support_device'] (bool), output['present'] "
        "('Ja'/'Nein'), output['confidence'] (0..1). KEINE Device-Typ-Klassifikation, "
        "keine weiteren Pathologien."
    )
    args_schema = SupportDeviceInput

    _REPO = "itsomk/chexpert-densenet121"
    _FILE = "pytorch_model.safetensors"

    def __init__(self, device: Optional[str] = None, threshold: float = 0.5):
        import config

        self.device = device or config.get_device()
        self.threshold = threshold
        self._model = None
        self._idx = _CHEXPERT_LABELS.index("Support Devices")

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            from medrax.utils.xrv_utils import safe_console

            with safe_console():
                path = hf_hub_download(repo_id=self._REPO, filename=self._FILE)
            net = _build_chexpert_densenet()
            net.load_state_dict(load_file(path), strict=False)
            self._model = net.eval().to(self.device)
            print("[support-device] CheXpert-DenseNet121 geladen.")
            return True
        except Exception as e:
            print("[support-device] Modell nicht verfügbar:", e)
            self._model = None
            return False

    def _transform(self):
        from torchvision import transforms

        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def analyze(self, image_path: str) -> dict:
        import torch

        from medrax.utils.image_utils import load_image_any

        img = load_image_any(image_path)  # RGB
        x = self._transform()(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self._model(x))[0].cpu().numpy()
        prob = float(probs[self._idx])
        present = prob >= self.threshold
        return {
            "support_device": bool(present),
            "present": "Ja" if present else "Nein",
            "confidence": round(prob, 4),
            "threshold": self.threshold,
        }

    def _run(self, image_path: str):
        if not Path(image_path).exists():
            return failed(f"Bild nicht gefunden: {image_path}", image_path=image_path)
        if not self._ensure_model():
            return failed(
                "CheXpert-DenseNet (itsomk/chexpert-densenet121) nicht verfügbar. Installiere:  "
                "pip install huggingface_hub safetensors",
                image_path=image_path, tool="support_device_classifier",
            )
        try:
            res = self.analyze(image_path)
        except Exception as e:
            return failed(f"Support-Device-Klassifikation fehlgeschlagen: {e}", image_path=image_path)
        return res, make_metadata("completed", image_path=image_path, model=self._REPO)
