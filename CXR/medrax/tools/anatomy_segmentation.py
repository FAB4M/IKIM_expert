"""XRVAnatomySegmenterTool – TorchXRayVision ChestX-Det PSPNet (nur Anatomie-Segmenter).
Liefert Lunge/Zonen/Herz/Mediastinum; mit Grad-CAM zu grober Lokalisation kombiniert.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .base import BaseCXRTool, BaseModel, Field, failed, make_metadata


class AnatomyInput(BaseModel):
    image_path: str = Field(..., description="Pfad zu einem CXR-Bild (PNG/JPG/DICOM).")


class XRVAnatomySegmenterTool(BaseCXRTool):
    name = "xrv_anatomy_segmenter"
    description = (
        "Anatomischer Segmenter (TorchXRayVision ChestX-Det). Liefert vereinfachte "
        "anatomische Regionen: rechte/linke Lunge, obere/mittlere/untere Lungenzone, "
        "Herzregion, Mediastinum. NUTZEN nach dem Classifier, um die Grad-CAM-Aktivierung "
        "grob zu verorten (z. B. 'rechtes Oberfeld' oder 'Herzregion'). Eingabe: image_path "
        "(+ intern Heatmap). Ausgabe: output['regions'] (welche Regionen erkannt) und "
        "output['localization'] mit 'localized' (bool) + 'zone'. KEIN Pathologie-Classifier, "
        "KEINE Detektion; Lokalisation ist grob und nur bei belastbarer Aktivierung gesetzt."
    )
    args_schema = AnatomyInput

    # Ziel-Strukturen (XRV ChestX-Det Target-Namen)
    _WANT = {
        "Right Lung": ("rechtes", "lung"),
        "Left Lung": ("linkes", "lung"),
        "Heart": ("Herzregion", "region"),
        "Mediastinum": ("Mediastinum", "region"),
    }

    def __init__(self, device: Optional[str] = None):
        import config

        self.device = device or config.get_device()
        self._model = None
        self._targets = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torchxrayvision as xrv

            from medrax.utils.xrv_utils import safe_console

            with safe_console():  # schützt vor '█'-UnicodeError beim Gewichte-Download
                self._model = xrv.baseline_models.chestx_det.PSPNet()
            self._model = self._model.eval().to(self.device)
            self._targets = list(self._model.targets)
            print("[xrv-anatomy] PSPNet geladen. Targets:", len(self._targets))
            return True
        except Exception as e:
            print("[xrv-anatomy] TorchXRayVision-Segmenter nicht verfügbar:", e)
            self._model = None
            return False

    def segment(self, image_path: str) -> dict:
        """Liefert {available, zones:{name:mask}, regions:{...}}."""
        if not self._ensure_model():
            return {"available": False, "zones": {}, "regions": {}}

        import torch

        from medrax.tools.localization import split_lung_zones
        from medrax.utils.xrv_utils import load_xrv_tensor

        tensor, _ = load_xrv_tensor(image_path, size=512)
        with torch.no_grad():
            logits = self._model(tensor.to(self.device))
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # [T,512,512]

        idx = {name: i for i, name in enumerate(self._targets)}

        def mask_for(name):
            i = idx.get(name)
            return (probs[i] > 0.5) if i is not None else None

        right = mask_for("Right Lung")
        left = mask_for("Left Lung")
        heart = mask_for("Heart")
        med = mask_for("Mediastinum")

        zones: Dict[str, np.ndarray] = {}
        if right is not None:
            zones.update(split_lung_zones(right, "rechtes"))
        if left is not None:
            zones.update(split_lung_zones(left, "linkes"))
        if heart is not None and heart.sum() > 0:
            zones["Herzregion"] = heart
        if med is not None and med.sum() > 0:
            zones["Mediastinum"] = med

        def present(m) -> bool:
            return bool(m is not None and m.sum() > 0.001 * m.size)

        regions = {
            "rechte_lunge": present(right),
            "linke_lunge": present(left),
            "herz": present(heart),
            "mediastinum": present(med),
        }
        return {"available": True, "zones": zones, "regions": regions}

    def _run(self, image_path: str, heatmap: Optional[np.ndarray] = None,
             top_prob: float = 0.0, top_name: str = "No finding"):
        from pathlib import Path

        if not Path(image_path).exists():
            return failed(f"Bild nicht gefunden: {image_path}", image_path=image_path)

        seg = self.segment(image_path)
        if not seg["available"]:
            return failed(
                "TorchXRayVision-Anatomie-Segmenter nicht verfügbar. "
                "Installiere ihn mit:  pip install torchxrayvision",
                image_path=image_path, tool="xrv_anatomy_segmenter",
            )

        localization = None
        if heatmap is not None and seg["zones"]:
            from medrax.tools.localization import localize

            localization = localize(heatmap, seg["zones"], top_prob, top_name, method="xrv")

        return (
            {
                "regions": seg["regions"],
                "zones_available": sorted(seg["zones"].keys()),
                "localization": localization,
            },
            make_metadata("completed", image_path=image_path, method="xrv"),
        )
