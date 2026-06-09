"""MyChestXRayClassifierTool – VinBigData-15 Multi-Label-Classifier + Grad-CAM.

Sigmoid-Wahrscheinlichkeiten je Pathologie; Heatmap der wahrscheinlichsten Pathologie.
Screening, keine Detektion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseCXRTool, BaseModel, Field, failed, make_metadata


class ChestXRayClassifierInput(BaseModel):
    image_path: str = Field(
        ..., description="Pfad zu einem CXR-Bild (PNG/JPG; DICOM wird ebenfalls akzeptiert)."
    )


class MyChestXRayClassifierTool(BaseCXRTool):
    name = "my_chest_xray_classifier"
    description = (
        "Eigener, selbst trainierter Chest-X-Ray-Classifier (VinBigData-15: 14 Befunde + "
        "'No finding'). Gibt für JEDE der 15 Klassen eine Wahrscheinlichkeit (0..1, Sigmoid) "
        "zurück und erzeugt eine Grad-CAM-Heatmap der wahrscheinlichsten Pathologie. "
        "NUTZEN, um ein Röntgen-Thorax-Bild auf Auffälligkeiten zu screenen. "
        "Eingabe: image_path (PNG/JPG/DICOM). Ausgabe: output['predictions'] = {Klasse: prob}, "
        "output['heatmap_path'] = Overlay-PNG, output['top'] = wahrscheinlichste Klassen. "
        "INTERPRETATION: Wahrscheinlichkeiten sind Screening-Hinweise, KEINE Diagnose und "
        "KEINE Objektdetektion. Hohe 'No finding'-Wahrscheinlichkeit spricht gegen Befunde."
    )
    args_schema = ChestXRayClassifierInput

    def __init__(self, checkpoint: Optional[str] = None, device: Optional[str] = None,
                 img_size: Optional[int] = None):
        import config

        from medrax.training.labels import NO_FINDING_INDEX, VINBIGDATA_CLASSES
        from medrax.training.model import get_gradcam_target_layer, load_checkpoint
        from medrax.utils.paths import find_classifier_checkpoint

        self.classes = list(VINBIGDATA_CLASSES)
        self.no_finding_index = NO_FINDING_INDEX
        self.device = device or config.get_device()
        self.img_size = img_size or config.IMG_SIZE
        self.viz_dir = Path(config.VIZ_DIR)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = checkpoint or find_classifier_checkpoint()
        if ckpt_path is None:
            raise FileNotFoundError(
                "Kein Classifier-Checkpoint gefunden (weights/classifier/).\n"
                "-> Trainiere zuerst:  python -m medrax.training.train_classifier "
                "--max-samples 10 --epochs 1 --batch-size 2\n"
                "   oder erzeuge einen Mini-Checkpoint mit scripts/test_classifier_10_images.py."
            )
        self.checkpoint_path = str(ckpt_path)
        self.model, self.ckpt = load_checkpoint(ckpt_path, device=self.device)
        self.target_layer = get_gradcam_target_layer(self.model)

    # ------------------------------------------------------------------ core
    def analyze(self, image_path: str, make_heatmap: bool = True) -> dict:
        """Vollständige Analyse als verschachteltes Dict (für Agent/Pipeline)."""
        import torch

        from medrax.utils.image_utils import image_to_tensor, load_image_any, overlay_heatmap, save_image

        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

        pil = load_image_any(p)
        tensor = image_to_tensor(pil, self.img_size).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(tensor)).cpu().numpy().ravel()

        predictions = {self.classes[i]: float(round(float(probs[i]), 4)) for i in range(len(probs))}
        order = np.argsort(-probs)
        top = [(self.classes[i], float(round(float(probs[i]), 4))) for i in order[:5]]

        # wahrscheinlichste *Pathologie* (ohne 'No finding') für die Lokalisation
        path_order = [i for i in order if i != self.no_finding_index]
        top_path_idx = int(path_order[0]) if path_order else int(order[0])
        top_path_name = self.classes[top_path_idx]
        top_path_prob = float(probs[top_path_idx])

        heatmap = None
        heatmap_path = None
        if make_heatmap and self.target_layer is not None:
            try:
                from medrax.training.gradcam import compute_gradcam

                heatmap, _ = compute_gradcam(self.model, self.target_layer, tensor, top_path_idx)
                overlay = overlay_heatmap(pil, heatmap, alpha=0.45)
                heatmap_path = str(self.viz_dir / f"{p.stem}_gradcam_{top_path_name.replace('/', '-')}.png")
                save_image(overlay, heatmap_path)
            except Exception as e:
                print(f"[classifier] Grad-CAM übersprungen: {e}")

        metadata = make_metadata(
            "completed",
            image_path=str(p),
            model_checkpoint=self.checkpoint_path,
            label_set="VinBigData-15",
            device=self.device,
            img_size=self.img_size,
            heatmap_path=heatmap_path,
        )
        return {
            "predictions": predictions,
            "top": top,
            "top_pathology": {"name": top_path_name, "prob": round(top_path_prob, 4),
                              "index": top_path_idx},
            "heatmap": heatmap,             # np.ndarray (für Lokalisation; nicht JSON)
            "heatmap_path": heatmap_path,
            "input_image": str(p),
            "metadata": metadata,
        }

    def _run(self, image_path: str):
        """(output, metadata)."""
        try:
            res = self.analyze(image_path)
        except FileNotFoundError as e:
            return failed(str(e), image_path=image_path)
        except Exception as e:
            return failed(f"Klassifikation fehlgeschlagen: {e}", image_path=image_path)

        output = {
            "predictions": res["predictions"],
            "top": res["top"],
            "top_pathology": res["top_pathology"],
            "heatmap_path": res["heatmap_path"],
        }
        return output, res["metadata"]
