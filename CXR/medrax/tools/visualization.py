"""
ImageVisualizerTool – lädt ein CXR (PNG/JPG/DICOM) und speichert eine Vorschau,
optional mit Ground-Truth-Bounding-Boxes (NUR Visualisierung/Sanity, KEINE Detektion).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .base import BaseCXRTool, BaseModel, Field, failed, make_metadata


class ImageVisualizerInput(BaseModel):
    image_path: str = Field(..., description="Pfad zu einem CXR-Bild (PNG/JPG/DICOM).")


class ImageVisualizerTool(BaseCXRTool):
    name = "image_visualizer"
    description = (
        "Lädt ein Röntgenbild (PNG/JPG/DICOM) und speichert eine PNG-Vorschau unter "
        "outputs/visualizations. NUTZEN, um ein Bild für den Nutzer sichtbar zu machen "
        "oder als Basis für Overlays. Eingabe: image_path. Ausgabe: output['image_path'] "
        "= Pfad zur Vorschau, plus Bildgröße in den Metadaten."
    )
    args_schema = ImageVisualizerInput

    def __init__(self, output_dir: Optional[str] = None):
        import config

        self.output_dir = Path(output_dir) if output_dir else Path(config.VIZ_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run(self, image_path: str):
        from medrax.utils.image_utils import load_image_any, save_image

        try:
            img = load_image_any(image_path)
        except Exception as e:
            return failed(str(e), image_path=image_path)

        out = self.output_dir / f"{Path(image_path).stem}_preview.png"
        save_image(img, out)
        return (
            {"image_path": str(out)},
            make_metadata("completed", source=image_path, image_path=str(out),
                          width=img.size[0], height=img.size[1]),
        )


def draw_boxes(image_path: str, boxes: List[dict], out_path: str,
               heatmap=None, alpha: float = 0.45) -> str:
    """
    Zeichnet GT-Boxen (und optional eine Grad-CAM-Heatmap) auf ein Bild.
    boxes: Liste von {x_min,y_min,x_max,y_max,label?}. NUR Visualisierung/Sanity.
    """
    from PIL import Image, ImageDraw

    from medrax.utils.image_utils import load_image_any, overlay_heatmap

    img = load_image_any(image_path)
    if heatmap is not None:
        img = overlay_heatmap(img, heatmap, alpha=alpha)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in boxes:
        try:
            xy = [float(b["x_min"]), float(b["y_min"]), float(b["x_max"]), float(b["y_max"])]
            draw.rectangle(xy, outline=(0, 255, 0), width=3)
            if b.get("label"):
                draw.text((xy[0] + 2, xy[1] + 2), str(b["label"]), fill=(0, 255, 0))
        except Exception:
            continue
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
