"""
Bild-Hilfsfunktionen: einheitliches Laden von PNG/JPG/DICOM, Konvertierung in
Modell-Tensoren (torchvision-Transforms) und Heatmap-Overlay (Grad-CAM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .dicom_utils import is_dicom, read_dicom_pil

PathLike = Union[str, Path]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_image_any(path: PathLike, apply_clahe: bool = True):
    """
    Lädt PNG/JPG/DICOM als PIL.Image. DICOM wird über dicom_utils dekodiert.
    Gibt ein RGB-Bild zurück (Graustufe auf 3 Kanäle dupliziert).
    """
    from PIL import Image

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Bild nicht gefunden: {p}\n-> Prüfe den Pfad (config.TEST_DIR etc.)."
        )

    if is_dicom(p):
        img = read_dicom_pil(p, apply_clahe=apply_clahe)  # Modus 'L'
    else:
        try:
            img = Image.open(p)
        except Exception as e:
            raise RuntimeError(f"Bild '{p.name}' konnte nicht geöffnet werden: {e}")
    return img.convert("RGB")


def build_eval_transform(img_size: int, mean=(0.5,), std=(0.5,)):
    """torchvision-Transform für Inferenz/Eval (deterministisch)."""
    import torchvision.transforms as T

    # mean/std auf 3 Kanäle erweitern, falls Graustufe angegeben wurde
    mean3 = tuple(mean) * 3 if len(mean) == 1 else tuple(mean)
    std3 = tuple(std) * 3 if len(std) == 1 else tuple(std)
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean3, std=std3),
        ]
    )


def build_train_transform(img_size: int, mean=(0.5,), std=(0.5,)):
    """torchvision-Transform fürs Training (leichte Augmentierung)."""
    import torchvision.transforms as T

    mean3 = tuple(mean) * 3 if len(mean) == 1 else tuple(mean)
    std3 = tuple(std) * 3 if len(std) == 1 else tuple(std)
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=8),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean3, std=std3),
        ]
    )


def image_to_tensor(img, img_size: int, mean=(0.5,), std=(0.5,)):
    """PIL.Image -> Tensor [1,3,H,W] für die Inferenz."""
    tf = build_eval_transform(img_size, mean, std)
    return tf(img).unsqueeze(0)


def overlay_heatmap(base_img, heatmap: np.ndarray, alpha: float = 0.45):
    """
    Legt eine Grad-CAM-Heatmap (HxW, Werte 0..1) über ein Basisbild (PIL RGB).
    Gibt ein PIL.Image (RGB) zurück. Nutzt cv2-Colormap, falls vorhanden.
    """
    from PIL import Image

    base = base_img.convert("RGB")
    W, H = base.size
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        hm = hm.squeeze()
    # normalisieren
    hm = hm - hm.min()
    denom = hm.max() + 1e-8
    hm = hm / denom

    try:
        import cv2

        hm_resized = cv2.resize(hm, (W, H))
        hm_uint8 = (hm_resized * 255).astype(np.uint8)
        color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)  # BGR
        color = color[:, :, ::-1]  # -> RGB
        overlay = (alpha * color + (1 - alpha) * np.asarray(base)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    except Exception:
        # Fallback ohne cv2: Heatmap als roter Kanal
        hm_img = Image.fromarray((hm * 255).astype(np.uint8)).resize((W, H)).convert("L")
        red = Image.merge("RGB", (hm_img, Image.new("L", (W, H)), Image.new("L", (W, H))))
        return Image.blend(base, red, alpha)


def save_image(img, path: PathLike) -> str:
    """Speichert ein PIL.Image; legt Elternverzeichnis an. Gibt Pfad als str zurück."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(p))
    return str(p)


def list_images(folder: PathLike, limit: Optional[int] = None):
    """Listet Bild-/DICOM-Dateien in einem Ordner (sortiert)."""
    p = Path(folder)
    if not p.exists():
        return []
    files = [
        f for f in sorted(p.iterdir())
        if f.is_file() and (f.suffix.lower() in IMAGE_EXTS or is_dicom(f))
    ]
    return files[:limit] if limit else files
