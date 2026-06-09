"""DICOM-Hilfsfunktionen: Decodierung (inkl. JPEG2000), Rescale, Window/Level,
MONOCHROME1-Invertierung, 8-bit, optional CLAHE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

# Endungen, die wir als DICOM behandeln
DICOM_EXTS = {".dcm", ".dicom", ".dic"}


class DicomReadError(RuntimeError):
    """Verständlicher Fehler beim DICOM-Lesen/Dekodieren."""


def is_dicom(path: PathLike) -> bool:
    p = Path(path)
    if p.suffix.lower() in DICOM_EXTS:
        return True
    # Manche DICOMs haben keine Endung -> Magic 'DICM' an Offset 128 prüfen
    try:
        with open(p, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def _require_pydicom():
    try:
        import pydicom  # noqa: F401

        return pydicom
    except Exception as e:  # pragma: no cover
        raise DicomReadError(
            "pydicom ist nicht installiert. Bitte ausführen:\n"
            "    pip install pydicom\n"
            f"(Originalfehler: {e})"
        )


def _apply_window(arr: np.ndarray, center: float, width: float) -> np.ndarray:
    """Window/Level wie bei radiologischer Darstellung."""
    width = max(float(width), 1.0)
    low = center - width / 2.0
    high = center + width / 2.0
    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low + 1e-8)
    return arr


def read_dicom_array(
    path: PathLike,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    apply_clahe: bool = True,
) -> np.ndarray:
    """
    Liest eine DICOM-Datei und gibt ein 8-bit-Graustufenbild (uint8, HxW) zurück.

    - Rescale-Slope/Intercept werden angewandt, falls vorhanden.
    - Wenn window_center/width gesetzt sind, wird Window/Level genutzt,
      sonst Min/Max-Normalisierung (wie im Notebook).
    - MONOCHROME1 wird invertiert.
    - Optional CLAHE (Kontrastverstärkung).
    """
    pydicom = _require_pydicom()
    p = Path(path)
    if not p.exists():
        raise DicomReadError(f"DICOM-Datei nicht gefunden: {p}")

    try:
        ds = pydicom.dcmread(str(p))
    except Exception as e:
        raise DicomReadError(f"DICOM konnte nicht gelesen werden ({p.name}): {e}")

    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception as e:
        raise DicomReadError(
            f"Pixel-Daten von '{p.name}' konnten nicht dekodiert werden: {e}\n"
            "Bei komprimierten (z. B. JPEG2000) DICOMs hilft eines davon:\n"
            "    pip install pylibjpeg pylibjpeg-openjpeg pylibjpeg-libjpeg\n"
            "    oder:  pip install python-gdcm"
        )

    # Rescale slope/intercept
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept

    # Window/Level oder Min/Max
    if window_center is None and window_width is None:
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        # WindowCenter/Width können MultiValue sein
        if wc is not None and ww is not None:
            try:
                wc = float(wc[0]) if hasattr(wc, "__iter__") else float(wc)
                ww = float(ww[0]) if hasattr(ww, "__iter__") else float(ww)
                window_center, window_width = wc, ww
            except Exception:
                window_center, window_width = None, None

    if window_center is not None and window_width is not None:
        arr = _apply_window(arr, float(window_center), float(window_width))
    else:
        amin, amax = float(arr.min()), float(arr.max())
        arr = (arr - amin) / (amax - amin + 1e-8)

    img = (arr * 255.0).clip(0, 255).astype(np.uint8)

    # MONOCHROME1 -> invertieren (helle Knochen erwartet)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        img = 255 - img

    if apply_clahe:
        img = apply_clahe_gray(img)

    return img


def apply_clahe_gray(img: np.ndarray) -> np.ndarray:
    """Leichter Gaussian-Blur + CLAHE (clipLimit=2.0, tile 8x8) – wie im Notebook."""
    try:
        import cv2

        blur = cv2.GaussianBlur(img, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(blur)
    except Exception:
        # cv2 nicht verfügbar -> ohne CLAHE zurückgeben (kein harter Fehler)
        return img


def read_dicom_pil(
    path: PathLike,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    apply_clahe: bool = True,
):
    """Wie read_dicom_array, aber als PIL.Image (Modus 'L')."""
    from PIL import Image

    arr = read_dicom_array(
        path,
        window_center=window_center,
        window_width=window_width,
        apply_clahe=apply_clahe,
    )
    return Image.fromarray(arr, mode="L")


def dicom_metadata(path: PathLike) -> dict:
    """Kleine, sichere Metadaten-Extraktion (für Tool-Outputs)."""
    pydicom = _require_pydicom()
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)

    def g(attr, default=None):
        return getattr(ds, attr, default)

    return {
        "Rows": int(g("Rows", 0) or 0),
        "Columns": int(g("Columns", 0) or 0),
        "PhotometricInterpretation": str(g("PhotometricInterpretation", "")),
        "BitsStored": int(g("BitsStored", 0) or 0),
        "Modality": str(g("Modality", "")),
        "TransferSyntaxUID": str(getattr(ds.file_meta, "TransferSyntaxUID", "")),
    }
