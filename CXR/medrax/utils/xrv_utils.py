"""
Gemeinsame Vorverarbeitung für TorchXRayVision-Modelle.

XRV erwartet 1-Kanal-Bilder, normalisiert auf den Bereich [-1024, 1024].
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


class _SafeWriter:
    """stdout-Wrapper, der nicht-kodierbare Zeichen ersetzt (Windows cp1252 + '█')."""

    def __init__(self, base):
        self._base = base

    def write(self, s):
        try:
            self._base.write(s)
        except UnicodeEncodeError:
            enc = getattr(self._base, "encoding", "ascii") or "ascii"
            self._base.write(s.encode(enc, "replace").decode(enc))

    def flush(self):
        try:
            self._base.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._base, name)


@contextlib.contextmanager
def safe_console():
    """
    Schützt vor UnicodeEncodeError, wenn TorchXRayVision beim Gewichte-Download
    Fortschrittsbalken mit '█' (U+2588) auf eine cp1252-Konsole druckt.
    """
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _SafeWriter(old_out), _SafeWriter(old_err)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def xrv_normalize(arr_uint8: np.ndarray) -> np.ndarray:
    """0..255 -> [-1024, 1024] (entspricht xrv.datasets.normalize(img, 255))."""
    a = arr_uint8.astype(np.float32)
    return (a / 255.0 * 2.0 - 1.0) * 1024.0


def load_xrv_tensor(image_path, size: int = 224, apply_clahe: bool = True):
    """
    Lädt PNG/JPG/DICOM als XRV-Eingabetensor.
    Rückgabe: (tensor [1,1,size,size] float32, pil_gray PIL.Image 'L' in size).
    """
    import torch
    from PIL import Image

    from medrax.utils.dicom_utils import is_dicom, read_dicom_array

    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    if is_dicom(p):
        arr = read_dicom_array(p, apply_clahe=apply_clahe)  # uint8 HxW
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.open(p).convert("L")

    img_resized = img.resize((size, size))
    arr = np.asarray(img_resized, dtype=np.uint8)
    norm = xrv_normalize(arr)
    tensor = torch.from_numpy(norm)[None, None].float()  # [1,1,H,W]
    return tensor, img_resized
