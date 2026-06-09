"""Kleine Pfad-Hilfsfunktionen rund um config.py."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Legt ein Verzeichnis (rekursiv) an und gibt es als Path zurück."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: PathLike) -> Path:
    """Stellt sicher, dass das Elternverzeichnis existiert."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def unique_path(folder: PathLike, stem: str, suffix: str) -> Path:
    """Erzeugt einen nicht-kollidierenden Pfad folder/stem(_n).suffix."""
    folder = ensure_dir(folder)
    suffix = suffix if suffix.startswith(".") else "." + suffix
    candidate = folder / f"{stem}{suffix}"
    i = 1
    while candidate.exists():
        candidate = folder / f"{stem}_{i}{suffix}"
        i += 1
    return candidate


def find_classifier_checkpoint() -> Optional[Path]:
    """
    Sucht einen Classifier-Checkpoint an den üblichen Orten:
    weights/classifier/best_model.pt -> last_model.pt -> models/**/*.pt(h).
    Gibt None zurück, wenn nichts gefunden wird.
    """
    import config

    candidates = [config.BEST_CKPT, config.LAST_CKPT]
    for c in candidates:
        if Path(c).exists():
            return Path(c)

    # Fallback: in models/ nach *.pt / *.pth suchen (Notebook speichert dort)
    model_dir = Path(config.MODEL_DIR)
    if model_dir.exists():
        hits = sorted(model_dir.rglob("*.pt")) + sorted(model_dir.rglob("*.pth"))
        if hits:
            return hits[0]
    return None
