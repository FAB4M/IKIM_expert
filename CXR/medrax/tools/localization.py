"""Lokalisierungs-Logik: Grad-CAM-Heatmap × Zonen-Masken -> grobe Zone (mit Gating)."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

# Patienten-Felder (Adjektiv passend zu "Feld")
_SIDE_ADJ = ["rechtes", "linkes"]
_BANDS = ["Oberfeld", "Mittelfeld", "Unterfeld"]


def heatmap_entropy(heatmap: np.ndarray) -> float:
    """Normalisierte Entropie der Heatmap (0=fokussiert .. 1=diffus)."""
    h = np.asarray(heatmap, dtype=np.float64).ravel()
    h = np.clip(h, 0, None)
    s = h.sum()
    if s <= 1e-8:
        return 1.0
    p = h / s
    p = p[p > 0]
    if len(p) <= 1:
        return 1.0
    ent = -np.sum(p * np.log(p))
    return float(ent / math.log(len(p)))


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    H, W = size
    mask = np.asarray(mask)
    if mask.shape == (H, W):
        return mask.astype(bool)
    try:
        import cv2

        m = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return m.astype(bool)
    except Exception:
        from PIL import Image

        m = Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H))
        return np.asarray(m) > 127


def grid_zones(h: int, w: int) -> Dict[str, np.ndarray]:
    """3x2-Raster über das Bild (nur Notbehelf, KEINE Anatomie)."""
    zones = {}
    for r, band in enumerate(_BANDS):
        for c, side in enumerate(_SIDE_ADJ):
            mask = np.zeros((h, w), dtype=bool)
            y0, y1 = int(r * h / 3), int((r + 1) * h / 3)
            x0, x1 = int(c * w / 2), int((c + 1) * w / 2)
            mask[y0:y1, x0:x1] = True
            zones[f"{side} {band}"] = mask
    return zones


def split_lung_zones(mask: np.ndarray, side_adj: str) -> Dict[str, np.ndarray]:
    """Teilt eine Lungenmaske in Ober-/Mittel-/Unterfeld (nach y-Bounding-Box)."""
    zones: Dict[str, np.ndarray] = {}
    mask = np.asarray(mask).astype(bool)
    if mask.sum() == 0:
        return zones
    ys = np.where(mask.any(axis=1))[0]
    y0, y1 = ys.min(), ys.max()
    thirds = np.linspace(y0, y1 + 1, 4).astype(int)
    for b, band in enumerate(_BANDS):
        z = np.zeros_like(mask)
        z[thirds[b]:thirds[b + 1], :] = mask[thirds[b]:thirds[b + 1], :]
        zones[f"{side_adj} {band}"] = z
    return zones


def localize(
    heatmap: np.ndarray,
    zones: Optional[Dict[str, np.ndarray]],
    top_prob: float,
    top_name: str,
    method: str,
) -> dict:
    """
    Grad-CAM x Zonen -> grobe Lokalisation mit Gating (Schwellen aus config):
      * top_prob >= LOC_MIN_PATHOLOGY_PROB
      * top_name != 'No finding'
      * Top-Zonen-Anteil >= LOC_MIN_ZONE_DOMINANCE
      * Margin (Top - 2.) >= LOC_MIN_ZONE_MARGIN
      * Heatmap-Entropie <= LOC_MAX_HEATMAP_ENTROPY
    """
    import config

    hm = np.asarray(heatmap, dtype=np.float32)
    H, W = hm.shape
    if not zones:
        zones = grid_zones(H, W)
        method = "grid"
    else:
        zones = {k: resize_mask(m, (H, W)) for k, m in zones.items() if np.asarray(m).sum() > 0}
        if not zones:
            zones = grid_zones(H, W)
            method = "grid"

    activations = {name: (float(hm[m].mean()) if m.sum() > 0 else 0.0)
                   for name, m in zones.items()}
    total = sum(activations.values()) + 1e-8
    fractions = {k: v / total for k, v in activations.items()}
    ranked = sorted(fractions.items(), key=lambda x: x[1], reverse=True)
    top_zone, top_frac = ranked[0]
    second_frac = ranked[1][1] if len(ranked) > 1 else 0.0
    entropy = heatmap_entropy(hm)

    reasons = []
    if top_name == "No finding":
        reasons.append("Top-Klasse ist 'No finding'")
    if top_prob < config.LOC_MIN_PATHOLOGY_PROB:
        reasons.append(f"Wahrscheinlichkeit {top_prob:.2f} < {config.LOC_MIN_PATHOLOGY_PROB}")
    if top_frac < config.LOC_MIN_ZONE_DOMINANCE:
        reasons.append(f"keine dominante Zone (Anteil {top_frac:.2f})")
    if (top_frac - second_frac) < config.LOC_MIN_ZONE_MARGIN:
        reasons.append(f"Zonen zu ähnlich (Margin {top_frac - second_frac:.2f})")
    if entropy > config.LOC_MAX_HEATMAP_ENTROPY:
        reasons.append(f"Heatmap zu diffus (Entropie {entropy:.2f})")

    localized = len(reasons) == 0
    return {
        "localized": localized,
        "zone": top_zone if localized else None,
        "zone_fraction": round(top_frac, 3),
        "margin": round(top_frac - second_frac, 3),
        "heatmap_entropy": round(entropy, 3),
        "method": method,  # 'xrv' | 'ianpan' | 'grid'
        "ranking": [(k, round(v, 3)) for k, v in ranked[:6]],
        "reason": None if localized else "; ".join(reasons),
        "note": (
            "Anatomische Zonen (TorchXRayVision)." if method == "xrv"
            else "Anatomische Zonen (ianpan)." if method == "ianpan"
            else "Grobes Bildraster, KEINE anatomische Segmentierung."
        ),
    }
