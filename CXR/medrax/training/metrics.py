"""
Metriken für Multi-Label-CXR-Klassifikation.

Robust gegen degenerierte Fälle (wenige Bilder, Klassen ohne Positive/Negative):
betroffene Werte werden zu NaN und klar gemeldet, statt das Skript abstürzen zu lassen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC, NaN wenn nur eine Klasse vorhanden ist."""
    try:
        from sklearn.metrics import roc_auc_score

        if y_true.sum() > 0 and (1 - y_true).sum() > 0:
            return float(roc_auc_score(y_true, y_score))
    except Exception:
        pass
    return float("nan")


def compute_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    threshold: float = 0.5,
) -> Dict:
    """
    targets/probs: [N, C]. Liefert pro-Klasse und aggregierte Metriken.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    targets = np.asarray(targets)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)

    n_samples = int(targets.shape[0])
    per_class = {}
    aucs, f1s, precs, recs = [], [], [], []

    for i, name in enumerate(class_names):
        yt = targets[:, i].astype(int)
        ys = probs[:, i]
        yp = preds[:, i]
        auc = _safe_auc(yt, ys)
        # F1/Precision/Recall sind auch bei nur einer Klasse berechenbar (ggf. 0)
        f1 = float(f1_score(yt, yp, zero_division=0))
        prec = float(precision_score(yt, yp, zero_division=0))
        rec = float(recall_score(yt, yp, zero_division=0))

        per_class[name] = {
            "auc": auc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "n_positive": int(yt.sum()),
        }
        if not np.isnan(auc):
            aucs.append(auc)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

    macro_auc = float(np.mean(aucs)) if aucs else float("nan")
    micro_auc = _safe_auc(targets.ravel().astype(int), probs.ravel())

    result = {
        "n_samples": n_samples,
        "threshold": threshold,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "macro_precision": float(np.mean(precs)) if precs else float("nan"),
        "macro_recall": float(np.mean(recs)) if recs else float("nan"),
        "per_class": per_class,
        "warnings": [],
    }

    # Verständliche Hinweise bei aussagelosen Metriken
    if n_samples < 20:
        result["warnings"].append(
            f"Nur {n_samples} Beispiele – Metriken (besonders AUC) sind statistisch "
            "nicht aussagekräftig. Für echte Werte mehr Daten nutzen (Colab: 15.000)."
        )
    if np.isnan(macro_auc):
        result["warnings"].append(
            "Makro-AUC = NaN: keine Klasse hatte sowohl Positive als auch Negative."
        )
    return result


def format_metrics(metrics: Dict) -> str:
    """Hübsche Textausgabe der Metriken."""
    lines = [
        f"Samples: {metrics['n_samples']}  |  Threshold: {metrics['threshold']}",
        f"macro AUC: {metrics['macro_auc']:.4f}   micro AUC: {metrics['micro_auc']:.4f}",
        f"macro F1: {metrics['macro_f1']:.4f}   "
        f"macro P: {metrics['macro_precision']:.4f}   macro R: {metrics['macro_recall']:.4f}",
        "-" * 60,
        f"{'Klasse':<22}{'AUC':>8}{'F1':>8}{'Prec':>8}{'Rec':>8}{'n+':>6}",
    ]
    for name, m in metrics["per_class"].items():
        auc = "nan" if np.isnan(m["auc"]) else f"{m['auc']:.3f}"
        lines.append(
            f"{name:<22}{auc:>8}{m['f1']:>8.3f}{m['precision']:>8.3f}"
            f"{m['recall']:>8.3f}{m['n_positive']:>6}"
        )
    for w in metrics.get("warnings", []):
        lines.append(f"[!] {w}")
    return "\n".join(lines)


def save_metrics(metrics: Dict, json_path, csv_path: Optional[str] = None) -> str:
    """Speichert Metriken als JSON (und optional per-Klasse als CSV)."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if csv_path:
        try:
            import pandas as pd

            rows = [{"class": k, **v} for k, v in metrics["per_class"].items()]
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[metrics] CSV konnte nicht geschrieben werden: {e}")
    return str(json_path)
