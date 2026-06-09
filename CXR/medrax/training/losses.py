"""Loss-Funktionen für Multi-Label-Training (BCEWithLogitsLoss + pos_weight)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def compute_pos_weight(targets: np.ndarray, clip: float = 20.0) -> torch.Tensor:
    """
    pos_weight pro Klasse = #Negative / #Positive (gegen Klassenungleichgewicht),
    geklippt auf [.., clip]. targets: [N, C] (0/1).
    """
    targets = np.asarray(targets, dtype=np.float32)
    n = targets.shape[0]
    pos = targets.sum(axis=0)
    neg = n - pos
    pos_weight = neg / (pos + 1e-6)
    pos_weight = np.minimum(pos_weight, clip)
    # Klassen ohne Positive: Gewicht auf 1.0 (neutral) statt clip
    pos_weight[pos == 0] = 1.0
    return torch.tensor(pos_weight, dtype=torch.float32)


def build_criterion(pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    """BCEWithLogitsLoss (Multi-Label). pos_weight optional."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
