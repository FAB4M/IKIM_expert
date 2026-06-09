"""Reproduzierbarkeit: globale Seeds setzen."""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Setzt Seeds für random, numpy und torch (falls vorhanden)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
