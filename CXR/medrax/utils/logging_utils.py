"""Einheitliches Logging: Konsole + optionale Logdatei unter logs/."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(
    name: str = "kimi_cxr",
    logfile: Optional[PathLike] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Liefert einen konfigurierten Logger. Schreibt auf die Konsole und – falls
    logfile gesetzt ist – zusätzlich in eine Datei (Verzeichnis wird angelegt).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Doppelte Handler vermeiden (z. B. bei Reimport in Notebooks)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if logfile is None:
        try:
            import config

            logfile = Path(config.LOG_DIR) / f"{name}.log"
        except Exception:
            logfile = None

    if logfile is not None:
        try:
            lf = Path(logfile)
            lf.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(lf, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:  # Logging darf nie hart fehlschlagen
            logger.warning("Logdatei konnte nicht erstellt werden: %s", e)

    return logger
