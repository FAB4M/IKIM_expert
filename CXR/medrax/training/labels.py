"""CXR-Labels.

Aktiv: VinBigData-15 (14 Befunde + 'No finding', class_id-Reihenfolge).
Die TorchXRayVision-18-Liste dient nur als Referenz-Mapping.
"""

from __future__ import annotations

from typing import Dict, List

# --------------------------------------------------------------------------
# AKTIV: VinBigData-15 (Index == offizielle class_id)
# --------------------------------------------------------------------------
VINBIGDATA_CLASSES: List[str] = [
    "Aortic enlargement",   # 0
    "Atelectasis",          # 1
    "Calcification",        # 2
    "Cardiomegaly",         # 3
    "Consolidation",        # 4
    "ILD",                  # 5
    "Infiltration",         # 6
    "Lung Opacity",         # 7
    "Nodule/Mass",          # 8
    "Other lesion",         # 9
    "Pleural effusion",     # 10
    "Pleural thickening",   # 11
    "Pneumothorax",         # 12
    "Pulmonary fibrosis",   # 13
    "No finding",           # 14
]

NUM_CLASSES: int = len(VINBIGDATA_CLASSES)
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(VINBIGDATA_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: name for i, name in enumerate(VINBIGDATA_CLASSES)}

# Index der "No finding"-Klasse (für Lokalisierungs-Gating wichtig)
NO_FINDING_INDEX: int = CLASS_TO_IDX["No finding"]

# Referenz (nicht aktiv): TorchXRayVision – 18 Pathologien
TXRV_REFERENCE: List[str] = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis", "Fracture",
    "Hernia", "Infiltration", "Lung Lesion", "Lung Opacity", "Mass",
    "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax",
]

# Best-effort-Mapping VinBigData -> TorchXRayVision (NUR Referenz/Anzeige).
# Klassen ohne sinnvolles Pendant zeigen auf None.
VINBIG_TO_TXRV: Dict[str, str | None] = {
    "Aortic enlargement": "Enlarged Cardiomediastinum",  # grobe Näherung
    "Atelectasis": "Atelectasis",
    "Calcification": None,
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "ILD": "Infiltration",            # grobe Näherung
    "Infiltration": "Infiltration",
    "Lung Opacity": "Lung Opacity",
    "Nodule/Mass": "Nodule",          # bzw. Mass
    "Other lesion": "Lung Lesion",
    "Pleural effusion": "Effusion",
    "Pleural thickening": "Pleural Thickening",
    "Pneumothorax": "Pneumothorax",
    "Pulmonary fibrosis": "Fibrosis",
    "No finding": None,
}


def get_active_classes() -> List[str]:
    """Aktives Label-Set des Classifiers (VinBigData-15)."""
    return list(VINBIGDATA_CLASSES)


def txrv_name(vinbig_name: str) -> str | None:
    """Liefert den (Referenz-)TXRV-Namen zu einer VinBigData-Klasse, falls vorhanden."""
    return VINBIG_TO_TXRV.get(vinbig_name)
