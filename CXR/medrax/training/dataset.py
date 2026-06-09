"""CXR-Dataset. Quellen: wide-CSV, VinBigData-Detection-CSV (aggregiert), oder Ordner ohne
Labels (Dummy-Modus). PNG/JPG/DICOM, optionales DICOM->PNG-Caching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .labels import CLASS_TO_IDX, NO_FINDING_INDEX, NUM_CLASSES, VINBIGDATA_CLASSES

PathLike = Union[str, Path]

# Reihenfolge, in der nach Bilddateien zu einer image_id gesucht wird
_IMG_SUFFIXES = [".png", ".jpg", ".jpeg", ".dicom", ".dcm", ".bmp", ".tif", ".tiff"]


# ==========================================================================
# Hilfsfunktionen zum Aufbau der Sample-Liste
# ==========================================================================
def _resolve_image_path(image_dir: Path, image_id: str) -> Optional[Path]:
    """Sucht zu einer image_id (ohne Endung) die passende Bilddatei."""
    # Falls image_id bereits eine Endung hat:
    p = image_dir / image_id
    if p.exists():
        return p
    for suf in _IMG_SUFFIXES:
        cand = image_dir / f"{image_id}{suf}"
        if cand.exists():
            return cand
    return None


def build_image_level_labels(csv_path: PathLike, class_names: List[str] = VINBIGDATA_CLASSES):
    """
    Aggregiert die VinBigData-Detection-CSV zu image-level Multi-Label-Vektoren.
    Rückgabe: DataFrame mit Spalten [image_id, target(np.ndarray)].
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns:
        raise ValueError(f"CSV ohne 'image_id'-Spalte: {csv_path}")

    name_to_idx = {n: i for i, n in enumerate(class_names)}

    def agg(group) -> np.ndarray:
        vec = np.zeros(len(class_names), dtype=np.float32)
        if "class_id" in group.columns:
            for cid in group["class_id"].dropna().astype(int).unique():
                if 0 <= cid < len(class_names):
                    vec[cid] = 1.0
        elif "class_name" in group.columns:
            for cn in group["class_name"].dropna().unique():
                if cn in name_to_idx:
                    vec[name_to_idx[cn]] = 1.0
        return vec

    rows = []
    for image_id, group in df.groupby("image_id"):
        rows.append({"image_id": str(image_id), "target": agg(group)})
    return pd.DataFrame(rows)


def samples_from_vinbig_csv(
    csv_path: PathLike,
    image_dir: PathLike,
    class_names: List[str] = VINBIGDATA_CLASSES,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Erzeugt Samples aus VinBigData-Detection-CSV + Bildordner."""
    image_dir = Path(image_dir)
    label_df = build_image_level_labels(csv_path, class_names)

    # Schnelle Vorab-Prüfung: passt der Bildordner überhaupt zur CSV?
    # (Vermeidet langsame .exists()-Scans über tausende IDs auf OneDrive/Drive,
    #  wenn die Bilder lokal gar nicht vorliegen.)
    probe_ids = label_df["image_id"].head(25).tolist()
    if not any(_resolve_image_path(image_dir, pid) is not None for pid in probe_ids):
        raise FileNotFoundError(
            f"Keine der ersten {len(probe_ids)} image_ids aus '{Path(csv_path).name}' "
            f"wurde in {image_dir} gefunden.\n"
            "-> Lokal liegen die Train-Bilder evtl. nicht vor (nur in Colab). "
            "Für einen lokalen Smoke-Test den DUMMY-Modus auf data/VinBigData/test nutzen."
        )

    samples, missing = [], 0
    for _, row in label_df.iterrows():
        path = _resolve_image_path(image_dir, row["image_id"])
        if path is None:
            missing += 1
            continue
        samples.append({"image_path": str(path), "target": row["target"], "image_id": row["image_id"]})
        if max_samples and len(samples) >= max_samples:
            break

    if not samples:
        raise FileNotFoundError(
            f"Keine Bilder gefunden, die zu '{csv_path}' passen, in: {image_dir}\n"
            "-> Lokal liegen die Train-Bilder evtl. nicht vor (nur in Colab). "
            "Für einen lokalen Smoke-Test den DUMMY-Modus auf data/VinBigData/test nutzen."
        )
    if missing:
        print(f"[dataset] Hinweis: {missing} image_ids ohne passende Bilddatei übersprungen.")
    return samples


def is_wide_label_csv(csv_path: PathLike, class_names: List[str] = VINBIGDATA_CLASSES) -> bool:
    """Heuristik: enthält die CSV bereits 0/1-Spalten je Pathologie?"""
    import pandas as pd

    cols = set(pd.read_csv(csv_path, nrows=1).columns)
    overlap = len(cols.intersection(set(class_names)))
    return overlap >= max(2, len(class_names) // 2)


def samples_from_wide_csv(
    csv_path: PathLike,
    image_dir: PathLike,
    class_names: List[str] = VINBIGDATA_CLASSES,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Samples aus 'wide' CSV (image_id + Spalte je Klasse)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    image_dir = Path(image_dir)
    present = [c for c in class_names if c in df.columns]

    samples = []
    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        path = _resolve_image_path(image_dir, image_id)
        if path is None:
            continue
        vec = np.zeros(len(class_names), dtype=np.float32)
        for c in present:
            try:
                vec[CLASS_TO_IDX[c]] = float(row[c]) if not pd.isna(row[c]) else 0.0
            except Exception:
                pass
        samples.append({"image_path": str(path), "target": vec, "image_id": image_id})
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def samples_from_folder_dummy(
    folder: PathLike,
    class_names: List[str] = VINBIGDATA_CLASSES,
    max_samples: Optional[int] = None,
    dummy_mode: str = "random",
    seed: int = 42,
) -> List[Dict]:
    """
    DUMMY-MODUS (nur technischer Smoke-Test!): listet Bilder eines Ordners und
    weist KÜNSTLICHE Labels zu. Es findet KEIN echtes/medizinisches Lernen statt.

    dummy_mode: 'random' (zufällige Multi-Hot-Vektoren) oder 'zeros' (alles 'No finding').
    """
    from medrax.utils.image_utils import list_images

    files = list_images(folder, limit=max_samples)
    if not files:
        raise FileNotFoundError(
            f"Keine Bilder/DICOMs gefunden in: {folder}\n"
            "-> Erwartet werden z. B. die 108 .dicom-Dateien unter data/VinBigData/test."
        )

    print(
        "\n[dataset] !!! DUMMY-LABEL-MODUS !!! Künstliche Labels – KEIN echtes Training, "
        "nur technischer Smoke-Test. Ergebnisse sind medizinisch bedeutungslos.\n"
    )
    rng = np.random.default_rng(seed)
    samples = []
    for f in files:
        vec = np.zeros(len(class_names), dtype=np.float32)
        if dummy_mode == "random":
            # 0-3 zufällige Befunde setzen
            k = int(rng.integers(0, 4))
            if k == 0:
                vec[NO_FINDING_INDEX] = 1.0
            else:
                idxs = rng.choice(len(class_names) - 1, size=k, replace=False)
                for ix in idxs:
                    vec[int(ix)] = 1.0
        else:  # zeros
            vec[NO_FINDING_INDEX] = 1.0
        samples.append({"image_path": str(f), "target": vec, "image_id": f.stem, "dummy": True})
    return samples


def build_samples(
    data_dir: Optional[PathLike] = None,
    csv_path: Optional[PathLike] = None,
    image_dir: Optional[PathLike] = None,
    class_names: List[str] = VINBIGDATA_CLASSES,
    max_samples: Optional[int] = None,
    dummy_labels: bool = False,
    dummy_mode: str = "random",
) -> List[Dict]:
    """
    Auto-Auswahl der Quelle:
      - dummy_labels=True ODER keine CSV  -> Ordner-DUMMY-Modus (image_dir)
      - CSV ist 'wide'                     -> samples_from_wide_csv
      - sonst (Detection-CSV)              -> samples_from_vinbig_csv
    """
    image_dir = Path(image_dir) if image_dir else None

    if dummy_labels or not csv_path:
        if image_dir is None:
            raise ValueError("Für den DUMMY-Modus muss image_dir gesetzt sein.")
        return samples_from_folder_dummy(
            image_dir, class_names, max_samples=max_samples, dummy_mode=dummy_mode
        )

    if is_wide_label_csv(csv_path, class_names):
        return samples_from_wide_csv(csv_path, image_dir, class_names, max_samples)
    return samples_from_vinbig_csv(csv_path, image_dir, class_names, max_samples)


# ==========================================================================
# Dataset
# ==========================================================================
class CXRDataset:
    """
    PyTorch-Dataset (erbt erst zur Laufzeit von torch.utils.data.Dataset,
    damit das Modul auch ohne torch importierbar bleibt).
    """

    def __init__(
        self,
        samples: List[Dict],
        transform=None,
        img_size: int = 224,
        cache_dir: Optional[PathLike] = None,
        apply_clahe: bool = True,
        strict: bool = False,
    ):
        self.samples = samples
        self.transform = transform
        self.img_size = img_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.apply_clahe = apply_clahe
        self.strict = strict
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def _load_image(self, sample: Dict):
        from medrax.utils.dicom_utils import is_dicom, read_dicom_pil
        from medrax.utils.image_utils import load_image_any

        path = Path(sample["image_path"])
        # Cache für DICOMs
        if self.cache_dir and is_dicom(path):
            cached = self.cache_dir / f"{path.stem}.png"
            if cached.exists():
                from PIL import Image

                return Image.open(cached).convert("RGB")
            img = read_dicom_pil(path, apply_clahe=self.apply_clahe)
            try:
                img.save(cached)
            except Exception:
                pass
            return img.convert("RGB")
        return load_image_any(path, apply_clahe=self.apply_clahe)

    def __getitem__(self, idx):
        import torch

        sample = self.samples[idx]
        try:
            img = self._load_image(sample)
        except Exception as e:
            if self.strict:
                raise
            print(f"[dataset] WARN: Bild nicht lesbar ({sample['image_path']}): {e} "
                  "-> schwarzes Ersatzbild.")
            from PIL import Image

            img = Image.new("RGB", (self.img_size, self.img_size))

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            from medrax.utils.image_utils import build_eval_transform

            tensor = build_eval_transform(self.img_size)(img)

        target = torch.tensor(np.asarray(sample["target"], dtype=np.float32))
        return tensor, target


def make_torch_dataset(*args, **kwargs):
    """
    Erzeugt eine echte torch.utils.data.Dataset-Instanz (Mixin zur Laufzeit),
    damit CXRDataset mit DataLoader funktioniert.
    """
    from torch.utils.data import Dataset as TorchDataset

    class _TorchCXR(CXRDataset, TorchDataset):
        pass

    return _TorchCXR(*args, **kwargs)
