"""
Wandelt die VinBigData-Detection-CSV (train.csv) in eine image-level
Multi-Label-CSV um (image_id + eine 0/1-Spalte je Pathologie).

Nützlich fürs (Colab-)Training und zum Inspizieren der Labelverteilung.

Beispiel:
    python scripts/build_vinbigdata_labels.py
    python scripts/build_vinbigdata_labels.py --csv data/VinBigData/train.csv \
        --out data/processed/vinbig_multilabel.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from medrax.training.dataset import build_image_level_labels
from medrax.training.labels import VINBIGDATA_CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.TRAIN_CSV))
    ap.add_argument("--out", default=str(config.PROCESSED_DIR / "vinbig_multilabel.csv"))
    args = ap.parse_args()

    from pathlib import Path

    import numpy as np
    import pandas as pd

    if not Path(args.csv).exists():
        print(f"[FEHLER] CSV nicht gefunden: {args.csv}")
        return 1

    print(f"Lese {args.csv} ...")
    label_df = build_image_level_labels(args.csv, VINBIGDATA_CLASSES)
    mat = np.stack(label_df["target"].values)
    wide = pd.DataFrame(mat.astype(int), columns=VINBIGDATA_CLASSES)
    wide.insert(0, "image_id", label_df["image_id"].values)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.out, index=False)
    print(f"Geschrieben: {args.out}  ({len(wide)} Bilder)")

    print("\nLabel-Verteilung (Anzahl positiver Bilder je Klasse):")
    counts = wide[VINBIGDATA_CLASSES].sum().sort_values(ascending=False)
    for name, c in counts.items():
        print(f"  {name:<22} {int(c)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
