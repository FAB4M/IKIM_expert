"""
Konvertiert DICOMs eines Ordners als PNG in einen Cache.
Beschleunigt spätere Läufe (keine wiederholte JPEG2000-Dekodierung) und ermöglicht
das Vor-Dekodieren des Trainings-Datensatzes in einem SEPARATEN, einmaligen Schritt.

Beispiele:
    # lokal (108 Testbilder), 512px, 2 Prozesse:
    python scripts/convert_dicom_to_png_cache.py --src data/VinBigData/test \
        --dst data/cache/test_png --size 512 --workers 2

    # Colab (15.000 Trainingsbilder) nach lokalem /content:
    python scripts/convert_dicom_to_png_cache.py --src <train_dir> \
        --dst /content/train_png --size 512 --workers 4

Idempotent: bereits vorhandene PNGs werden übersprungen.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402  (setzt u. a. KMP_DUPLICATE_LIB_OK)
from medrax.utils.dicom_utils import is_dicom, read_dicom_pil
from medrax.utils.image_utils import list_images


def _decode_one(task):
    """Worker: dekodiert ein DICOM -> (optional resized) PNG. Modul-Level für Multiprocessing."""
    src, dst_dir, size, clahe = task
    out = Path(dst_dir) / f"{Path(src).stem}.png"
    if out.exists():
        return ("skip", str(out))
    try:
        img = read_dicom_pil(src, apply_clahe=clahe)
        if size:
            img = img.resize((size, size))
        img.save(str(out))
        return ("ok", str(out))
    except Exception as e:  # einzelnes Bild darf den Lauf nicht abbrechen
        return ("fail", f"{Path(src).name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(config.TEST_DIR), help="Quellordner mit DICOMs")
    ap.add_argument("--dst", default=str(config.CACHE_DIR), help="Zielordner (PNG-Cache)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--size", type=int, default=None,
                    help="Resize auf NxN beim Cachen (z. B. 512). Ohne Angabe: Originalgröße.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallele Decode-Prozesse (CPU-gebundenes JPEG2000-Decoding).")
    ap.add_argument("--no-clahe", action="store_true")
    args = ap.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = list_images(args.src, limit=args.limit)
    dicoms = [f for f in files if is_dicom(f)]
    print(f"DICOMs in {args.src}: {len(dicoms)} | size={args.size} | workers={args.workers}")
    if not dicoms:
        print("[WARN] Keine DICOMs gefunden.")
        return 0

    clahe = not args.no_clahe
    tasks = [(str(f), str(dst), args.size, clahe) for f in dicoms]
    ok = fail = skip = 0
    total = len(tasks)

    def _tally(status, info, i):
        nonlocal ok, fail, skip
        if status == "ok":
            ok += 1
        elif status == "skip":
            skip += 1
        else:
            fail += 1
            print(f"  [WARN] {info}")
        if i % 200 == 0 or i == total:
            print(f"  ... {i}/{total} (ok={ok}, skip={skip}, fail={fail})")

    if args.workers and args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (status, info) in enumerate(ex.map(_decode_one, tasks, chunksize=8), 1):
                _tally(status, info, i)
    else:
        for i, task in enumerate(tasks, 1):
            status, info = _decode_one(task)
            _tally(status, info, i)

    print(f"\nFertig. ok={ok}, skip(bereits vorhanden)={skip}, fail={fail}")
    print(f"Cache: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
