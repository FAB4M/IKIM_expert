"""
Evaluation des eigenen Classifiers auf einem Datensatz.

Beispiel:
    python -m medrax.training.evaluate_classifier --checkpoint weights/classifier/best_model.pt \
        --max-samples 50

Schreibt Metriken nach outputs/predictions/ (JSON + per-Klasse-CSV) und
per-Bild-Vorhersagen als JSON. Bei wenigen/ungelabelten Bildern: klare Hinweise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config
from medrax.training.dataset import build_samples, make_torch_dataset
from medrax.training.labels import VINBIGDATA_CLASSES
from medrax.training.metrics import compute_metrics, format_metrics, save_metrics
from medrax.training.model import load_checkpoint
from medrax.utils.image_utils import build_eval_transform
from medrax.utils.logging_utils import get_logger
from medrax.utils.paths import find_classifier_checkpoint

log = get_logger("evaluate")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate VinBigData-15 CXR classifier")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--image-dir", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--dummy-labels", action="store_true")
    p.add_argument("--out-json", type=str, default=str(config.PRED_DIR / "eval_metrics.json"))
    p.add_argument("--out-csv", type=str, default=str(config.PRED_DIR / "eval_per_class.csv"))
    p.add_argument("--pred-json", type=str, default=str(config.PRED_DIR / "eval_predictions.json"))
    return p.parse_args(argv)


def _resolve_image_dir(args):
    from medrax.utils.image_utils import list_images

    if args.image_dir:
        return Path(args.image_dir)
    train_dir = Path(config.TRAIN_DIR)
    if train_dir.exists() and list_images(train_dir, limit=1):
        return train_dir
    return Path(config.TEST_DIR)


def main(argv=None):
    import torch
    from torch.utils.data import DataLoader

    args = parse_args(argv)
    config.ensure_dirs()
    device = args.device or config.get_device()

    ckpt_path = args.checkpoint or find_classifier_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError(
            "Kein Checkpoint gefunden. Trainiere zuerst oder gib --checkpoint an.\n"
            "z. B. python -m medrax.training.train_classifier --max-samples 10 --epochs 1 --batch-size 2"
        )
    log.info("Lade Checkpoint: %s", ckpt_path)
    model, ckpt = load_checkpoint(ckpt_path, device=device)

    image_dir = _resolve_image_dir(args)
    csv = Path(args.csv) if args.csv else (
        config.TRAIN_CSV if Path(config.TRAIN_CSV).exists() else None
    )
    log.info("image_dir=%s | csv=%s", image_dir, csv)

    try:
        samples = build_samples(csv_path=csv, image_dir=image_dir,
                                max_samples=args.max_samples, dummy_labels=args.dummy_labels)
        labeled = not args.dummy_labels and csv is not None
    except FileNotFoundError as e:
        log.warning("Keine gelabelten Bilder: %s -> DUMMY-Modus (Metriken bedeutungslos).", e)
        samples = build_samples(csv_path=None, image_dir=image_dir,
                                max_samples=args.max_samples, dummy_labels=True)
        labeled = False

    ds = make_torch_dataset(samples, transform=build_eval_transform(args.img_size),
                            img_size=args.img_size, cache_dir=config.CACHE_DIR)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_t, all_p = [], []
    model.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            all_p.append(probs)
            all_t.append(targets.numpy())
    probs = np.concatenate(all_p, axis=0)
    targets = np.concatenate(all_t, axis=0)

    # Per-Bild-Vorhersagen speichern (immer sinnvoll)
    preds = []
    for s, p in zip(samples, probs):
        preds.append({
            "image_id": s.get("image_id"),
            "image_path": s["image_path"],
            "predictions": {VINBIGDATA_CLASSES[i]: round(float(p[i]), 4) for i in range(len(p))},
        })
    Path(args.pred_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.pred_json, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    log.info("Per-Bild-Vorhersagen: %s", args.pred_json)

    if not labeled:
        print("\n[evaluate] Hinweis: Keine echten Labels vorhanden (DUMMY/ungelabelt).")
        print("           Es werden nur Vorhersagen gespeichert, KEINE aussagekräftigen Metriken.")
        return

    metrics = compute_metrics(targets, probs, VINBIGDATA_CLASSES, threshold=args.threshold)
    print("\n" + format_metrics(metrics))
    save_metrics(metrics, args.out_json, args.out_csv)
    log.info("Metriken: %s | %s", args.out_json, args.out_csv)


if __name__ == "__main__":
    main()
