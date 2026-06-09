"""
Training des eigenen VinBigData-15 CXR-Classifiers.

Beispiele:
    # Lokaler Smoke-Test (108 DICOMs ohne Labels -> DUMMY-Modus):
    python -m medrax.training.train_classifier --max-samples 10 --epochs 1 --batch-size 2

    # Echtes Training (Colab, mit train.csv + train/-Bildern):
    python -m medrax.training.train_classifier --epochs 10 --batch-size 16 --backbone resnet18

Speichert weights/classifier/last_model.pt (jede Epoche) und best_model.pt (bester Val-Score).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import config
from medrax.training.dataset import build_samples, make_torch_dataset
from medrax.training.labels import VINBIGDATA_CLASSES
from medrax.training.losses import build_criterion, compute_pos_weight
from medrax.training.metrics import compute_metrics, format_metrics
from medrax.training.model import build_model, load_checkpoint, save_checkpoint
from medrax.utils.image_utils import build_eval_transform, build_train_transform
from medrax.utils.logging_utils import get_logger
from medrax.utils.seed import set_seed

log = get_logger("train")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train VinBigData-15 CXR classifier")
    p.add_argument("--csv", type=str, default=None,
                   help="Label-CSV (Detection oder wide). Default: config.TRAIN_CSV falls vorhanden.")
    p.add_argument("--image-dir", type=str, default=None,
                   help="Bildordner. Default: train/ falls vorhanden, sonst test/.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--backbone", type=str, default=config.BACKBONE)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="cpu|cuda (Default: auto)")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint zum Fortsetzen")
    p.add_argument("--dummy-labels", action="store_true",
                   help="DUMMY-Labels (technischer Smoke-Test)")
    p.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--amp", action="store_true", help="Mixed Precision (nur sinnvoll auf CUDA)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=str(config.CLASSIFIER_CKPT_DIR))
    return p.parse_args(argv)


def _resolve_sources(args):
    """Bestimmt csv + image_dir und entscheidet über DUMMY-Fallback."""
    from medrax.utils.image_utils import list_images

    if args.image_dir:
        image_dir = Path(args.image_dir)
    else:
        train_dir = Path(config.TRAIN_DIR)
        if train_dir.exists() and list_images(train_dir, limit=1):
            image_dir = train_dir
        else:
            image_dir = Path(config.TEST_DIR)

    csv = Path(args.csv) if args.csv else (
        config.TRAIN_CSV if Path(config.TRAIN_CSV).exists() else None
    )
    return csv, image_dir, args.dummy_labels


def build_loaders(args):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    csv, image_dir, dummy = _resolve_sources(args)
    log.info("CSV       = %s", csv)
    log.info("image_dir = %s", image_dir)

    try:
        samples = build_samples(
            csv_path=csv, image_dir=image_dir,
            max_samples=args.max_samples, dummy_labels=dummy,
        )
    except FileNotFoundError as e:
        if not dummy:
            log.warning("Keine gelabelten Bilder auffindbar: %s", e)
            log.warning(">> Fallback auf DUMMY-Modus (rein technischer Smoke-Test).")
            samples = build_samples(
                csv_path=None, image_dir=image_dir,
                max_samples=args.max_samples, dummy_labels=True,
            )
        else:
            raise

    n = len(samples)
    log.info("Geladene Samples: %d", n)
    if n < 4:
        log.warning("Sehr wenige Samples (%d) – Validierung = Training.", n)
        train_s, val_s = samples, samples
    else:
        idx = np.arange(n)
        tr, va = train_test_split(idx, test_size=args.val_split, random_state=args.seed)
        train_s = [samples[i] for i in tr]
        val_s = [samples[i] for i in va]

    train_ds = make_torch_dataset(
        train_s, transform=build_train_transform(args.img_size),
        img_size=args.img_size, cache_dir=config.CACHE_DIR,
    )
    val_ds = make_torch_dataset(
        val_s, transform=build_eval_transform(args.img_size),
        img_size=args.img_size, cache_dir=config.CACHE_DIR,
    )

    pin = (args.device or config.get_device()) == "cuda"  # schnellere Host->GPU-Transfers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    pos_weight = compute_pos_weight(np.stack([s["target"] for s in train_s]))
    return train_loader, val_loader, pos_weight


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    import torch
    from tqdm import tqdm

    model.train()
    running = 0.0
    n = 0
    for imgs, targets in tqdm(loader, desc="train", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        running += float(loss.item()) * imgs.size(0)
        n += imgs.size(0)
    return running / max(n, 1)


def evaluate(model, loader, criterion, device):
    import torch
    from tqdm import tqdm

    model.eval()
    all_t, all_p = [], []
    running, n = 0.0, 0
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="val", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            loss = criterion(logits, targets)
            running += float(loss.item()) * imgs.size(0)
            n += imgs.size(0)
            all_t.append(targets.cpu().numpy())
            all_p.append(torch.sigmoid(logits).cpu().numpy())
    val_loss = running / max(n, 1)
    targets = np.concatenate(all_t, axis=0)
    probs = np.concatenate(all_p, axis=0)
    metrics = compute_metrics(targets, probs, VINBIGDATA_CLASSES)
    return val_loss, metrics


def main(argv=None):
    import torch

    args = parse_args(argv)
    set_seed(args.seed)
    config.ensure_dirs()

    device = args.device or config.get_device()
    log.info("Device: %s | Backbone: %s | img_size: %d", device, args.backbone, args.img_size)

    train_loader, val_loader, pos_weight = build_loaders(args)
    pos_weight = pos_weight.to(device)

    model = build_model(args.backbone, pretrained=args.pretrained, device=device)
    criterion = build_criterion(pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    use_amp = args.amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        log.info("Resume von %s", args.resume)
        _, ckpt = load_checkpoint(args.resume, device=device, build_if_needed=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scaler is not None and "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", 0.0))
        log.info("Fortsetzen ab Epoche %d (best=%.4f)", start_epoch, best_metric)

    out_dir = Path(args.out_dir)
    last_ckpt = out_dir / "last_model.pt"
    best_ckpt = out_dir / "best_model.pt"

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        macro_auc = metrics["macro_auc"]
        log.info(
            "Epoche %d | train_loss=%.4f val_loss=%.4f macro_auc=%s",
            epoch, train_loss, val_loss,
            "nan" if np.isnan(macro_auc) else f"{macro_auc:.4f}",
        )

        # last immer speichern
        save_checkpoint(last_ckpt, model, optimizer, scaler, epoch=epoch, best_metric=best_metric)

        # best nach macro_auc, sonst (falls nan) nach niedrigstem val_loss
        score = macro_auc if not np.isnan(macro_auc) else (1.0 - min(val_loss, 1.0))
        if score >= best_metric:
            best_metric = score
            save_checkpoint(best_ckpt, model, optimizer, scaler, epoch=epoch, best_metric=best_metric)
            log.info("  -> neues bestes Modell gespeichert (%.4f): %s", best_metric, best_ckpt)

    print()
    print(format_metrics(metrics))
    print(f"\nCheckpoints: {last_ckpt}\n             {best_ckpt}")
    return str(best_ckpt)


if __name__ == "__main__":
    main()
