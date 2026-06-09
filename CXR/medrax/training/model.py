"""CXR-Classifier: torchvision-Backbone (ResNet18/EfficientNet-B0, optional timm) + eigener Head.
Multi-Label-Logits + Grad-CAM-Ziel-Layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .labels import NUM_CLASSES, VINBIGDATA_CLASSES

SUPPORTED_TV_BACKBONES = {"resnet18", "resnet34", "resnet50", "efficientnet_b0"}


class CXRClassifier(nn.Module):
    """Multi-Label-CXR-Classifier mit torchvision-Backbone und eigenem Head."""

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self._use_timm = False
        self._target_layer = None

        if backbone in SUPPORTED_TV_BACKBONES:
            self._build_torchvision(backbone, num_classes, pretrained, dropout)
        else:
            # Versuch: timm-Backbone (z. B. convnext_tiny)
            self._build_timm(backbone, num_classes, pretrained)

    # ------------------------------------------------------------------ TV
    def _build_torchvision(self, backbone, num_classes, pretrained, dropout):
        import torchvision

        weights = "DEFAULT" if pretrained else None
        try:
            net = getattr(torchvision.models, backbone)(weights=weights)
        except Exception as e:
            print(
                f"[model] Konnte vortrainierte Gewichte für '{backbone}' nicht laden "
                f"({e}). Nutze zufällige Initialisierung."
            )
            net = getattr(torchvision.models, backbone)(weights=None)

        if backbone.startswith("resnet"):
            in_feat = net.fc.in_features
            # eigener Head
            net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
            self.net = net
            self._target_layer = net.layer4[-1]
        elif backbone.startswith("efficientnet"):
            in_feat = net.classifier[1].in_features
            net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
            self.net = net
            self._target_layer = net.features[-1]
        else:  # pragma: no cover
            raise ValueError(f"Backbone nicht unterstützt: {backbone}")

    # ---------------------------------------------------------------- timm
    def _build_timm(self, backbone, num_classes, pretrained):
        try:
            import timm
        except Exception as e:
            raise ValueError(
                f"Backbone '{backbone}' ist kein torchvision-Backbone und timm ist "
                f"nicht installiert ({e}).\n"
                f"-> Unterstützte torchvision-Backbones: {sorted(SUPPORTED_TV_BACKBONES)}\n"
                f"-> oder:  pip install timm"
            )
        self._use_timm = True
        self.net = timm.create_model(
            backbone, pretrained=pretrained, num_classes=num_classes, in_chans=3
        )
        # Bestes-Effort-Ziel-Layer für Grad-CAM
        try:
            self._target_layer = self.net.feature_info  # nur Marker; echtes Layer unten
        except Exception:
            self._target_layer = None
        # Häufige Fälle abdecken
        for attr in ("stages", "blocks", "features"):
            if hasattr(self.net, attr):
                mod = getattr(self.net, attr)
                try:
                    self._target_layer = mod[-1]
                except Exception:
                    self._target_layer = mod
                break

    # -------------------------------------------------------------- forward
    def forward(self, x):
        return self.net(x)

    @property
    def target_layer(self):
        """Conv-Layer für Grad-CAM."""
        return self._target_layer


def build_model(
    backbone: str = "resnet18",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    device: Optional[str] = None,
) -> CXRClassifier:
    """Erzeugt das Modell und verschiebt es auf das Zielgerät."""
    model = CXRClassifier(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
    if device:
        model = model.to(device)
    return model


def get_gradcam_target_layer(model: nn.Module):
    """Liefert das Grad-CAM-Ziel-Layer eines CXRClassifier."""
    if isinstance(model, CXRClassifier):
        return model.target_layer
    return getattr(model, "_target_layer", None)


# ----------------------------------------------------------------- Checkpoints
def save_checkpoint(
    path,
    model: CXRClassifier,
    optimizer=None,
    scaler=None,
    epoch: int = 0,
    best_metric: float = 0.0,
    extra: Optional[dict] = None,
) -> str:
    """Speichert einen Checkpoint inkl. Metadaten (für Resume + Tool-Inferenz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "backbone": getattr(model, "backbone_name", "resnet18"),
        "num_classes": getattr(model, "num_classes", NUM_CLASSES),
        "classes": list(VINBIGDATA_CLASSES),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, str(path))
    return str(path)


def load_checkpoint(
    path,
    device: str = "cpu",
    build_if_needed: bool = True,
):
    """
    Lädt einen Checkpoint. Gibt (model, ckpt_dict) zurück.
    Baut das Modell anhand der gespeicherten Metadaten neu auf.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint nicht gefunden: {path}\n"
            "-> Trainiere zuerst (medrax.training.train_classifier) oder erzeuge einen "
            "Mini-Checkpoint (scripts/test_classifier_10_images.py)."
        )
    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    if not build_if_needed:
        return None, ckpt

    backbone = ckpt.get("backbone", "resnet18")
    num_classes = ckpt.get("num_classes", NUM_CLASSES)
    model = build_model(backbone=backbone, num_classes=num_classes, pretrained=False, device=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
