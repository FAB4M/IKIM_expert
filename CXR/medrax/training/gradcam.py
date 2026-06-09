"""
Abhängigkeitsfreies Grad-CAM (nur PyTorch).

Erzeugt für eine gewählte Klasse eine Heatmap (HxW, 0..1), die zeigt, welche
Bildregionen die Modellaktivierung am stärksten treiben. Wird vom Classifier-Tool
für die grobe Lokalisation (zusammen mit den TorchXRayVision-Anatomiezonen) genutzt.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Klassisches Grad-CAM über Forward-/Backward-Hooks am Ziel-Conv-Layer."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module]):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._handles = []
        if target_layer is not None:
            self._handles.append(target_layer.register_forward_hook(self._save_activation))
            # full backward hook (PyTorch >= 1.8)
            try:
                self._handles.append(
                    target_layer.register_full_backward_hook(self._save_gradient)
                )
            except Exception:  # pragma: no cover
                self._handles.append(
                    target_layer.register_backward_hook(self._save_gradient)
                )

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        """
        input_tensor: [1,3,H,W]
        class_idx: Zielklasse (None -> argmax der Sigmoid-Wahrscheinlichkeiten)
        Rückgabe: (heatmap HxW float 0..1, class_idx)
        """
        if self.target_layer is None:
            raise RuntimeError(
                "Kein Grad-CAM-Ziel-Layer verfügbar (Backbone nicht unterstützt). "
                "Nutze resnet18/efficientnet_b0."
            )

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)  # [1, C]
        probs = torch.sigmoid(logits)
        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        acts = self._activations  # [1, K, h, w]
        grads = self._gradients   # [1, K, h, w]
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM: Aktivierungen/Gradienten nicht erfasst.")

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, K, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        # auf Eingabegröße hochskalieren
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy().astype(np.float32)

        cam -= cam.min()
        denom = cam.max() + 1e-8
        cam /= denom
        return cam, class_idx


def compute_gradcam(model, target_layer, input_tensor, class_idx=None):
    """Komfort-Wrapper: erzeugt eine Grad-CAM-Heatmap und räumt Hooks auf."""
    cam = GradCAM(model, target_layer)
    try:
        heatmap, idx = cam(input_tensor, class_idx)
    finally:
        cam.remove()
    return heatmap, idx
