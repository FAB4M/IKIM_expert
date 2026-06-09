"""
Smoke-Test des eigenen Classifiers mit (mindestens) 10 Bildern.

Prüft technisch:
  * Dataset lädt >= 10 Bilder (lokal: 108 DICOMs in data/VinBigData/test)
  * Labels werden geladen (lokal DUMMY, da Testbilder unlabeled sind)
  * Mini-Training (1 Epoche) läuft und speichert einen Checkpoint
  * eine Vorhersage läuft durch (inkl. Grad-CAM)

WICHTIG: Lokal werden DUMMY-Labels genutzt (Testbilder haben keine echten Labels).
Das ist KEIN echtes Training – nur ein technischer Smoke-Test.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from medrax.utils.image_utils import list_images


def main():
    print("=" * 70)
    print(" SMOKE-TEST: Classifier mit 10 Bildern (DUMMY-Labels, technisch)")
    print("=" * 70)
    config.ensure_dirs()
    print(config.summary())

    imgs = list_images(config.TEST_DIR)
    print(f"\nGefundene Bilder in {config.TEST_DIR}: {len(imgs)}")
    if len(imgs) < 10:
        print(
            "\n[FEHLER] Es werden mindestens 10 Bilder erwartet.\n"
            "Erwartete Struktur (lokal):\n"
            f"  {config.TEST_DIR}\\<image_id>.dicom  (>= 10 Dateien)\n"
            "Lege dort einige CXR-DICOM/PNG-Dateien ab und starte erneut."
        )
        return 1

    # 1) Mini-Training (nutzt automatischen DUMMY-Fallback)
    print("\n--- 1) Mini-Training (10 Bilder, 1 Epoche, resnet18@128, CPU) ---")
    from medrax.training.train_classifier import main as train_main

    best = train_main([
        "--max-samples", "10", "--epochs", "1", "--batch-size", "2",
        "--no-pretrained", "--img-size", "128",
    ])

    # 2) Checkpoint vorhanden?
    print("\n--- 2) Checkpoint-Prüfung ---")
    if not (config.BEST_CKPT.exists() and config.LAST_CKPT.exists()):
        print("[FEHLER] Checkpoint(s) wurden nicht gespeichert.")
        return 1
    print(f"OK: {config.BEST_CKPT.name}, {config.LAST_CKPT.name}")

    # 3) Vorhersage über das Tool
    print("\n--- 3) Vorhersage (Classifier-Tool + Grad-CAM) ---")
    from medrax.tools.my_classifier_tool import MyChestXRayClassifierTool

    tool = MyChestXRayClassifierTool(img_size=128)
    res = tool.analyze(str(imgs[0]))
    print("Bild:", os.path.basename(res["input_image"]))
    print("Top-3:", res["top"][:3])
    print("Grad-CAM:", res["heatmap_path"])

    print("\n[PASS] Classifier-Smoke-Test erfolgreich (technisch).")
    print("       Hinweis: Mit DUMMY-Labels sind die Werte medizinisch bedeutungslos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
