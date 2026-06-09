"""
Test der neuen Anatomie-/Orientierungs-Tools:

  1. XRVAnatomySegmenterTool   – TorchXRayVision Anatomie-Segmenter (Lunge/Herz/Mediastinum)
                                  + grobe Lokalisation der Grad-CAM-Aktivierung
  2. ChestXrayBasicTool        – ianpan/chest-x-ray-basic (View AP/PA/lateral, Lunge re/li, Herz, CTR)
  3. SupportDeviceClassifierTool – CheXpert-DenseNet, nur 'Support Devices' (Ja/Nein + Confidence)

Beim ersten Lauf werden Modellgewichte heruntergeladen (kann etwas dauern).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from medrax.utils.image_utils import list_images
from medrax.utils.paths import find_classifier_checkpoint


def main():
    print("=" * 70)
    print(" TEST: Anatomie-/Orientierungs-Tools (XRV, ianpan, Support-Device)")
    print("=" * 70)
    config.ensure_dirs()

    imgs = list_images(config.TEST_DIR, limit=1)
    if not imgs:
        print(f"[FEHLER] Keine Bilder in {config.TEST_DIR}.")
        return 1

    # DICOM -> PNG (die Tools arbeiten am besten mit PNG)
    from medrax.tools.dicom import DicomProcessorTool

    out, meta = DicomProcessorTool()._run(dicom_path=str(imgs[0]))
    png = out.get("image_path", str(imgs[0]))
    print("Testbild (PNG):", os.path.basename(png), "\n")

    # Checkpoint für die Grad-CAM (XRV-Lokalisation)
    if find_classifier_checkpoint() is None:
        print("[INFO] Kein Checkpoint – erzeuge Mini-Checkpoint (DUMMY) ...")
        from medrax.training.train_classifier import main as train_main
        train_main(["--max-samples", "8", "--epochs", "1", "--batch-size", "2",
                    "--no-pretrained", "--img-size", "128"])

    from medrax.tools.my_classifier_tool import MyChestXRayClassifierTool
    clf = MyChestXRayClassifierTool(img_size=128)
    res = clf.analyze(png)
    print("Top-Pathologie:", res["top_pathology"], "\n")

    # 1) XRV Anatomie-Segmenter + Lokalisation
    print("--- 1) XRVAnatomySegmenterTool ---")
    from medrax.tools.anatomy_segmentation import XRVAnatomySegmenterTool
    out, meta = XRVAnatomySegmenterTool()._run(
        image_path=png, heatmap=res["heatmap"],
        top_prob=res["top_pathology"]["prob"], top_name=res["top_pathology"]["name"])
    if meta.get("analysis_status") == "completed":
        print("Regionen      :", out["regions"])
        print("Zonen         :", out["zones_available"])
        loc = out.get("localization") or {}
        print("Lokalisation  :", loc.get("localized"), "| Zone:", loc.get("zone"),
              "| Grund:", loc.get("reason"))
    else:
        print("[INFO]", out.get("error"))

    # 2) ianpan View/Anatomie
    print("\n--- 2) ChestXrayBasicTool (ianpan) ---")
    from medrax.tools.view_anatomy import ChestXrayBasicTool
    out, meta = ChestXrayBasicTool()._run(image_path=png)
    if meta.get("analysis_status") == "completed":
        print("View          :", out.get("view"))
        print("Lunge re/li   :", out.get("right_lung"), "/", out.get("left_lung"),
              "| Herz:", out.get("heart"), "| CTR:", out.get("cardiothoracic_ratio"))
    else:
        print("[INFO]", out.get("error"))

    # 3) Support-Device
    print("\n--- 3) SupportDeviceClassifierTool ---")
    from medrax.tools.support_devices import SupportDeviceClassifierTool
    out, meta = SupportDeviceClassifierTool()._run(image_path=png)
    if meta.get("analysis_status") == "completed":
        print("Support-Device:", out.get("present"), "| Confidence:", out.get("confidence"))
    else:
        print("[INFO]", out.get("error"))

    print("\n[PASS] Anatomie-/Orientierungs-Tools durchgelaufen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
