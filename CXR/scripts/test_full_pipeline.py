"""
End-to-End-Test der gesamten Pipeline.

Ablauf:
  config laden -> Checkpoint sicherstellen (sonst Mini-Checkpoint, DUMMY) ->
  1 Bild analysieren (Classifier + Grad-CAM) -> Anatomie-Lokalisation (TorchXRayVision) ->
  Qwen-Fließtext (oder strukturierter Fallback bei 'LLM unreachable').
Ergebnis wird als JSON unter outputs/predictions gespeichert.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from medrax.utils.image_utils import list_images
from medrax.utils.paths import find_classifier_checkpoint


def main():
    print("=" * 70)
    print(" END-TO-END PIPELINE-TEST")
    print("=" * 70)
    config.ensure_dirs()
    print(config.summary(), "\n")

    imgs = list_images(config.TEST_DIR, limit=1)
    if not imgs:
        print(f"[FEHLER] Keine Bilder in {config.TEST_DIR}. Lege CXR-DICOM/PNG dort ab.")
        return 1
    img = str(imgs[0])

    # Checkpoint sicherstellen
    if find_classifier_checkpoint() is None:
        print("[INFO] Kein Checkpoint gefunden – erzeuge Mini-Checkpoint (DUMMY) ...")
        from medrax.training.train_classifier import main as train_main
        train_main(["--max-samples", "8", "--epochs", "1", "--batch-size", "2",
                    "--no-pretrained", "--img-size", "128"])

    # Agent
    from medrax.agent.initialize_agent import initialize_agent

    agent = initialize_agent(verbose=True)
    print("\n[Pipeline] Analysiere:", os.path.basename(img))
    res = agent.analyze_image(img, question="Was siehst du auf diesem Röntgenbild?")

    print("\n--- ERGEBNIS ---")
    print("pipeline ok :", res.get("ok"))
    print("LLM genutzt :", res.get("llm_ok"), "(False = Ollama nicht erreichbar -> Fallback)")
    view = res.get("view") or {}
    if view:
        print("View/Anatomie:", view.get("view"), "| Lunge re/li:", view.get("right_lung"),
              "/", view.get("left_lung"), "| Herz:", view.get("heart"),
              "| CTR:", view.get("cardiothoracic_ratio"))
    loc = res.get("localization") or {}
    print("Lokalisation:", loc.get("localized"), "| Methode:", loc.get("method"),
          "| Zone:", loc.get("zone"))
    sup = res.get("support_device") or {}
    if sup:
        print("Support-Dev :", sup.get("present"), "| Confidence:", sup.get("confidence"))
    print("Grad-CAM    :", res.get("heatmap_path"))
    print("\n--- ANTWORT (Radiologen-Stil bzw. Fallback) ---")
    print(res.get("answer"))

    # JSON speichern (heatmap-Array nicht serialisierbar -> raus)
    out = {k: v for k, v in res.items() if k != "llm_error"}
    out_path = config.PRED_DIR / "full_pipeline_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nErgebnis gespeichert: {out_path}")

    print("\n[PASS] Pipeline durchgelaufen.")
    if not res.get("llm_ok"):
        print("       Hinweis: Für die Fließtext-Antwort Ollama starten "
              f"(ollama serve; ollama pull {config.OPENAI_MODEL}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
