"""
Agent auf EINEM CXR-Bild ausführen (PNG/JPG/DICOM).

Beispiele:
    python scripts/run_agent.py --image /content/train_png/<id>.png
    python scripts/run_agent.py --image data/VinBigData/test/<id>.dicom \
        --question "Was siehst du auf diesem Röntgenbild?"

Ablauf: (DICOM->PNG) -> eigener Classifier + Grad-CAM -> Anatomie-Lokalisation (XRV)
        -> View (ianpan) -> Support-Device -> Qwen-Fließtext (oder strukturierter Fallback).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Pfad zum CXR (PNG/JPG/DICOM)")
    ap.add_argument("--question", default="Was siehst du auf diesem Röntgenbild?")
    ap.add_argument("--tools", nargs="*", default=None,
                    help="Optionale Tool-Auswahl (Default: alle Core-Tools)")
    ap.add_argument("--save-json", default=str(config.PRED_DIR / "agent_result.json"))
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"[FEHLER] Bild nicht gefunden: {args.image}")
        return 1

    from medrax.agent.initialize_agent import initialize_agent

    agent = initialize_agent(selected_tools=args.tools)
    print(f"\n[run_agent] Analysiere: {os.path.basename(args.image)}\n")
    res = agent.analyze_image(args.image, question=args.question)

    print("=" * 70)
    print(" ANTWORT (Radiologen-Stil bzw. strukturierter Fallback)")
    print("=" * 70)
    print(res.get("answer"))

    print("\n----- Strukturierte Signale -----")
    print("LLM genutzt :", res.get("llm_ok"), "(False = kein LLM erreichbar -> Fallback-Text)")
    print("Top         :", res.get("top"))
    view = res.get("view") or {}
    if view:
        print("View        :", view.get("view"), "| CTR:", view.get("cardiothoracic_ratio"))
    loc = res.get("localization") or {}
    print("Lokalisation:", loc.get("localized"), "| Zone:", loc.get("zone"),
          "| Methode:", loc.get("method"))
    sup = res.get("support_device") or {}
    if sup:
        print("Support-Dev :", sup.get("present"), "| Confidence:", sup.get("confidence"))
    print("Grad-CAM    :", res.get("heatmap_path"))

    out = {k: v for k, v in res.items() if k != "llm_error"}
    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print("\nErgebnis gespeichert:", args.save_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
