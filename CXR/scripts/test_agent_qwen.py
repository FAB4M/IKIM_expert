"""
Test der Qwen/Ollama-Erreichbarkeit.

Sendet eine kurze Frage und prüft, ob eine Antwort kommt. Ein NICHT laufender
Server ist KEIN harter Fehler – es wird eine klare Meldung ausgegeben.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from medrax.agent.qwen_client import QwenClient


def main():
    print("=" * 70)
    print(" TEST: Qwen / Ollama Erreichbarkeit")
    print("=" * 70)
    print(f"OPENAI_BASE_URL : {config.OPENAI_BASE_URL}")
    print(f"OPENAI_MODEL    : {config.OPENAI_MODEL}")
    print(f"OPENAI_API_KEY  : {config.OPENAI_API_KEY[:6]}...\n")

    client = QwenClient()
    res = client.ping()

    if res.get("ok"):
        print("[PASS] Qwen erreichbar. Antwort:")
        print("   ", res["content"].replace("\n", " ")[:200])
        return 0

    print("[INFO] Qwen/Ollama NICHT erreichbar (das ist ok, wenn der Server nicht läuft).")
    print("   Fehler:", res.get("error"))
    print("   Detail:", str(res.get("detail"))[:200])
    print("\n" + (res.get("hint") or ""))
    print("\nSo startest du Ollama lokal:")
    print("    1) https://ollama.com installieren")
    print("    2) ollama serve")
    print(f"    3) ollama pull {config.OPENAI_MODEL}")
    print(f"    4) erneut: python scripts/test_agent_qwen.py")
    # Kein harter Fehler -> Exit 0, damit Pipelines nicht abbrechen.
    return 0


if __name__ == "__main__":
    sys.exit(main())
