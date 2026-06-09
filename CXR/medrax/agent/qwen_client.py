"""Qwen-LLM-Client (OpenAI-kompatibel, via requests). Liest OPENAI_* aus config.
Fehlender Server -> klare Meldung statt Crash.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """Entfernt <think>...</think>-Blöcke (qwen3 'thinking')."""
    return _THINK_RE.sub("", text or "").strip()


class QwenClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or config.OPENAI_BASE_URL).rstrip("/")
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT

    # -------------------------------------------------------------- helpers
    @property
    def _chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _unreachable(self, detail: str) -> Dict:
        return {
            "ok": False,
            "error": "LLM unreachable / Ollama server not running",
            "detail": detail,
            "hint": (
                f"Server unter {self.base_url} nicht erreichbar. Starte Ollama:\n"
                f"    ollama serve\n    ollama pull {self.model}\n"
                "oder passe OPENAI_BASE_URL/OPENAI_MODEL in .env an."
            ),
        }

    # ----------------------------------------------------------------- chat
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 800,
        strip_think: bool = True,
    ) -> Dict:
        """Sendet eine Chat-Anfrage. Rückgabe: {ok, content} oder {ok:False, error,...}."""
        import requests

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            r = requests.post(self._chat_url, json=payload, headers=headers, timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            return self._unreachable(f"ConnectionError: {e}")
        except requests.exceptions.Timeout as e:
            return self._unreachable(f"Timeout nach {self.timeout}s: {e}")
        except Exception as e:
            return self._unreachable(f"Unerwarteter Fehler: {e}")

        if r.status_code != 200:
            # Modell evtl. nicht gezogen
            return {
                "ok": False,
                "error": f"HTTP {r.status_code}",
                "detail": r.text[:400],
                "hint": f"Ist das Modell '{self.model}' vorhanden? -> ollama pull {self.model}",
            }

        try:
            data = r.json()
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            return {"ok": False, "error": "Antwort nicht lesbar", "detail": f"{e}: {r.text[:300]}"}

        if strip_think:
            content = _strip_think(content)
        return {"ok": True, "content": content, "model": self.model}

    # ----------------------------------------------------------------- ping
    def ping(self) -> Dict:
        """Kurzer Erreichbarkeits-Test."""
        res = self.chat(
            [{"role": "user", "content": "Antworte nur mit OK, wenn du erreichbar bist."}],
            temperature=0.0,
            max_tokens=16,
        )
        if res.get("ok"):
            res["reachable"] = True
        return res

    def simple(self, system: str, user: str, **kw) -> Dict:
        """Komfort: System+User -> Antwort."""
        return self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], **kw
        )


if __name__ == "__main__":
    client = QwenClient()
    print(f"Teste {client.model} @ {client.base_url} ...")
    res = client.ping()
    if res.get("ok"):
        print("OK – Antwort:", res["content"])
    else:
        print("Nicht erreichbar:", res.get("error"))
        print(res.get("hint", ""))
