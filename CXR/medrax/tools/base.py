"""Basis-Tool-Schnittstelle: name, description, args_schema, _run -> (output, metadata).

LangChain-kompatibel, aber ohne harte LangChain-Abhängigkeit (to_langchain_tool()).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# pydantic ist optional – wird für args_schema / LangChain genutzt, falls vorhanden.
try:
    from pydantic import BaseModel, Field  # type: ignore

    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover
    _HAS_PYDANTIC = False

    class BaseModel:  # minimaler Fallback
        pass

    def Field(default=None, **kwargs):  # type: ignore
        return default


class BaseCXRTool:
    """Basisklasse aller Tools."""

    name: str = "base_cxr_tool"
    description: str = "Basis-Tool (bitte überschreiben)."
    args_schema: Optional[type] = None

    def _run(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError

    # bequemer Aufruf
    def __call__(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self._run(**kwargs)

    def run(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self._run(**kwargs)

    # ---------------------------------------------------------- LangChain
    def to_langchain_tool(self):
        """
        Baut – falls langchain installiert ist – ein StructuredTool/Tool, das
        diese Klasse kapselt. Sonst verständlicher Hinweis.
        """
        try:
            from langchain_core.tools import StructuredTool
        except Exception:
            try:
                from langchain.tools import StructuredTool  # ältere Versionen
            except Exception as e:
                raise RuntimeError(
                    "LangChain ist nicht installiert. Für den LangChain-Adapter:\n"
                    "    pip install langchain langchain-core\n"
                    f"(Originalfehler: {e})"
                )

        def _func(**kwargs):
            output, metadata = self._run(**kwargs)
            return {"output": output, "metadata": metadata}

        kwargs = dict(name=self.name, description=self.description, func=_func)
        if self.args_schema is not None and _HAS_PYDANTIC:
            kwargs["args_schema"] = self.args_schema
        return StructuredTool.from_function(**kwargs)


def make_metadata(status: str = "completed", **kwargs) -> Dict[str, Any]:
    """Hilfsfunktion für einheitliche Metadaten."""
    md = {"analysis_status": status}
    md.update(kwargs)
    return md


def failed(error: str, **meta) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Einheitliche Fehlerrückgabe (output, metadata)."""
    return {"error": error}, make_metadata("failed", error=error, **meta)
