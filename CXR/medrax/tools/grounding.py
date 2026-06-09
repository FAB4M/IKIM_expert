"""GroundingTool – deaktivierter Lazy-Stub (nicht ausgebaut)."""

from __future__ import annotations

from .base import BaseCXRTool, BaseModel, Field, failed


class GroundingInput(BaseModel):
    image_path: str = Field(..., description="Pfad zu einem CXR-Bild (PNG/JPG).")
    prompt: str = Field("", description="Zu lokalisierende Phrase/Region.")


class GroundingTool(BaseCXRTool):
    name = "grounding"
    description = (
        "[DEAKTIVIERT] Phrase-Grounding/Lokalisierung per Box. Nicht im Core enthalten. "
        "Zum Aktivieren Grounding-Gewichte + transformers installieren und "
        "diese Klasse implementieren. Gibt aktuell einen Hinweis zurück."
    )
    args_schema = GroundingInput
    enabled = False

    def _run(self, image_path: str, prompt: str = ""):
        return failed(
            "GroundingTool ist deaktiviert (Lazy-Stub). Zum Aktivieren: Grounding-"
            "Gewichte + transformers installieren und medrax/tools/grounding.py implementieren.",
            image_path=image_path, tool="grounding", enabled=False,
        )
