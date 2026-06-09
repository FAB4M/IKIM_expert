"""ImageGenerationTool – deaktivierter Lazy-Stub (CXR-Bildgenerierung, nicht ausgebaut)."""

from __future__ import annotations

from .base import BaseCXRTool, BaseModel, Field, failed


class GenerationInput(BaseModel):
    prompt: str = Field("", description="Textbeschreibung des zu generierenden CXR.")


class ImageGenerationTool(BaseCXRTool):
    name = "image_generation"
    description = (
        "[DEAKTIVIERT] Synthetische CXR-Bildgenerierung. Nicht im Core. Zum Aktivieren: "
        "diffusers/transformers + Modellgewichte installieren und diese Klasse implementieren."
    )
    args_schema = GenerationInput
    enabled = False

    def _run(self, prompt: str = ""):
        return failed(
            "ImageGenerationTool ist deaktiviert (Lazy-Stub). Zum Aktivieren: diffusers/transformers "
            "+ Gewichte installieren und medrax/tools/optional_generation.py implementieren.",
            tool="image_generation", enabled=False,
        )
