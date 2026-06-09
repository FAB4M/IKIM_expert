"""LlavaMedVQATool – deaktivierter Lazy-Stub (VQA, nicht ausgebaut)."""

from __future__ import annotations

from .base import BaseCXRTool, BaseModel, Field, failed


class VQAInput(BaseModel):
    image_path: str = Field(..., description="Pfad zu einem CXR-Bild (PNG/JPG).")
    question: str = Field("", description="Frage zum Bild.")


class LlavaMedVQATool(BaseCXRTool):
    name = "llava_med_vqa"
    description = (
        "[DEAKTIVIERT] Visual Question Answering für CXR (LLaVA-Med/CheXagent). Nicht im Core. "
        "Zum Aktivieren: transformers + accelerate installieren, Modellgewichte laden und diese "
        "Klasse implementieren. Hinweis: Der Qwen-Agent kann währenddessen Textfragen anhand der "
        "Classifier-/Segmentierungs-Ergebnisse beantworten."
    )
    args_schema = VQAInput
    enabled = False

    def _run(self, image_path: str, question: str = ""):
        return failed(
            "LlavaMedVQATool ist deaktiviert (Lazy-Stub). Zum Aktivieren: transformers/accelerate "
            "+ LLaVA-Med-Gewichte installieren und medrax/tools/llava_med.py implementieren.",
            image_path=image_path, tool="llava_med_vqa", enabled=False,
        )
