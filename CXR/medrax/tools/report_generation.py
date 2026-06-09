"""ReportGenerationTool – deaktivierter Lazy-Stub (Report-Generierung, nicht ausgebaut).
Die Textzusammenfassung übernimmt im Core der Qwen-Agent.
"""

from __future__ import annotations

from .base import BaseCXRTool, BaseModel, Field, failed


class ReportInput(BaseModel):
    image_path: str = Field(..., description="Pfad zu einem CXR-Bild (PNG/JPG).")


class ReportGenerationTool(BaseCXRTool):
    name = "report_generation"
    description = (
        "[DEAKTIVIERT] Dedizierte CXR-Report-Generierung. Nicht im Core. Zum Aktivieren: "
        "transformers + Report-Modellgewichte installieren und diese Klasse implementieren. "
        "Im Core übernimmt die Textzusammenfassung der Qwen-Agent."
    )
    args_schema = ReportInput
    enabled = False

    def _run(self, image_path: str):
        return failed(
            "ReportGenerationTool ist deaktiviert (Lazy-Stub). Zum Aktivieren: transformers + "
            "Report-Gewichte installieren und medrax/tools/report_generation.py implementieren.",
            image_path=image_path, tool="report_generation", enabled=False,
        )
