"""DicomProcessorTool – DICOM -> PNG (via medrax.utils.dicom_utils)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from .base import BaseCXRTool, BaseModel, Field, failed, make_metadata


class DicomProcessorInput(BaseModel):
    dicom_path: str = Field(..., description="Pfad zur DICOM-Datei (.dcm/.dicom).")
    window_center: Optional[float] = Field(
        None, description="Optionales Window-Center (sonst aus DICOM/Min-Max)."
    )
    window_width: Optional[float] = Field(
        None, description="Optionale Window-Width (sonst aus DICOM/Min-Max)."
    )


class DicomProcessorTool(BaseCXRTool):
    name = "dicom_processor"
    description = (
        "Konvertiert eine DICOM-Röntgendatei (.dcm/.dicom) in ein Standard-PNG für die "
        "weitere Analyse. NUTZEN, wenn der Eingabepfad eine DICOM-Datei ist (Classifier/"
        "Segmentierung erwarten PNG/JPG). Eingabe: dicom_path. Ausgabe: output['image_path'] "
        "= Pfad zum erzeugten PNG. Wendet Rescale, Window/Level, MONOCHROME1-Invertierung "
        "und Kontrastverstärkung (CLAHE) an."
    )
    args_schema = DicomProcessorInput

    def __init__(self, output_dir: Optional[str] = None, apply_clahe: bool = True):
        import config

        self.output_dir = Path(output_dir) if output_dir else Path(config.CACHE_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.apply_clahe = apply_clahe

    def _run(self, dicom_path: str, window_center: Optional[float] = None,
             window_width: Optional[float] = None):
        from medrax.utils.dicom_utils import DicomReadError, dicom_metadata, read_dicom_pil

        p = Path(dicom_path)
        if not p.exists():
            return failed(f"DICOM-Datei nicht gefunden: {dicom_path}", dicom_path=dicom_path)

        try:
            img = read_dicom_pil(
                p, window_center=window_center, window_width=window_width,
                apply_clahe=self.apply_clahe,
            )
            out_path = self.output_dir / f"processed_dicom_{uuid.uuid4().hex[:8]}.png"
            img.save(str(out_path))
            meta = make_metadata(
                "completed", dicom_path=dicom_path, image_path=str(out_path),
                **dicom_metadata(p),
            )
            return {"image_path": str(out_path)}, meta
        except DicomReadError as e:
            return failed(str(e), dicom_path=dicom_path)
        except Exception as e:
            return failed(f"DICOM-Verarbeitung fehlgeschlagen: {e}", dicom_path=dicom_path)
