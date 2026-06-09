"""
Tool-Registry mit selektivem Laden.

build_tools(selected_tools=[...]) lädt NUR die angeforderten Tools (damit nicht
immer alle – ggf. großen – Modelle initialisiert werden). Tools werden lazy
importiert; schlägt die Initialisierung eines Tools fehl (z. B. fehlender
Checkpoint), wird es mit klarer Meldung übersprungen, ohne den Rest zu blockieren.

Beispiel:
    from medrax.tools import build_tools
    tools = build_tools(["DicomProcessorTool", "MyChestXRayClassifierTool"])
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Core-Tools (laufen lokal; Anatomie-/Device-Modelle laden ihre Gewichte lazy)
CORE_DEFAULT: List[str] = [
    "DicomProcessorTool",
    "ImageVisualizerTool",
    "MyChestXRayClassifierTool",
    "XRVAnatomySegmenterTool",
    "ChestXrayBasicTool",
    "SupportDeviceClassifierTool",
]

# Optionale, deaktivierte Lazy-Stubs
OPTIONAL_STUBS: List[str] = [
    "GroundingTool",
    "LlavaMedVQATool",
    "ReportGenerationTool",
    "ImageGenerationTool",
]


def _factory(name: str):
    """Lazy-Konstruktor je Tool-Klassenname."""
    if name == "DicomProcessorTool":
        from .dicom import DicomProcessorTool
        return DicomProcessorTool()
    if name == "ImageVisualizerTool":
        from .visualization import ImageVisualizerTool
        return ImageVisualizerTool()
    if name == "MyChestXRayClassifierTool":
        from .my_classifier_tool import MyChestXRayClassifierTool
        return MyChestXRayClassifierTool()
    if name == "XRVAnatomySegmenterTool":
        from .anatomy_segmentation import XRVAnatomySegmenterTool
        return XRVAnatomySegmenterTool()
    if name == "ChestXrayBasicTool":
        from .view_anatomy import ChestXrayBasicTool
        return ChestXrayBasicTool()
    if name == "SupportDeviceClassifierTool":
        from .support_devices import SupportDeviceClassifierTool
        return SupportDeviceClassifierTool()
    if name == "GroundingTool":
        from .grounding import GroundingTool
        return GroundingTool()
    if name == "LlavaMedVQATool":
        from .llava_med import LlavaMedVQATool
        return LlavaMedVQATool()
    if name == "ReportGenerationTool":
        from .report_generation import ReportGenerationTool
        return ReportGenerationTool()
    if name == "ImageGenerationTool":
        from .optional_generation import ImageGenerationTool
        return ImageGenerationTool()
    raise KeyError(f"Unbekanntes Tool: {name}")


ALL_TOOL_NAMES: List[str] = CORE_DEFAULT + OPTIONAL_STUBS


def build_tools(selected_tools: Optional[List[str]] = None, strict: bool = False) -> Dict[str, object]:
    """
    Initialisiert die ausgewählten Tools. selected_tools = None -> CORE_DEFAULT.
    Rückgabe: dict {ClassName: tool_instance}.
    """
    names = selected_tools if selected_tools is not None else CORE_DEFAULT
    tools: Dict[str, object] = {}
    for name in names:
        try:
            tools[name] = _factory(name)
            print(f"[tools] geladen: {name}")
        except Exception as e:
            msg = f"[tools] übersprungen: {name} ({e})"
            if strict:
                raise
            print(msg)
    return tools


def tool_descriptions(tools: Dict[str, object]) -> str:
    """Erzeugt eine kompakte Beschreibung der geladenen Tools (für den Agent-Prompt)."""
    lines = []
    for cls_name, tool in tools.items():
        name = getattr(tool, "name", cls_name)
        desc = getattr(tool, "description", "")
        lines.append(f"- {name} ({cls_name}): {desc}")
    return "\n".join(lines)
