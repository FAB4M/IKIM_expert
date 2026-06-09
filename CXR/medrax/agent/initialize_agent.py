"""CXR-Agent: Bildpfad -> (DICOM->PNG) -> Classifier+Grad-CAM -> Anatomie/View/Support-Device
-> Qwen-Fließtext (oder strukturierter Fallback). Ohne harte LangChain-Abhängigkeit.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from medrax.agent.prompts import SYSTEM_PROMPT, build_findings_message, prob_to_words
from medrax.agent.qwen_client import QwenClient
from medrax.tools import CORE_DEFAULT, build_tools, tool_descriptions

_PATH_RE = re.compile(r"[\"']?([A-Za-z]:\\[^\"'\n]+?\.(?:png|jpg|jpeg|dcm|dicom|bmp|tif|tiff)|/[^\s\"'\n]+?\.(?:png|jpg|jpeg|dcm|dicom|bmp|tif|tiff)|[^\s\"'\n]+?\.(?:png|jpg|jpeg|dcm|dicom|bmp|tif|tiff))[\"']?", re.IGNORECASE)


def extract_image_path(text: str) -> Optional[str]:
    """Findet einen Bild-/DICOM-Pfad in einem Text."""
    if not text:
        return None
    m = _PATH_RE.search(text)
    if m:
        cand = m.group(1).strip().strip("\"'")
        return cand
    # vielleicht ist der ganze Text ein Pfad
    if Path(text.strip()).exists():
        return text.strip()
    return None


class CXRAgent:
    def __init__(self, tools: Dict[str, object], llm: QwenClient):
        self.tools = tools
        self.llm = llm
        self.system_prompt = SYSTEM_PROMPT + "\n\nLoaded tools:\n" + tool_descriptions(tools)

    # ------------------------------------------------------- Hilfsfunktionen
    def _to_png_if_dicom(self, image_path: str) -> str:
        from medrax.utils.dicom_utils import is_dicom

        if not is_dicom(image_path):
            return image_path
        dicom_tool = self.tools.get("DicomProcessorTool")
        if dicom_tool is None:
            return image_path  # Classifier kann DICOM auch direkt lesen
        output, meta = dicom_tool._run(dicom_path=image_path)
        if meta.get("analysis_status") == "completed":
            return output["image_path"]
        return image_path

    def _safe_tool_output(self, cls_name: str, **kwargs) -> Optional[Dict]:
        """Ruft ein Tool auf und liefert dessen output nur bei Erfolg, sonst None."""
        tool = self.tools.get(cls_name)
        if tool is None:
            return None
        try:
            out, meta = tool._run(**kwargs)
            if meta.get("analysis_status") == "completed":
                return out
            print(f"[agent] {cls_name}: {out.get('error')}")
        except Exception as e:
            print(f"[agent] {cls_name} Fehler: {e}")
        return None

    def _fallback_text(self, analysis: Dict, localization: Optional[Dict],
                       view: Optional[Dict], support: Optional[Dict]) -> str:
        """Structured text answer when the LLM is not reachable (English)."""
        preds = analysis.get("predictions", {})
        top = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:4]
        parts = ["[LLM not reachable - structured summary]"]
        if view:
            ctr = view.get("cardiothoracic_ratio")
            parts.append(f"Acquisition: view {view.get('view')}"
                         + (f", CTR {ctr}" if ctr is not None else ""))
        parts.append("Most salient classes (screening probabilities):")
        for name, p in top:
            parts.append(f"  - {name}: {p:.2f} ({prob_to_words(p)})")
        if localization and localization.get("localized"):
            parts.append(f"Coarse localization: {localization.get('zone')} "
                         f"({localization.get('method')}).")
        else:
            parts.append("No reliable localization (not localized).")
        if support:
            parts.append(f"Support device: {support.get('present')} "
                         f"(confidence {support.get('confidence')}).")
        from medrax.agent.prompts import DISCLAIMER
        parts.append(f"Note: {DISCLAIMER}")
        return "\n".join(parts)

    # ------------------------------------------------------------- Pipeline
    def analyze_image(self, image_path: str,
                      question: str = "What do you see on this chest X-ray?") -> Dict:
        image_path = str(image_path)
        if not Path(image_path).exists():
            return {"ok": False, "answer": f"Image not found: {image_path}"}

        clf = self.tools.get("MyChestXRayClassifierTool")
        if clf is None:
            return {"ok": False, "answer":
                    "Classifier tool not loaded (is a checkpoint present?). "
                    "Please train first or create a mini checkpoint."}

        png = self._to_png_if_dicom(image_path)

        # 1) Klassifikation + Grad-CAM
        analysis = clf.analyze(png)

        # 2) View/Anatomie (ianpan): AP/PA/lateral, Lunge re/li, Herz, CTR
        view = self._safe_tool_output("ChestXrayBasicTool", image_path=png)

        # 3) Anatomie-Lokalisation (TorchXRayVision) der Grad-CAM-Aktivierung
        localization = None
        anat = self.tools.get("XRVAnatomySegmenterTool")
        if anat is not None and analysis.get("heatmap") is not None:
            out, meta = anat._run(
                image_path=png,
                heatmap=analysis["heatmap"],
                top_prob=analysis["top_pathology"]["prob"],
                top_name=analysis["top_pathology"]["name"],
            )
            if meta.get("analysis_status") == "completed":
                localization = out.get("localization")
            else:
                print(f"[agent] XRVAnatomySegmenterTool: {out.get('error')}")

        # 4) Support-Device (CheXpert-DenseNet)
        support = self._safe_tool_output("SupportDeviceClassifierTool", image_path=png)

        # 5) Qwen-Fließtext
        user_msg = build_findings_message(analysis, localization, view=view,
                                          support=support, question=question)
        llm_res = self.llm.chat(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": user_msg}],
        )
        if llm_res.get("ok"):
            answer = llm_res["content"]
            llm_ok = True
        else:
            answer = self._fallback_text(analysis, localization, view, support)
            llm_ok = False

        return {
            "ok": True,
            "llm_ok": llm_ok,
            "answer": answer,
            "predictions": analysis.get("predictions"),
            "top": analysis.get("top"),
            "view": view,
            "localization": localization,
            "support_device": support,
            "heatmap_path": analysis.get("heatmap_path"),
            "input_image": image_path,
            "llm_error": None if llm_ok else llm_res,
        }

    # ----------------------------------------------------------------- ask
    def ask(self, text: str) -> Dict:
        """Sehr einfache Routing-Logik: Bildpfad -> analyze_image, sonst LLM-Chat."""
        path = extract_image_path(text)
        if path and Path(path).exists():
            return self.analyze_image(path, question=text)
        # kein Bild -> reiner LLM-Chat (oder Hinweis)
        res = self.llm.simple(self.system_prompt, text)
        if res.get("ok"):
            return {"ok": True, "llm_ok": True, "answer": res["content"]}
        return {"ok": True, "llm_ok": False,
                "answer": "Please provide a valid image path (PNG/JPG/DICOM). "
                          f"(LLM not reachable: {res.get('error')})"}


def initialize_agent(selected_tools: Optional[List[str]] = None,
                     llm: Optional[QwenClient] = None,
                     verbose: bool = True) -> CXRAgent:
    """
    Initialisiert den Agenten mit selektiv geladenen Tools und Qwen-Client.
    selected_tools=None -> CORE_DEFAULT.
    """
    import config

    config.ensure_dirs()
    names = selected_tools if selected_tools is not None else CORE_DEFAULT
    if verbose:
        print(f"[agent] Initialisiere Tools: {names}")
    tools = build_tools(names)
    client = llm or QwenClient()
    if verbose:
        print(f"[agent] LLM: {client.model} @ {client.base_url}")
    return CXRAgent(tools, client)
