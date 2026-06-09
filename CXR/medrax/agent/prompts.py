"""
System prompts and message building for the CXR agent.

The agent turns structured tool outputs (classifier probabilities, Grad-CAM
localization, view, support device) into a radiologist-style narrative answer
in ENGLISH. The wording is geared toward the FINDINGS / IMPRESSION report style
the language model was domain-adapted on.
"""

from __future__ import annotations

from typing import Dict, Optional

DISCLAIMER = (
    "Research and development project, NOT clinically validated. Not a medical "
    "diagnosis; all statements are technical assistive signals of self-trained models."
)

SYSTEM_PROMPT = f"""\
You are an AI assistant for reading chest X-rays (CXR). You orchestrate specialized
tools and summarize their results. Always answer in ENGLISH.

Signals provided to you as structured tool results:
- Classifier (VinBigData-15): probabilities (0..1) for 14 findings + 'No finding'.
  These are SCREENING hints, NOT a diagnosis and NOT object detection/boxes.
  A high 'No finding' probability argues against relevant findings.
- View/anatomy (ianpan): projection (AP/PA/lateral), detection of right/left lung and
  heart, cardiothoracic ratio (CTR; > ~0.5 may suggest an enlarged cardiac silhouette).
- Anatomic localization (TorchXRayVision): the coarse zone of the strongest Grad-CAM
  activation (e.g. 'right upper zone' or 'cardiac region'), plus detected regions.
- Support-device detector: whether a support device (tube/line/pacemaker/drain) is
  likely visible (yes/no + confidence).

RULES for your answer:
1. Write a coherent, radiologist-style narrative in ENGLISH, in the style of a brief
   chest radiograph report (you may use 'Findings:' and 'Impression:'). 3-7 sentences.
   No bare bullet lists, no JSON.
2. Name the most salient classes with their probability in words (e.g. 'with increased
   probability', 'borderline', 'unlikely').
3. Mention a localization ONLY if the localization tool returns localized=true. If
   localized=false (e.g. No finding, low probability, or a diffuse heatmap), do NOT
   claim any zone — at most state that no clear localization is possible.
4. Be cautious ('compatible with', 'suggestive of'), no definitive clinical statement.
   With a high 'No finding' probability, describe the image as unremarkable.
5. Mention the projection (view) and, if relevant, the support device.
6. ALWAYS close with a short note: {DISCLAIMER}
"""


def prob_to_words(p: float) -> str:
    if p >= 0.80:
        return "markedly increased probability"
    if p >= 0.60:
        return "increased probability"
    if p >= 0.40:
        return "borderline probability"
    if p >= 0.20:
        return "low probability"
    return "very low probability"


def build_findings_message(
    analysis: Dict,
    localization: Optional[Dict] = None,
    view: Optional[Dict] = None,
    support: Optional[Dict] = None,
    question: str = "What do you see on this chest X-ray?",
    top_k: int = 5,
) -> str:
    """
    Builds the user message for Qwen from the structured tool results.
    analysis    : output of MyChestXRayClassifierTool.analyze(...)
    localization: output['localization'] of XRVAnatomySegmenterTool
    view        : output of ChestXrayBasicTool (view, lungs, heart, CTR)
    support     : output of SupportDeviceClassifierTool
    """
    preds: Dict[str, float] = analysis.get("predictions", {})
    ordered = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_k]

    lines = [f"User question: {question}", "", "Structured tool results:"]

    if view:
        v = view.get("view", "?")
        ctr = view.get("cardiothoracic_ratio")
        ctr_txt = f", CTR={ctr}" if ctr is not None else ""
        lines.append(
            f"View/anatomy (ianpan): view={v}, right lung={view.get('right_lung')}, "
            f"left lung={view.get('left_lung')}, heart={view.get('heart')}{ctr_txt}."
        )

    lines.append("Classifier (VinBigData-15) – top probabilities:")
    for name, p in ordered:
        lines.append(f"  - {name}: {p:.2f} ({prob_to_words(p)})")

    if localization is not None:
        if localization.get("localized"):
            lines.append(
                f"Localization (TorchXRayVision): localized=true, zone='{localization.get('zone')}', "
                f"method={localization.get('method')} (coarse; image-left = patient-right is accounted for)."
            )
        else:
            lines.append(
                f"Localization: localized=false (reason: {localization.get('reason')}). "
                "-> Do NOT claim any zone."
            )
    else:
        lines.append("Localization: not available.")

    if support:
        lines.append(
            f"Support device: {support.get('present')} (confidence {support.get('confidence')})."
        )

    lines.append("")
    lines.append(
        "From this, write a short radiologist-style narrative report (in English) following the rules. "
        "Mention the view and – if relevant – the support device; mention localization only if localized=true."
    )
    return "\n".join(lines)
