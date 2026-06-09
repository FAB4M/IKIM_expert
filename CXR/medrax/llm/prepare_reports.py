"""CheXpert-Plus-Berichte -> sauberer JSONL-Korpus fürs LLM-Pretraining.
Nur section_findings + section_impression; stdlib csv (kein pandas).

    python -m medrax.llm.prepare_reports --max-samples 20000
    python -m medrax.llm.prepare_reports --preview-only   # QC
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Defaults aus config (mit Fallback, falls config nicht importierbar ist)
try:
    import config

    _DEFAULT_CSV = str(config.CHEXPERT_CSV) if hasattr(config, "CHEXPERT_CSV") else None
    _DEFAULT_OUT = str(config.REPORTS_CORPUS) if hasattr(config, "REPORTS_CORPUS") else None
    _BASE = str(config.BASE_DIR)
except Exception:
    _BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _DEFAULT_CSV = None
    _DEFAULT_OUT = None

if not _DEFAULT_CSV:
    _DEFAULT_CSV = os.path.join(
        _BASE, "data", "chexpert", "metadata", "table_subsets", "df_chexpert_plus_240401.csv"
    )
if not _DEFAULT_OUT:
    _DEFAULT_OUT = os.path.join(_BASE, "data", "chexpert", "processed", "reports_corpus.jsonl")

# CSV-Limit hochsetzen (mehrzeilige Report-Felder können groß sein)
csv.field_size_limit(10 * 1024 * 1024)

_WS = re.compile(r"\s+")
# Fehlendes Leerzeichen nach Satzzeichen einfügen ("2.no" -> "2. no", "lung.No" -> "lung. No").
# Trifft NUR Punkt+Buchstabe (nicht Dezimalzahlen wie "2.5 cm").
_SENT_SPACE = re.compile(r"([.!?])([A-Za-z])")
# Anonymisierungs-/Boilerplate-Sätze (falls sie doch in findings/impression auftauchen)
_BOILERPLATE = [
    re.compile(r"This report has been anonymized.*?(?:patient\.|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"I have personally reviewed the images.*?(?:above\.|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bBy:\s*Dr\.[^\n]*", re.IGNORECASE),
    re.compile(r"_{3,}"),  # Unterschriftslinien
]
# Reine Platzhalter / leere Inhalte
_PLACEHOLDER = re.compile(r"^(none|n/?a|nil|\W*)$", re.IGNORECASE)


def clean_text(t: str) -> str:
    """Mehrzeiliges Abschnittsfeld -> eine saubere Zeile."""
    if not t:
        return ""
    for pat in _BOILERPLATE:
        t = pat.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    t = _SENT_SPACE.sub(r"\1 \2", t)
    # führende Doppelpunkte / Bindestriche entfernen
    t = t.lstrip(":-").strip()
    if _PLACEHOLDER.match(t):
        return ""
    return t


# --- Zusätzliche Normalisierung (vom User gewünscht) ----------------------
# Gängige CXR-Abkürzungen, die nach dem Kleinschreiben wieder groß werden:
_ABBR = [
    "CTR", "SVC", "IVC", "ETT", "ET", "NG", "NGT", "OG", "OGT", "IV", "PICC",
    "CVC", "CVL", "AP", "PA", "CT", "MRI", "US", "ICU", "CCU", "ED", "ER",
    "CHF", "COPD", "TB", "PE", "CABG", "LLL", "RLL", "RUL", "LUL", "RML",
    "IJ", "CXR", "EKG", "ECG", "ICD", "AICD", "VP", "GE", "NGT",
]
_ABBR_RE = re.compile(r"\b(" + "|".join(a.lower() for a in sorted(set(_ABBR), key=len, reverse=True)) + r")\b")
_SPINE_RE = re.compile(r"\b([tlc])(\d{1,2})\b")  # t4 -> T4
_SENT_START_RE = re.compile(r"(^|[.!?]\s+)([a-z])")

# Führende Technik-/View-Zeile (z. B. "PA AND LATERAL CHEST RADIOGRAPH.")
_TECH_KEYWORDS = ("RADIOGRAPH", "VIEW", "PROJECTION", "PORTABLE")
_FIRST_SENT_RE = re.compile(r"\s*(?:\d+\.\s*)?([^.]*\.)\s*")


def _has_lower(t: str) -> bool:
    return any(c.islower() for c in t)


def normalize_case(t: str) -> str:
    """GROSSBUCHSTABEN-Text -> normale Schreibweise; gemischter Text bleibt unverändert."""
    if not t or _has_lower(t):
        return t
    t = t.lower()
    t = _SENT_START_RE.sub(lambda m: m.group(1) + m.group(2).upper(), t)
    t = _ABBR_RE.sub(lambda m: m.group(1).upper(), t)
    t = _SPINE_RE.sub(lambda m: m.group(1).upper() + m.group(2), t)
    return t


def strip_leading_technique(t: str) -> str:
    """Entfernt bis zu 2 führende Technik-/View-Sätze (Aufnahmebeschreibung)."""
    for _ in range(2):
        m = _FIRST_SENT_RE.match(t)
        if not m:
            break
        first = m.group(1)
        up = first.upper()
        is_tech = (
            len(first) < 95
            and (any(k in up for k in _TECH_KEYWORDS)
                 or re.search(r"\bCHEST\b.*\d", up))  # "CHEST, ONE VIEW: 2-10-2001"
        )
        if is_tech:
            t = t[m.end():]
        else:
            break
    return t.strip()


def refine(t: str) -> str:
    """Technikzeilen entfernen + Schreibweise normalisieren."""
    return normalize_case(strip_leading_technique(t))


def build_example(findings: str, impression: str) -> str:
    """Formatiert ein Trainingsbeispiel im Befund-Stil."""
    parts = []
    if findings:
        parts.append("FINDINGS: " + findings)
    if impression:
        parts.append("IMPRESSION: " + impression)
    return "\n".join(parts)


def iter_clean_reports(csv_path, min_chars, split=None, max_scan=None, normalize=True):
    """Generator über (text, meta). Liefert bereits gesäuberte, gefilterte Beispiele."""
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        i_find = col.get("section_findings")
        i_imp = col.get("section_impression")
        i_split = col.get("split")
        if i_find is None and i_imp is None:
            raise ValueError(
                "Weder 'section_findings' noch 'section_impression' in der CSV gefunden. "
                f"Spalten: {header[:10]} ..."
            )

        stats = {"scanned": 0, "kept": 0, "empty": 0, "short": 0, "dup": 0, "split_skip": 0}
        seen = set()

        for row in reader:
            if max_scan and stats["scanned"] >= max_scan:
                break
            stats["scanned"] += 1
            if len(row) <= max(i for i in (i_find, i_imp, i_split) if i is not None):
                stats["empty"] += 1
                continue
            if split and i_split is not None and row[i_split].strip() != split:
                stats["split_skip"] += 1
                continue

            findings = clean_text(row[i_find]) if i_find is not None else ""
            impression = clean_text(row[i_imp]) if i_imp is not None else ""
            if normalize:
                findings = refine(findings)
                impression = refine(impression)
            if not findings and not impression:
                stats["empty"] += 1
                continue

            text = build_example(findings, impression)
            if len(text) < min_chars:
                stats["short"] += 1
                continue

            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            if h in seen:
                stats["dup"] += 1
                continue
            seen.add(h)

            stats["kept"] += 1
            yield text, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="CheXpert reports -> clean LM corpus")
    ap.add_argument("--csv", default=_DEFAULT_CSV)
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--max-samples", type=int, default=20000, help="max. behaltene Berichte")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--split", default=None, help="optional nur ein Split (z. B. 'train')")
    ap.add_argument("--max-scan", type=int, default=None, help="nur erste N Zeilen scannen (QC)")
    ap.add_argument("--preview", type=int, default=5, help="so viele Beispiele drucken")
    ap.add_argument("--preview-only", action="store_true", help="nur QC, nichts schreiben")
    ap.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                    help="GROSSBUCHSTABEN normalisieren + Technikzeilen entfernen")
    args = ap.parse_args(argv)

    if not os.path.exists(args.csv):
        print(f"[FEHLER] CSV nicht gefunden: {args.csv}")
        return 1

    print("=" * 78)
    print(" QUALITÄTSKONTROLLE – CheXpert-Berichte (section_findings + section_impression)")
    print("=" * 78)
    print("CSV:", args.csv)
    print("Out:", "(preview-only, keine Datei)" if args.preview_only else args.out)
    print()

    out_f = None
    if not args.preview_only:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out_f = open(args.out, "w", encoding="utf-8")

    last_stats = None
    n_written = 0
    char_sum = 0
    word_sum = 0
    printed = 0

    try:
        for text, stats in iter_clean_reports(
            args.csv, min_chars=args.min_chars, split=args.split,
            max_scan=args.max_scan, normalize=args.normalize,
        ):
            last_stats = stats
            char_sum += len(text)
            word_sum += len(text.split())

            if printed < args.preview:
                printed += 1
                print(f"------------------ BEISPIEL {printed} ------------------")
                print(text)
                print()

            if out_f is not None:
                out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n_written += 1

            if n_written >= args.max_samples:
                break
    finally:
        if out_f is not None:
            out_f.close()

    print("=" * 78)
    if last_stats:
        s = last_stats
        print(f"Gescannt: {s['scanned']} | behalten: {s['kept']} | leer: {s['empty']} "
              f"| zu kurz: {s['short']} | Duplikate: {s['dup']} | split übersprungen: {s['split_skip']}")
    if n_written:
        print(f"Beispiele genutzt: {n_written} | Ø {char_sum // n_written} Zeichen "
              f"| Ø {word_sum // n_written} Wörter")
    if not args.preview_only:
        print(f"Korpus geschrieben: {args.out}")
    else:
        print("(preview-only – keine Datei geschrieben)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
