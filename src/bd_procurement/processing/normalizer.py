import hashlib
import re
from typing import Any, Dict, Optional, Tuple

LEGAL_ENTITIES = [
    (r"\b(p\.?\s*t\.?|perseroan\s+terbatas)\b", "pt"),
    (r"\b(c\.?\s*v\.?|commanditaire\s+vennootschap)\b", "cv"),
    (r"\b(u\.?\s*d\.?|usaha\s+dagang)\b", "ud"),
    (r"\b(p\.?\s*d\.?|perusahaan\s+daerah)\b", "pd"),
    (r"\b(perum|perusahaan\s+umum)\b", "perum"),
    (r"\bpersero\b", "persero"),
    (r"\bkoperasi\b", "koperasi"),
    (r"\byayasan\b", "yayasan"),
    (r"\btbk\.?\b", "tbk"),
]


def clean_legal_entity_name(raw_name: str) -> Tuple[str, str, Optional[str]]:
    if not raw_name or not raw_name.strip():
        return "", "unknown", None

    text = raw_name.lower().strip()

    detected_legal_form = "other"
    for pattern, std_form in LEGAL_ENTITIES:
        if re.search(pattern, text):
            detected_legal_form = std_form

            text = re.sub(pattern, "", text)
            break

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if detected_legal_form in ("pt", "cv", "ud", "pd"):
        canonical_name = f"{detected_legal_form.upper()} {text.title()}".strip()
    else:
        canonical_name = text.title().strip()

    alias_hash = hashlib.sha256(canonical_name.lower().encode("utf-8")).hexdigest()[:12]
    return canonical_name, detected_legal_form, alias_hash


def build_organization_dimension_record(
    raw_id: str,
    raw_name: str,
    org_type: str,
    extra_attributes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    canonical_name, legal_form, alias_hash = clean_legal_entity_name(raw_name)

    norm_id = f"org_{raw_id}" if raw_id and raw_id.strip() else f"hash_{alias_hash}"

    return {
        "org_key": norm_id,
        "raw_id": raw_id,
        "raw_name": raw_name,
        "canonical_name": canonical_name,
        "legal_form": legal_form,
        "org_type": org_type,
        "alias_fingerprint": alias_hash,
        "confidence": 0.95 if raw_id else 0.75,
        "needs_review": False if raw_id else True,
        "attributes": extra_attributes or {},
    }
