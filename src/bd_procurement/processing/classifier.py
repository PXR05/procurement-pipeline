import re
from typing import Any, Dict, List, Optional

from bd_procurement.config import IT_EXCLUSION_KEYWORDS, IT_KEYWORDS


class ITDomainClassifier:
    def __init__(
        self,
        keywords_dict: Optional[Dict[str, List[str]]] = None,
        exclusion_list: Optional[List[str]] = None,
    ):
        self.keywords = keywords_dict or IT_KEYWORDS
        self.exclusions = [
            ex.lower() for ex in (exclusion_list or IT_EXCLUSION_KEYWORDS)
        ]

        self._exclusion_regex = (
            re.compile(
                r"\b(" + "|".join(re.escape(term) for term in self.exclusions) + r")\b",
                re.IGNORECASE,
            )
            if self.exclusions
            else None
        )

        self._compiled_subcategories = {}
        for subcat, terms in self.keywords.items():
            pattern = re.compile(
                r"\b(" + "|".join(re.escape(term) for term in terms) + r")\b",
                re.IGNORECASE,
            )
            self._compiled_subcategories[subcat] = pattern

    def classify(self, title: str, category: Optional[str] = None) -> Dict[str, Any]:
        full_text = f"{title or ''} {category or ''}".strip().lower()
        if not full_text:
            return {
                "is_it": False,
                "subcategories": [],
                "primary_subcategory": "NON_IT",
                "matched_keywords": [],
                "confidence": 0.0,
                "flagged_exclusion": False,
            }

        if self._exclusion_regex and self._exclusion_regex.search(full_text):
            return {
                "is_it": False,
                "subcategories": [],
                "primary_subcategory": "EXCLUDED",
                "matched_keywords": [],
                "confidence": 0.0,
                "flagged_exclusion": True,
            }

        matched_subcats = []
        matched_tokens = []

        for subcat, pattern in self._compiled_subcategories.items():
            matches = pattern.findall(full_text)
            if matches:
                matched_subcats.append(subcat)
                matched_tokens.extend([m.lower() for m in matches])

        if not matched_subcats:
            return {
                "is_it": False,
                "subcategories": [],
                "primary_subcategory": "NON_IT",
                "matched_keywords": [],
                "confidence": 0.0,
                "flagged_exclusion": False,
            }

        primary_subcat = matched_subcats[0]
        confidence = min(1.0, 0.6 + 0.15 * len(set(matched_tokens)))

        return {
            "is_it": True,
            "subcategories": matched_subcats,
            "primary_subcategory": primary_subcat,
            "matched_keywords": list(set(matched_tokens)),
            "confidence": round(confidence, 2),
            "flagged_exclusion": False,
        }
