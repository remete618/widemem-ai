from __future__ import annotations

import re
from typing import List, Optional, Tuple

from widemem.core.types import YMYLConfig

_YMYL_STRONG_PATTERNS = {
    "health": [
        r"\b(blood pressure|cholesterol level|diabetes diagnosis|mental health"
        r"|cancer treatment|anxiety disorder|depression diagnosis)\b",
    ],
    "medical": [
        r"\b(lab results|medical condition|chronic condition"
        r"|medical history|treatment plan|medical emergency)\b",
    ],
    "financial": [
        r"\b(bank account|savings account|credit score|credit card|mortgage rate"
        r"|loan payment|tax return|retirement fund|401k|IRA contribution"
        r"|pension plan|capital gains)\b",
    ],
    "legal": [
        r"\b(power of attorney|child custody|legal action|court order"
        r"|legal counsel|divorce proceedings|estate planning)\b",
    ],
    "safety": [
        r"\b(emergency contact|next of kin|blood type|epipen"
        r"|pacemaker|DNR order|medical alert)\b",
    ],
    "insurance": [
        r"\b(insurance policy|insurance premium|insurance claim"
        r"|health insurance|life insurance|coverage plan)\b",
    ],
    "tax": [
        r"\b(tax return|tax filing|W-2|1099|IRS audit"
        r"|capital gains tax|tax deduction)\b",
    ],
    "pharmaceutical": [
        r"\b(side effect|drug interaction|contraindication"
        r"|over-the-counter|prescription medication)\b",
    ],
}

_YMYL_WEAK_PATTERNS = {
    "health": [
        r"\b(doctor|hospital|diagnosis|symptom|allerg|medication|prescription|surgery|disease|illness|therapy|therapist|depression|anxiety|diabetes|cancer)\b",
    ],
    "medical": [
        r"\b(clinic|treatment|dosage|vaccine|chronic|acute|ambulance|nurse|specialist|referral|x-ray|MRI|scan)\b",
    ],
    "financial": [
        r"\b(bank|savings|invest|portfolio|stock|bond|mortgage|loan|debt|credit|salary|income|retirement|pension|budget|expense)\b",
    ],
    "legal": [
        r"\b(lawyer|attorney|lawsuit|court|contract|custody|divorce|estate|liability|compliance|regulation|verdict|settlement)\b",
    ],
    "safety": [
        r"\b(evacuation|flood|fire alarm)\b",
    ],
    "insurance": [
        r"\b(insurance|premium|deductible|coverage|claim|copay|beneficiary)\b",
    ],
    "tax": [
        r"\b(deduction|audit|write-off|exemption|filing)\b",
    ],
    "pharmaceutical": [
        r"\b(drug|pharmaceutical|dosage|pharmacist|prescription)\b",
    ],
}


class YMYLResult:
    __slots__ = ("category", "confidence")

    def __init__(self, category: Optional[str], confidence: str = "none") -> None:
        self.category = category
        self.confidence = confidence

    @property
    def is_ymyl(self) -> bool:
        return self.category is not None

    @property
    def is_strong(self) -> bool:
        return self.confidence == "strong"


def classify_ymyl_detailed(text: str, config: YMYLConfig) -> YMYLResult:
    if not config.enabled:
        return YMYLResult(None, "none")

    text_lower = text.lower()

    for category in config.categories:
        patterns = _YMYL_STRONG_PATTERNS.get(category, [])
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return YMYLResult(category, "strong")

    weak_hits: List[Tuple[str, int]] = []
    for category in config.categories:
        patterns = _YMYL_WEAK_PATTERNS.get(category, [])
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text_lower, re.IGNORECASE))
        if count > 0:
            weak_hits.append((category, count))

    if not weak_hits:
        return YMYLResult(None, "none")

    weak_hits.sort(key=lambda x: x[1], reverse=True)
    best_category, best_count = weak_hits[0]

    if best_count >= 2:
        return YMYLResult(best_category, "strong")

    return YMYLResult(best_category, "weak")


def classify_ymyl(text: str, config: YMYLConfig) -> Optional[str]:
    result = classify_ymyl_detailed(text, config)
    return result.category


def classify_ymyl_batch(texts: List[str], config: YMYLConfig) -> List[Optional[str]]:
    return [classify_ymyl(t, config) for t in texts]


def is_ymyl(text: str, config: YMYLConfig) -> bool:
    return classify_ymyl(text, config) is not None


def is_ymyl_strong(text: str, config: YMYLConfig) -> bool:
    return classify_ymyl_detailed(text, config).is_strong
