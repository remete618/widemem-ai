"""Prompt-injection sanitizer for memory ingestion.

Memory content is fed into LLM prompts at extraction time, conflict resolution
time, and answer time. Hostile content stored once can poison every later call.
This module strips the most well-known prompt-injection patterns before content
reaches the LLM.

Conservative by design: only matches well-established attack patterns to avoid
false positives on legitimate user content (e.g. "I forget everything by Monday"
is fine; "ignore all previous instructions" is not).

This is a baseline defense, not a complete solution. Defense-in-depth still
requires output validation, prompt structure that distinguishes data from
instruction, and provider-side guardrails.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Each pattern is (compiled_regex, category_label).
# Categories: instruction-override, system-tag, role-marker, action-injection,
# jailbreak, memory-attack.
_INJECTION_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # Direct instruction overrides (most common attack vector)
    (
        re.compile(
            r"\b(?:ignore|disregard)\s+(?:all\s+)?(?:previous|prior|above|earlier|the)\s+(?:instructions?|prompts?|messages?|context|rules?)\b",
            re.IGNORECASE,
        ),
        "instruction-override",
    ),
    (
        re.compile(
            r"\bforget\s+(?:everything\s+)?(?:previous|prior|the\s+above|all\s+instructions?|what\s+(?:I|you)\s+(?:said|told))\b",
            re.IGNORECASE,
        ),
        "instruction-override",
    ),
    # Explicit "new instructions" override
    (
        re.compile(
            r"\b(?:your\s+)?new\s+(?:instructions?|task|directive|role)\s+(?:is|are)\b",
            re.IGNORECASE,
        ),
        "instruction-override",
    ),
    # System-prompt injection markers
    (re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE), "system-tag"),
    (re.compile(r"\[\s*/?\s*system\s*\]", re.IGNORECASE), "system-tag"),
    (re.compile(r"<\s*/?\s*\|im_start\|\s*>", re.IGNORECASE), "system-tag"),
    (re.compile(r"<\s*/?\s*\|im_end\|\s*>", re.IGNORECASE), "system-tag"),
    # Role injection at start of line (chat template confusion)
    (
        re.compile(r"^\s*(?:system|assistant)\s*:\s", re.IGNORECASE | re.MULTILINE),
        "role-marker",
    ),
    # Common jailbreak vocabulary
    (re.compile(r"\bDAN\s+mode\b", re.IGNORECASE), "jailbreak"),
    (re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE), "jailbreak"),
    (re.compile(r"\bjailbreak\b", re.IGNORECASE), "jailbreak"),
    # Memory-targeted destructive actions
    (
        re.compile(
            r"\b(?:delete|remove|drop|wipe|erase)\s+(?:all\s+)?(?:my\s+)?(?:memories|memory|facts|data|history|records?)\b",
            re.IGNORECASE,
        ),
        "memory-attack",
    ),
]

REDACT_MARKER = "[REDACTED]"


def detect_injection(text: str) -> List[str]:
    """Return categories of injection patterns found in text. Empty list if none.

    Use this when you want to log or audit without modifying the content.
    Returned categories may repeat if multiple distinct patterns from the same
    category are found.
    """
    if not text:
        return []
    found: List[str] = []
    for pattern, category in _INJECTION_PATTERNS:
        if pattern.search(text):
            found.append(category)
    return found


def sanitize(text: str, redact_marker: str = REDACT_MARKER) -> Tuple[str, List[str]]:
    """Strip known prompt-injection patterns. Replace with redact_marker.

    Returns (sanitized_text, list_of_categories_found). Categories may repeat if
    the same category was matched by multiple distinct patterns.

    On any non-empty match list, a warning is logged so the integrator can
    surface it (audit log, metric, etc.).
    """
    if not text:
        return text, []
    sanitized = text
    found: List[str] = []
    for pattern, category in _INJECTION_PATTERNS:
        if pattern.search(sanitized):
            found.append(category)
            sanitized = pattern.sub(redact_marker, sanitized)
    if found:
        logger.warning("Prompt-injection patterns sanitized: %s", found)
    return sanitized, found
