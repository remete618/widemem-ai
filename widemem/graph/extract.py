from __future__ import annotations

import json
from typing import List, Optional, Tuple

from widemem.providers.llm.base import BaseLLM

Triple = Tuple[str, str, str]  # (subject, relation, object), entities lowercased

TRIPLE_SYSTEM = (
    "You extract knowledge-graph relationships from a personal memory. "
    "Return typed triples that connect entities, so a downstream system can "
    "traverse from one entity to related ones."
)

TRIPLE_PROMPT = """Extract entity-relationship triples from the memory below.

Rules:
- Each triple is [subject, relation, object].
- subject and object are concrete entities: people, places, organizations,
  objects, activities, topics. NOT pronouns, NOT whole sentences.
- relation is a short snake_case verb phrase: lives_in, works_at, moved_from,
  pursuing, identifies_as, attended, owns, dislikes, related_to, happened_on.
- Lowercase everything. Keep multi-word entities intact ("san francisco").
- Prefer 1-3 high-signal triples. If the memory states no relationship between
  two entities, return an empty list rather than inventing one.
- Include date/time as an object with relation happened_on when present.
- SKIP emotional states, feelings, moods, and vague abstractions. They are
  not traversable structure and only add noise.
  * BAD: [caroline, expressed, happiness] / [caroline, feels, accepted]
  * GOOD: [caroline, attended, lgbtq support group] / [caroline, pursuing, adoption]
  Keep only durable relationships: identity, location, affiliation, kinship,
  possession, activities, organizations, events, and dates.

Memory: {text}

Return ONLY a JSON object: {{"triples": [["subject","relation","object"], ...]}}"""


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def extract_triples(llm: BaseLLM, text: str) -> List[Triple]:
    """One LLM call -> list of (subject, relation, object). Best-effort: on any
    parse/LLM failure returns [] (graph just gets no edges for this memory; the
    flat path is unaffected)."""
    if not text or not text.strip():
        return []
    try:
        data = llm.generate_json(TRIPLE_PROMPT.format(text=text), system=TRIPLE_SYSTEM)
    except Exception:
        return []
    return _parse_triples_dict(data)


def parse_triples(raw: Optional[str]) -> List[Triple]:
    if not raw:
        return []
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        data = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return []
    return _parse_triples_dict(data)


def _parse_triples_dict(data: dict) -> List[Triple]:
    if not isinstance(data, dict):
        return []
    out: List[Triple] = []
    for t in data.get("triples", []):
        if not isinstance(t, (list, tuple)) or len(t) != 3:
            continue
        s, r, o = _norm(t[0]), _norm(t[1]).replace(" ", "_"), _norm(t[2])
        if s and r and o and s != o:
            out.append((s, r, o))
    return out
