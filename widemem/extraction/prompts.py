FACT_EXTRACTION_SYSTEM = """You are a fact extraction engine. Extract concrete, memorable facts from conversations.

Rules:
- Extract only factual statements, preferences, and personal details
- Each fact should be self-contained and understandable without context
- Preserve the original language of the conversation
- Rate each fact's importance from 1-10:
  - 1-3: Trivial mentions, passing comments
  - 4-6: Useful context, moderate preferences
  - 7-9: Core identity, strong preferences, critical information
  - 10: Essential, defining facts
{ymyl_instruction}
{custom_topics_instruction}
Respond with JSON only."""

FACT_EXTRACTION_PROMPT = """Extract facts from this conversation:

{text}

Respond with this exact JSON format:
{{"facts": [{{"content": "the fact", "importance": 7, "category": "general"}}, ...]}}

If there are no meaningful facts, respond with: {{"facts": []}}"""

YMYL_INSTRUCTION = (
    "\nCRITICAL: Facts about health, medical conditions, medications, finances, "
    "legal matters, insurance, taxes, or safety are EXTREMELY important. "
    "Rate these at 8-10 importance and tag their category accordingly. "
    'These are "Your Money or Your Life" (YMYL) facts that must never be lost.'
)

CUSTOM_TOPICS_INSTRUCTION = """
Pay special attention to facts about these topics: {topics}. Extract these with higher priority."""


def build_extraction_system(
    ymyl_enabled: bool = False,
    custom_topics: list = None,
) -> str:
    ymyl_inst = YMYL_INSTRUCTION if ymyl_enabled else ""
    topics_inst = ""
    if custom_topics:
        topics_inst = CUSTOM_TOPICS_INSTRUCTION.format(topics=", ".join(custom_topics))

    return FACT_EXTRACTION_SYSTEM.format(
        ymyl_instruction=ymyl_inst,
        custom_topics_instruction=topics_inst,
    )
