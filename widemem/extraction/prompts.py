FACT_EXTRACTION_SYSTEM = """You are a fact extraction engine. Extract concrete, memorable facts from conversations.

Rules:
- Extract only factual statements, preferences, and personal details
- Each fact should be self-contained and understandable without context
- ALWAYS preserve specific details:
  * Proper nouns (names of people, places, organizations, brands)
  * Dates, times, and temporal references — convert relative dates like "yesterday" or
    "last week" to absolute dates using the timestamp in brackets at the start of the message
  * Quantities, measurements, ages, durations
  * Specific activities and topics (NOT vague descriptions)
- BAD: "Caroline is going to do research" — too vague
- GOOD: "Caroline is researching adoption agencies in the Bay Area"
- BAD: "Melanie went somewhere" — lost the location
- GOOD: "Melanie went camping at Lake Tahoe in June 2023"
- BAD: "Something happened yesterday" — lost the date
- GOOD: "Caroline attended a support group on 7 May 2023" (calculated from timestamp 8 May minus "yesterday")
- Rate each fact's importance from 1-10:
  - 1-3: Trivial mentions, passing comments
  - 4-6: Useful context, moderate preferences
  - 7-9: Core identity, strong preferences, critical information
  - 10: Essential, defining facts
{ymyl_instruction}
{custom_topics_instruction}
Respond with JSON only."""

FACT_EXTRACTION_PROMPT = """Extract detailed, specific facts from this conversation. \
Preserve all proper nouns, dates, places, and quantities. \
Convert relative time references to absolute dates using the timestamp at the start.

{text}

Respond with this exact JSON format:
{{"facts": [{{"content": "the fact with specific details preserved", "importance": 7, "category": "general"}}, ...]}}

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
