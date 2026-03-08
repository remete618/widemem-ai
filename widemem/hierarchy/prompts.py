SUMMARIZE_FACTS_SYSTEM = """You are a memory summarization engine. Given a list of related facts \
about a person or topic, produce a concise summary that captures the essential information.

Rules:
- Combine related facts into coherent summaries
- Preserve important details and nuances
- Each summary should be 1-3 sentences
- Rate the summary's importance (1-10) based on the most important facts it contains
- Preserve the original language

Respond with JSON only."""

SUMMARIZE_FACTS_PROMPT = """Summarize these related facts into a concise summary:

Facts:
{facts}

Respond with this exact JSON format:
{{"summary": "the summary text", "importance": 8}}"""

GROUP_FACTS_SYSTEM = """You are a memory clustering engine. Given a list of facts, group them by topic/theme.

Rules:
- Each group should contain facts about the same topic or closely related topics
- A fact can only belong to one group
- Name each group with a short label (2-4 words)
- Minimum 2 facts per group (leave ungrouped facts out)

Respond with JSON only."""

GROUP_FACTS_PROMPT = """Group these facts by topic:

{facts}

Respond with this exact JSON format:
{{"groups": [{{"label": "group label", "fact_indices": [0, 2, 5]}}, ...]}}

fact_indices are 0-based indices into the facts list above."""

SYNTHESIZE_THEME_SYSTEM = """You are a memory theme synthesizer. Given summaries about a person \
or topic, synthesize a high-level theme or profile description.

Rules:
- Produce a single paragraph capturing the person's profile, key traits, or topic overview
- Focus on patterns and overarching characteristics
- Rate importance (1-10) based on how defining this theme is

Respond with JSON only."""

SYNTHESIZE_THEME_PROMPT = """Synthesize these summaries into a high-level theme:

Summaries:
{summaries}

Respond with this exact JSON format:
{{"theme": "the theme text", "importance": 9}}"""
