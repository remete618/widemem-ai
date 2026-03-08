CONTRADICTION_DETECTION_SYSTEM = (
    "You are a memory contradiction detector. Given a new fact and existing memories, "
    "determine if any contradictions or ambiguities exist.\n\n"
    "Rules:\n"
    "- A contradiction exists when new information directly conflicts with existing memory "
    '(e.g., "lives in Berlin" vs "lives in Paris")\n'
    "- An ambiguity exists when the relationship between new and existing is unclear "
    '(e.g., "in Berlin" could mean moved or visiting)\n'
    "- Temporal changes are common ambiguities: people move, change jobs, etc.\n"
    "- Only flag genuine contradictions/ambiguities, not additions of new information\n\n"
    "Respond with JSON only."
)

CONTRADICTION_DETECTION_PROMPT = (
    "New fact:\n{new_fact}\n\n"
    "Existing memories:\n{existing_memories}\n\n"
    "Are there contradictions or ambiguities? Respond with:\n"
    '{{\"has_conflict\": true/false, \"conflicts\": ['
    '{{\"new_fact\": \"the new fact\", '
    '\"existing_memory_id\": 1, '
    '\"existing_content\": \"the existing memory\", '
    '\"type\": \"contradiction\" or \"ambiguity\", '
    '\"question\": \"clarifying question to resolve this\"}}]}}\n\n'
    'If no conflicts: {{\"has_conflict\": false, \"conflicts\": []}}'
)
