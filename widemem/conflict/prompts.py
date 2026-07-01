BATCH_CONFLICT_RESOLUTION_SYSTEM = (
    "You are a memory conflict resolution engine. "
    "Given new facts and existing memories, decide what actions to take.\n\n"
    "Actions:\n"
    "- ADD: The fact is new information not captured by any existing memory\n"
    "- UPDATE: The fact contradicts, refines, or supersedes an existing memory "
    "(provide the target memory ID)\n"
    "- DELETE: An existing memory is now invalid based on the new facts "
    "(provide the target memory ID)\n"
    "- NONE: The fact is already captured by an existing memory, no action needed\n\n"
    "Rules:\n"
    "- Be conservative: prefer NONE over ADD if a fact is already well-captured\n"
    "- Detect contradictions: \"moved to San Francisco\" should UPDATE \"lives in Boston\"\n"
    "- Detect refinements: \"works as a senior engineer at Google\" should UPDATE "
    "\"works at Google\"\n"
    "- One action per new fact\n"
    "- Respond with JSON only."
)

BATCH_CONFLICT_RESOLUTION_PROMPT = (
    "New facts to process:\n{new_facts}\n\n"
    "Existing memories:\n{existing_memories}\n\n"
    "For each new fact, decide the action. Respond with this exact JSON format:\n"
    '{{\"actions\": [{{\"fact_index\": 0, \"action\": \"add\", '
    '\"target_id\": null, \"importance\": 7}}, '
    '{{\"fact_index\": 1, \"action\": \"update\", '
    '\"target_id\": 3, \"importance\": 8}}, ...]}}\n\n'
    "fact_index corresponds to the index of the new fact (0-based).\n"
    "target_id is the ID of the existing memory to update/delete (null for add/none).\n"
    "importance is the importance score (1-10) for the resulting memory."
)

BATCH_CONFLICT_RESOLUTION_LINKED_PROMPT = (
    "New facts to process:\n{new_facts}\n\n"
    "Linked memories by fact:\n{linked_memories}\n\n"
    "Existing memories:\n{existing_memories}\n\n"
    "For each new fact, decide the action. target_id must refer to one of that "
    "fact's linked_memory_ids. If several linked memories could work, use the "
    "first linked memory ID listed as the cascade tiebreaker. Respond with this "
    "exact JSON format:\n"
    '{{\"actions\": [{{\"fact_index\": 0, \"action\": \"add\", '
    '\"target_id\": null, \"importance\": 7}}, '
    '{{\"fact_index\": 1, \"action\": \"update\", '
    '\"target_id\": 3, \"importance\": 8}}, '
    '{{\"fact_index\": 2, \"action\": \"none\", '
    '\"target_id\": null, \"importance\": 5}}, ...]}}\n\n'
    "fact_index corresponds to the index of the new fact (0-based).\n"
    "linked_memory_ids is the ordered list of candidate IDs for that fact.\n"
    "target_id is the ID of the existing memory to update/delete (null for add/none).\n"
    "importance is the importance score (1-10) for the resulting memory."
)
