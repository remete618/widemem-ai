BATCH_CONFLICT_RESOLUTION_SYSTEM = (
    "You are a memory conflict resolution engine. "
    "Given new facts and existing memories, decide what actions to take.\n\n"
    "Actions:\n"
    "- ADD: The fact contains information not fully captured by any existing memory\n"
    "- UPDATE: The fact makes an existing memory entirely obsolete, either by "
    "contradicting it (the two statements cannot both be true) or by restating "
    "everything it says plus more (provide target memory ID, update_kind, and for "
    "contradictions the contradicted text)\n"
    "- DELETE: An existing memory is now invalid based on the new facts "
    "(provide the target memory ID)\n"
    "- NONE: The fact is already fully captured by one specific existing memory; "
    "provide that memory's ID as target_id\n\n"
    "Rules:\n"
    "- Never discard information: an UPDATE replaces the old memory's text, so "
    "any detail of the old memory not restated in the new fact is destroyed\n"
    "- A contradiction means the old memory and the new fact cannot both be "
    "true (\"moved to San Francisco\" contradicts \"lives in Boston\"). "
    "Different events, different dates, different objects, or extra details "
    "about the same topic are NOT contradictions; use ADD so both are kept\n"
    "- For update with update_kind \"contradiction\", also set \"contradicts\" "
    "to the exact text of the contradicted memory, copied verbatim. If the "
    "memory also holds details that stay true, it is not a full contradiction; "
    "use ADD\n"
    "- Use update_kind \"refinement\" only when the new fact restates the whole "
    "old memory with more detail (\"works as a senior engineer at Google\" "
    "refines \"works at Google\")\n"
    "- Use NONE only when you can name the existing memory that already covers "
    "the fact; if you cannot, use ADD\n"
    "- One action per new fact\n"
    "- Respond with JSON only."
)

BATCH_CONFLICT_RESOLUTION_PROMPT = (
    "New facts to process:\n{new_facts}\n\n"
    "Existing memories:\n{existing_memories}\n\n"
    "For each new fact, decide the action. Respond with this exact JSON format:\n"
    '{{\"actions\": [{{\"fact_index\": 0, \"action\": \"add\", '
    '\"target_id\": null, \"importance\": 7}}, '
    '{{\"fact_index\": 1, \"action\": \"update\", \"target_id\": 3, '
    '\"update_kind\": \"contradiction\", '
    '\"contradicts\": \"exact text of memory 3\", \"importance\": 8}}, '
    '{{\"fact_index\": 2, \"action\": \"none\", '
    '\"target_id\": 5, \"importance\": 5}}, ...]}}\n\n'
    "fact_index corresponds to the index of the new fact (0-based).\n"
    "target_id is the ID of the existing memory to update/delete; for none it is "
    "the ID of the memory that already covers the fact (null for add).\n"
    "update_kind is required for update: \"contradiction\" or \"refinement\".\n"
    "contradicts is required for contradiction updates: the contradicted memory's "
    "text, copied verbatim.\n"
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
    '{{\"fact_index\": 1, \"action\": \"update\", \"target_id\": 3, '
    '\"update_kind\": \"contradiction\", '
    '\"contradicts\": \"exact text of memory 3\", \"importance\": 8}}, '
    '{{\"fact_index\": 2, \"action\": \"none\", '
    '\"target_id\": 4, \"importance\": 5}}, ...]}}\n\n'
    "fact_index corresponds to the index of the new fact (0-based).\n"
    "linked_memory_ids is the ordered list of candidate IDs for that fact.\n"
    "target_id is the ID of the existing memory to update/delete; for none it is "
    "the ID of the memory that already covers the fact (null for add).\n"
    "update_kind is required for update: \"contradiction\" or \"refinement\".\n"
    "contradicts is required for contradiction updates: the contradicted memory's "
    "text, copied verbatim.\n"
    "importance is the importance score (1-10) for the resulting memory."
)
