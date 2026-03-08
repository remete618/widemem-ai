#!/usr/bin/env python3
"""End-to-end test using real OpenAI LLM + embeddings.

Requires OPENAI_API_KEY environment variable.

Usage:
    OPENAI_API_KEY=sk-... python scripts/e2e_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    from widemem import WideMemory, MemoryConfig
    from widemem.core.types import (
        LLMConfig,
        EmbeddingConfig,
        VectorStoreConfig,
        YMYLConfig,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o-mini", api_key=api_key),
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small", api_key=api_key, dimensions=1536),
            vector_store=VectorStoreConfig(provider="faiss", path=f"{tmpdir}/vectors"),
            history_db_path=f"{tmpdir}/history.db",
            ymyl=YMYLConfig(enabled=True),
        )

        with WideMemory(config) as memory:
            # --- Test 1: Basic add + search ---
            print("Test 1: Add and search...")
            result = memory.add(
                "My name is Alice. I live in Berlin and I work as a software engineer at Google.",
                user_id="alice",
            )
            assert len(result.memories) > 0, "Expected at least 1 memory"
            print(f"  Extracted {len(result.memories)} memories:")
            for m in result.memories:
                print(f"    - {m.content} (importance: {m.importance})")

            results = memory.search("where does alice live", user_id="alice", top_k=5)
            assert len(results) > 0, "Expected search results"
            print(f"  Search 'where does alice live': {len(results)} results")
            for r in results:
                print(f"    - {r.memory.content} (score: {r.final_score:.3f})")

            # --- Test 2: Update via contradiction ---
            print("\nTest 2: Contradiction handling...")
            result2 = memory.add(
                "Alice just moved to Paris. She no longer lives in Berlin.",
                user_id="alice",
            )
            print(f"  Extracted {len(result2.memories)} memories")
            for m in result2.memories:
                print(f"    - {m.content} (importance: {m.importance})")

            results2 = memory.search("where does alice live", user_id="alice", top_k=3)
            print(f"  Search after update: {len(results2)} results")
            for r in results2:
                print(f"    - {r.memory.content} (score: {r.final_score:.3f})")

            # --- Test 3: YMYL detection ---
            print("\nTest 3: YMYL content...")
            result3 = memory.add(
                "Alice has been diagnosed with diabetes. Her blood type is O+. She takes metformin daily.",
                user_id="alice",
            )
            print(f"  Extracted {len(result3.memories)} memories:")
            for m in result3.memories:
                print(f"    - {m.content} (importance: {m.importance})")

            high_importance = [m for m in result3.memories if m.importance >= 6.0]
            print(f"  High importance (>=6.0): {len(high_importance)} of {len(result3.memories)}")

            # --- Test 4: History ---
            print("\nTest 4: History audit trail...")
            if result.memories:
                history = memory.get_history(result.memories[0].id)
                print(f"  History for first memory: {len(history)} entries")
                for h in history:
                    print(f"    - {h.action.value}: {h.new_content or h.old_content}")

            # --- Test 5: Get and delete ---
            print("\nTest 5: Get and delete...")
            if result3.memories:
                mem_id = result3.memories[0].id
                got = memory.get(mem_id)
                assert got is not None, "Expected to get memory by id"
                print(f"  Get: {got.content}")

                memory.delete(mem_id)
                got_after = memory.get(mem_id)
                assert got_after is None, "Expected memory to be deleted"
                print("  Delete: confirmed deleted")

            # --- Test 6: Persistence ---
            print("\nTest 6: Persistence...")
            count_before = len(memory.search("alice", user_id="alice", top_k=50))

        with WideMemory(config) as memory2:
            count_after = len(memory2.search("alice", user_id="alice", top_k=50))
            print(f"  Before close: {count_before} results, after reload: {count_after} results")
            assert count_after > 0, "Expected memories to persist after reload"

    print("\n--- ALL E2E TESTS PASSED ---")


if __name__ == "__main__":
    main()
