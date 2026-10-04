"""Use widemem as the retrieval backend in a LangChain RAG chain.

    pip install "widemem-ai[langchain,local]"
    ollama pull llama3.1:8b
    python examples/langchain_retriever.py

The retriever is an ordinary LangChain `BaseRetriever`, so it drops into any
chain that takes one. What widemem adds over a plain vector store is on the
write side: `add()` extracts facts, resolves contradictions against what is
already stored, and scores importance, so the retriever sees settled facts
rather than raw transcript chunks.
"""

from __future__ import annotations

from widemem import MemoryConfig, WideMemory
from widemem.core.types import RetrievalConfidence
from widemem.integrations.langchain import WidememRetriever


def main() -> None:
    memory = WideMemory(config=MemoryConfig())

    # The write side does the work. Note the contradiction: the second call
    # supersedes the first rather than storing both.
    memory.add("I live in Berlin and work as a backend engineer.", user_id="alice")
    memory.add("Actually I moved to Vienna last month.", user_id="alice")
    memory.add("I am allergic to penicillin.", user_id="alice")

    retriever = WidememRetriever(
        memory=memory,
        user_id="alice",
        top_k=5,
        # Return nothing rather than noise, so the chain can branch on an empty
        # list. LOW suits the local embedder, where short correct facts often
        # score LOW; with OpenAI embeddings, MODERATE is the stricter choice.
        min_confidence=RetrievalConfidence.LOW,
    )

    for question in ("Where does she live?", "Any drug allergies?"):
        documents = retriever.invoke(question)
        print(f"\n{question}")
        if not documents:
            print("  no confident match; the chain should say it does not know")
            continue
        for doc in documents:
            score = doc.metadata["final_score"]
            print(f"  [{score:.3f}] {doc.page_content}")

    # Dropping it into a real chain (needs a chat model, e.g. langchain-openai):
    #
    #     from langchain.chains import create_retrieval_chain
    #     chain = create_retrieval_chain(retriever, question_answer_chain)
    #     chain.invoke({"input": "Where does she live?"})

    memory.close()


if __name__ == "__main__":
    main()
