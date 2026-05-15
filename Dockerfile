FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY widemem/ widemem/

RUN pip install --no-cache-dir ".[server,anthropic,faiss,bm25]"

ENV WIDEMEM_DATA_PATH=/tmp/widemem-data
ENV WIDEMEM_LLM_PROVIDER=ollama
ENV WIDEMEM_EMBEDDING_PROVIDER=sentence-transformers

CMD sh -c "WIDEMEM_PORT=${PORT:-8000} WIDEMEM_HOST=0.0.0.0 python -m widemem.server"
