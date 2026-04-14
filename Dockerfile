FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY widemem/ widemem/

RUN pip install --no-cache-dir ".[mcp]"

ENV WIDEMEM_DATA_PATH=/data
ENV WIDEMEM_LLM_PROVIDER=ollama
ENV WIDEMEM_EMBEDDING_PROVIDER=sentence-transformers

VOLUME /data

ENTRYPOINT ["python", "-m", "widemem.mcp_server"]
