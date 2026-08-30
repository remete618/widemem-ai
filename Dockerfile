FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY widemem/ widemem/

# Must cover every provider the ENV defaults below select, or the
# container raises ImportError on its first request.
RUN pip install --no-cache-dir ".[server,anthropic,faiss,bm25,ollama]"

# Fail the build, not the first request, if a default provider is missing.
RUN python -c "import widemem.server, ollama, faiss"

# /tmp is cleared on restart on most runtimes, which silently emptied
# the memory store between runs. Keep state on a declared volume.
ENV WIDEMEM_DATA_PATH=/data
ENV WIDEMEM_LLM_PROVIDER=ollama
ENV WIDEMEM_EMBEDDING_PROVIDER=ollama

RUN useradd --create-home --uid 10001 widemem \
    && mkdir -p /data \
    && chown -R widemem:widemem /data /app

VOLUME ["/data"]
USER widemem

CMD sh -c "WIDEMEM_PORT=${PORT:-8000} WIDEMEM_HOST=0.0.0.0 python -m widemem.server"
