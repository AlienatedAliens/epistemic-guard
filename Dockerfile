FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir . \
    && useradd -m -u 1000 mcp \
    && chown -R mcp:mcp /app

USER mcp

ENTRYPOINT ["epistemic-guard", "mcp-server"]
