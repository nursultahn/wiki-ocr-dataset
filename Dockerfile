FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --frozen

COPY prepare_wiki_ocr_dataset.py ./

ENTRYPOINT ["uv", "run", "python", "prepare_wiki_ocr_dataset.py"]
CMD ["--help"]
