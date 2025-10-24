FROM python:3.13.9-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY prepare_wiki_ocr_dataset.py ./

ENTRYPOINT ["python", "prepare_wiki_ocr_dataset.py"]
CMD ["--help"]
