# Wikipedia OCR Dataset Builder

This repository contains a Python utility for building an OCR dataset from random Wikipedia articles exported as PDF files. The tool downloads PDFs, converts them to images, extracts text using `pdftotext`, and stores JSONL metadata describing the generated samples.

## Features

- Asynchronous fetching of random articles from multiple Wikipedia language editions
- Conversion of PDFs into PNG page images using `pdf2image`
- Text extraction through `pdftotext`
- Automatic filtering of documents with short text segments
- JSONL metadata export compatible with common OCR training pipelines

## Requirements

- Python **3.13.9**
- `pip` (comes with the official Python distributions)
- Poppler utilities (`pdftotext`, `pdftoppm`) available on the system path

Install Python and Poppler utilities using your operating system's package manager. On Debian/Ubuntu this looks like:

```bash
sudo apt-get update
sudo apt-get install python3.13 python3.13-venv poppler-utils
```

## Usage

Create a virtual environment, install the dependencies from `requirements.txt`, and run the script:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python prepare_wiki_ocr_dataset.py --langs kk ru tg --num_docs 5 --images_per_doc 3
```

Useful command-line options:

- `--langs`: Space-separated list of language codes (e.g. `kk ru tg`).
- `--num_docs`: Number of valid PDFs to collect per language.
- `--images_per_doc`: Maximum number of images stored for each PDF.
- `--out_dir`: Directory where the dataset will be saved (defaults to `./wiki_ocr_dataset`).
- `--min_text_len`: Minimum text length (in characters) for a document to be considered valid (defaults to `300`).
- `--seed`: Seed used for deterministic file naming.

The script prints a summary similar to:

```
[kk] 10 документов, 85 изображений
[ru] 8 документов, 73 изображений
Всего: 18 PDF, 158 изображений
```

## Docker

A Docker image is provided for reproducible execution without managing Python environments manually:

```bash
docker build -t wiki-ocr-dataset .
docker run --rm -v "$PWD/output":/data wiki-ocr-dataset \
  --langs kk ru tg --num_docs 5 --images_per_doc 3 --out_dir /data
```

The container installs `poppler-utils` so that `pdf2image` and `pdftotext` work out-of-the-box. It runs on Python 3.13.9 as requested and installs the application dependencies from `requirements.txt` using `pip`.
