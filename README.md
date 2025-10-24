# Wikipedia OCR Dataset Builder

This project turns cleaned Wikipedia articles into an OCR-ready dataset. For each requested language the tool fetches random
articles, sanitises their HTML, renders paginated PDFs, converts each page into a PNG image, and stores aligned text
transcriptions for use in OCR training or evaluation pipelines.

## Features

- Fetches random articles from multiple Wikipedia language editions with an identifying User-Agent header
- Cleans the HTML with BeautifulSoup, keeping only headings, paragraphs, block quotes, and list items
- Renders the cleaned article into a paginated PDF using WeasyPrint
- Converts each PDF page into a PNG image at 300 DPI and extracts page-aligned ground truth via `pdftotext`
- Writes per-document metadata (`meta.json`) alongside the rendered assets in a predictable directory hierarchy

## Requirements

- Python **3.13.9**
- `pip` (bundled with official Python installers)
- Poppler utilities (`pdftotext`, `pdftoppm`) on the system path
- Cairo and Pango libraries required by WeasyPrint (`libcairo2`, `libpango-1.0-0`, `libpangoft2-1.0-0`, `libgdk-pixbuf-2.0-0`)

On Debian/Ubuntu you can install the native dependencies with:

```bash
sudo apt-get update
sudo apt-get install python3.13 python3.13-venv poppler-utils libcairo2 libpango-1.0-0 libpangoft2-1.0-0 libgdk-pixbuf-2.0-0 fonts-dejavu
```

On macOS with Homebrew:

```bash
brew install pango gdk-pixbuf cairo libffi
```

## Usage

Create a virtual environment, install the dependencies, and run the generator:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python prepare_wiki_ocr_dataset.py --langs kk,ru,tg --num_docs 50 --output_dir ./wiki_ocr_dataset
```

Key command-line options:

- `--langs`: Comma-separated language codes (e.g. `kk,ru,tg`).
- `--num_docs`: Number of articles to collect for each language.
- `--output_dir`: Destination directory for the dataset structure.
- `--user_agent`: Identifying User-Agent string for Wikimedia API access (defaults to a generic project identifier).
- `--max_attempts`: Maximum number of fetch attempts per language before giving up.

> **Reminder:** Wikimedia's REST API requires a descriptive User-Agent that includes a way to contact you. Override the default
> value with something like `--user_agent "my-project/0.1 (contact@example.com)"` for large crawls. See the
> [User-Agent policy](https://meta.wikimedia.org/wiki/User-Agent_policy) for details.

### Output structure

The dataset is organised as follows:

```
wiki_ocr_dataset/
  ru/
    ru_001/
      ru_001.pdf
      images/
        page_001.png
        page_002.png
      text_gt/
        page_001.txt
        page_002.txt
      meta.json
```

Each `meta.json` file contains metadata similar to:

```json
{
  "lang": "ru",
  "title": "Бенедиктов,_Владимир_Валентинович",
  "source_url": "https://ru.wikipedia.org/wiki/Бенедиктов,_Владимир_Валентинович",
  "num_pages": 3,
  "text_length": 3895,
  "pages": [
    {"image": "images/page_001.png", "text": "text_gt/page_001.txt"},
    {"image": "images/page_002.png", "text": "text_gt/page_002.txt"}
  ]
}
```

## Docker

A Dockerfile is provided to run the pipeline in a containerised environment with all binary dependencies pre-installed:

```bash
docker build -t wiki-ocr-dataset .
docker run --rm -v "$PWD/output":/data wiki-ocr-dataset \
  --langs kk,ru,tg --num_docs 50 --output_dir /data
```

The container image is based on Python 3.13.9 and installs the Python dependencies from `requirements.txt` using `pip`.
