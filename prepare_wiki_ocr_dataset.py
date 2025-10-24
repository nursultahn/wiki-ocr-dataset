"""Build an OCR dataset from cleaned Wikipedia articles rendered to PDF."""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
from tqdm import tqdm


def ensure_weasyprint_runtime() -> None:
    """Validate that the native dependencies required by WeasyPrint are present."""

    required_libs = {
        "Cairo": ("cairo", "libcairo", "libcairo-2"),
        "Pango": ("pango-1.0", "libpango-1.0-0"),
        "PangoFT2": ("pangoft2-1.0", "libpangoft2-1.0-0"),
        "GDK-PixBuf": ("gdk_pixbuf-2.0", "libgdk_pixbuf-2.0-0"),
        "GObject": ("gobject-2.0", "libgobject-2.0-0"),
    }

    missing = [
        name
        for name, candidates in required_libs.items()
        if not any(find_library(candidate) for candidate in candidates)
    ]

    if missing:
        formatted = ", ".join(sorted(missing))
        raise RuntimeError(
            "Missing native libraries for WeasyPrint: "
            f"{formatted}.\n"
            "Install the packages listed in the README (for example, on macOS: "
            "`brew install pango gdk-pixbuf cairo libffi`)."
        )


ensure_weasyprint_runtime()

from weasyprint import HTML

LOGGER = logging.getLogger(__name__)

RANDOM_TITLE_URL = "https://{lang}.wikipedia.org/api/rest_v1/page/random/title"
ARTICLE_HTML_URL = "https://{lang}.wikipedia.org/api/rest_v1/page/html/{title}"
WIKI_PAGE_URL = "https://{lang}.wikipedia.org/wiki/{title}"

DEFAULT_USER_AGENT = "wiki-ocr-dataset/1.0 (+https://github.com/example/wiki-ocr-dataset)"
DEFAULT_LANGS = ("ru",)
DEFAULT_NUM_DOCS = 5
DEFAULT_OUTPUT_DIR = Path("./wiki_ocr_dataset")

REMOVABLE_TAGS = {
    "table",
    "sup",
    "style",
    "script",
    "header",
    "footer",
    "nav",
    "figure",
    "img",
    "a",
    "math",
    "cite",
}
ALLOWED_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"}


@dataclass
class ArticleData:
    """Container for prepared article assets."""

    lang: str
    title: str
    html_content: str
    plain_text: str


def parse_langs(value: str) -> List[str]:
    langs = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not langs:
        raise argparse.ArgumentTypeError("--langs must include at least one language code")
    return langs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Wikipedia-based OCR dataset rendered from cleaned HTML.",
    )
    parser.add_argument(
        "--langs",
        type=parse_langs,
        default=list(DEFAULT_LANGS),
        help="Comma separated list of language codes (e.g. kk,ru,tg)",
    )
    parser.add_argument(
        "--num_docs",
        type=int,
        default=DEFAULT_NUM_DOCS,
        help="Number of articles to collect per language.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the dataset will be stored.",
    )
    parser.add_argument(
        "--user_agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="User-Agent header for Wikimedia API requests.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=200,
        help="Maximum fetch attempts per language before giving up.",
    )

    args = parser.parse_args(argv)
    if args.num_docs < 1:
        parser.error("--num_docs must be at least 1")
    if args.max_attempts < args.num_docs:
        parser.error("--max_attempts must be >= --num_docs")
    if not args.user_agent.strip():
        parser.error("--user_agent must not be empty")
    return args


def clean_text(text: str) -> str:
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_random_title(session: requests.Session, lang: str) -> Optional[str]:
    url = RANDOM_TITLE_URL.format(lang=lang)
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        LOGGER.warning("[%s] Failed to fetch random title: %s", lang, exc)
        return None

    items = data.get("items")
    if not items:
        LOGGER.warning("[%s] Random title response missing items: %s", lang, data)
        return None
    title = items[0].get("title")
    if not title:
        LOGGER.warning("[%s] Random title entry missing title: %s", lang, items[0])
        return None
    return title


def fetch_article_html(session: requests.Session, lang: str, title: str) -> Optional[str]:
    encoded_title = quote(title, safe="")
    url = ARTICLE_HTML_URL.format(lang=lang, title=encoded_title)
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        LOGGER.warning("[%s] Failed to fetch HTML for '%s': %s", lang, title, exc)
        return None


def sanitize_html(html_content: str) -> ArticleData:
    soup = BeautifulSoup(html_content, "html.parser")
    body = soup.body or soup

    for tag_name in REMOVABLE_TAGS:
        for element in body.find_all(tag_name):
            element.decompose()

    blocks: List[str] = []
    texts: List[str] = []

    for element in body.find_all(True):
        if element.name not in ALLOWED_TAGS:
            continue
        text_value = element.get_text(" ", strip=True)
        if not text_value:
            continue
        if element.name == "li":
            fragment = f"<p>• {text_value}</p>"
        else:
            fragment = f"<{element.name}>{text_value}</{element.name}>"
        blocks.append(fragment)
        texts.append(text_value)

    combined_html = "\n".join(blocks)
    cleaned_text = clean_text(" ".join(texts))

    html_output = (
        "<html><head><meta charset='utf-8'>"
        "<style>body{font-family:'DejaVu Sans',sans-serif;font-size:12pt;line-height:1.4;margin:1in;}"
        "p{margin:0 0 0.8em 0;}"
        "blockquote{margin:0.8em 1.4em;font-style:italic;}"
        "</style></head><body>"
        f"{combined_html}"
        "</body></html>"
    )

    return ArticleData(lang="", title="", html_content=html_output, plain_text=cleaned_text)


def prepare_article(session: requests.Session, lang: str, title: str) -> Optional[ArticleData]:
    html_raw = fetch_article_html(session, lang, title)
    if not html_raw:
        return None

    article = sanitize_html(html_raw)
    if not article.plain_text:
        LOGGER.info("[%s] Skipping '%s' because cleaned text is empty", lang, title)
        return None

    article.lang = lang
    article.title = title
    return article


def write_pdf(html_content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_content).write_pdf(str(output_path))


def convert_pdf_to_images(pdf_path: Path, images_dir: Path) -> List[Path]:
    images_dir.mkdir(parents=True, exist_ok=True)
    images = convert_from_path(str(pdf_path), dpi=300)
    saved_paths: List[Path] = []
    for index, image in enumerate(images, start=1):
        image_path = images_dir / f"page_{index:03d}.png"
        image.save(image_path, "PNG")
        saved_paths.append(image_path)
    return saved_paths


def extract_text_per_page(pdf_path: Path, text_dir: Path, num_pages: int) -> List[Path]:
    text_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("pdftotext not found. Install poppler-utils/poppler.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pdftotext failed: {exc.stderr.decode(errors='ignore').strip()}") from exc

    decoded = result.stdout.decode("utf-8", errors="ignore")
    pages_raw = decoded.split("\f")
    paths: List[Path] = []
    for idx in range(1, num_pages + 1):
        raw_text = pages_raw[idx - 1] if idx - 1 < len(pages_raw) else ""
        cleaned = clean_text(raw_text)
        text_path = text_dir / f"page_{idx:03d}.txt"
        text_path.write_text(cleaned, encoding="utf-8")
        paths.append(text_path)
    return paths


def build_metadata(lang: str, title: str, images: List[Path], texts: List[Path]) -> dict:
    encoded_title = quote(title.replace(" ", "_"), safe="")
    source_url = WIKI_PAGE_URL.format(lang=lang, title=encoded_title)
    text_length = sum(len(path.read_text(encoding="utf-8")) for path in texts)
    pages = []
    for image_path, text_path in zip(images, texts):
        pages.append(
            {
                "image": f"images/{image_path.name}",
                "text": f"text_gt/{text_path.name}",
            }
        )
    metadata = {
        "lang": lang,
        "title": title,
        "source_url": source_url,
        "num_pages": len(images),
        "text_length": text_length,
        "pages": pages,
    }
    return metadata


def process_language(
    session: requests.Session,
    lang: str,
    num_docs: int,
    output_dir: Path,
    max_attempts: int,
) -> None:
    lang_dir = output_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    collected = 0
    attempts = 0
    progress = tqdm(total=num_docs, desc=f"[{lang}]", unit="doc")

    while collected < num_docs and attempts < max_attempts:
        attempts += 1
        title = fetch_random_title(session, lang)
        if not title:
            continue

        article = prepare_article(session, lang, title)
        if not article:
            continue

        collected += 1
        doc_id = f"{lang}_{collected:03d}"
        doc_dir = lang_dir / doc_id
        pdf_path = doc_dir / f"{doc_id}.pdf"
        images_dir = doc_dir / "images"
        text_dir = doc_dir / "text_gt"

        try:
            write_pdf(article.html_content, pdf_path)
            image_paths = convert_pdf_to_images(pdf_path, images_dir)
            text_paths = extract_text_per_page(pdf_path, text_dir, len(image_paths))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("[%s] Failed to render assets for '%s': %s", lang, title, exc)
            # Clean partially written doc directory
            cleanup_paths([doc_dir])
            collected -= 1
            continue

        metadata = build_metadata(lang, title, image_paths, text_paths)
        meta_path = doc_dir / "meta.json"
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        LOGGER.info(
            "[%s] Saved article '%s' with %d pages to %s",
            lang,
            title,
            len(image_paths),
            doc_dir,
        )
        progress.update(1)

    progress.close()
    if collected < num_docs:
        LOGGER.warning(
            "[%s] Collected %d/%d documents after %d attempts.",
            lang,
            collected,
            num_docs,
            attempts,
        )


def cleanup_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.is_dir():
            for child in path.iterdir():
                if child.is_dir():
                    cleanup_paths([child])
                else:
                    child.unlink(missing_ok=True)
            path.rmdir()
        elif path.exists():
            path.unlink(missing_ok=True)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging()

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.langs:
        process_language(
            session=session,
            lang=lang,
            num_docs=args.num_docs,
            output_dir=args.output_dir,
            max_attempts=args.max_attempts,
        )


if __name__ == "__main__":
    main()
