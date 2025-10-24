"""Utilities to build a lightweight OCR dataset from Wikipedia PDF exports."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import ssl
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import aiohttp
import certifi
from aiohttp import ClientResponseError
from pdf2image import convert_from_path
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

API_TITLE_URL = "https://{lang}.wikipedia.org/api/rest_v1/page/random/title"
API_PDF_URL = "https://{lang}.wikipedia.org/api/rest_v1/page/pdf/{title}"
DEFAULT_LANGS = ("ru",)
DEFAULT_NUM_DOCS = 5
DEFAULT_IMAGES_PER_DOC = 5
DEFAULT_MIN_TEXT_LEN = 300
DEFAULT_USER_AGENT = "wiki-ocr-dataset/1.0 (+https://github.com/example/wiki-ocr-dataset)"
MAX_ATTEMPTS_FACTOR = 10
DOWNLOAD_CONCURRENCY = 4
CHUNK_SIZE = 1 << 14


class DatasetBuildError(Exception):
    """Base exception for dataset preparation errors."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download random Wikipedia PDFs, convert to images, and export OCR metadata",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=list(DEFAULT_LANGS),
        help="Languages to download from Wikipedia (e.g. kk ru tg)",
    )
    parser.add_argument(
        "--num_docs",
        type=int,
        default=DEFAULT_NUM_DOCS,
        help="Maximum number of valid PDF documents to fetch per language",
    )
    parser.add_argument(
        "--images_per_doc",
        type=int,
        default=DEFAULT_IMAGES_PER_DOC,
        help="Maximum number of images to keep per PDF",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./wiki_ocr_dataset"),
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--min_text_len",
        type=int,
        default=DEFAULT_MIN_TEXT_LEN,
        help="Minimum character length for extracted text",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=DOWNLOAD_CONCURRENCY,
        help="Maximum number of concurrent PDF downloads",
    )
    parser.add_argument(
        "--user_agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="User-Agent header to include with Wikipedia API requests",
    )

    args = parser.parse_args(argv)
    if args.num_docs < 1:
        parser.error("--num_docs must be positive")
    if args.images_per_doc < 1:
        parser.error("--images_per_doc must be positive")
    if args.min_text_len < 1:
        parser.error("--min_text_len must be positive")
    if args.max_concurrency < 1:
        parser.error("--max_concurrency must be positive")
    if not args.user_agent.strip():
        parser.error("--user_agent must not be empty")
    return args


async def fetch_random_title(session: aiohttp.ClientSession, lang: str) -> str:
    url = API_TITLE_URL.format(lang=lang)
    async with session.get(url) as response:
        response.raise_for_status()
        payload = await response.json()
    try:
        items = payload["items"]
        if not items:
            raise DatasetBuildError("Empty response from Wikipedia random title API")
        return items[0]["title"]
    except (KeyError, IndexError) as exc:
        raise DatasetBuildError(f"Unexpected response format: {payload}") from exc


async def download_pdf(
    session: aiohttp.ClientSession,
    lang: str,
    title: str,
    destination: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    from urllib.parse import quote

    encoded_title = quote(title, safe="")
    url = API_PDF_URL.format(lang=lang, title=encoded_title)
    async with semaphore:
        async with session.get(url) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    handle.write(chunk)


def convert_pdf_to_images(pdf_path: Path, image_dir: Path, doc_id: str, limit: int) -> List[Path]:
    pages = convert_from_path(str(pdf_path))
    saved_images: List[Path] = []
    for index, page in enumerate(pages[:limit], start=1):
        image_path = image_dir / f"{doc_id}_page_{index:03d}.png"
        page.save(image_path, "PNG")
        saved_images.append(image_path)
    return saved_images


def extract_text(pdf_path: Path, text_dir: Path, doc_id: str) -> Tuple[Path, int]:
    text_path = text_dir / f"{doc_id}.txt"
    try:
        subprocess.run(
            ["pdftotext", str(pdf_path), str(text_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise DatasetBuildError(
            "pdftotext not found. Please install poppler-utils/poppler"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise DatasetBuildError(f"pdftotext failed: {exc.stderr.decode().strip()}") from exc

    text_content = text_path.read_text(encoding="utf-8", errors="ignore")
    return text_path, len(text_content)


def ensure_directories(base: Path, lang: str) -> Dict[str, Path]:
    lang_dir = base / lang
    pdf_dir = lang_dir / "pdf"
    img_dir = lang_dir / "images"
    text_dir = lang_dir / "text_gt"
    for directory in (lang_dir, pdf_dir, img_dir, text_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {"base": lang_dir, "pdf": pdf_dir, "images": img_dir, "text": text_dir}


def cleanup(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)


async def process_language(
    session: aiohttp.ClientSession,
    lang: str,
    args: argparse.Namespace,
) -> Dict[str, int]:
    directories = ensure_directories(args.out_dir, lang)
    metadata_path = directories["base"] / "metadata.jsonl"
    rng = random.Random(f"{args.seed}-{lang}")
    semaphore = asyncio.Semaphore(args.max_concurrency)
    summaries = {"docs": 0, "images": 0}
    metadata_entries: List[Dict[str, object]] = []

    attempts = 0
    progress = tqdm(total=args.num_docs, desc=f"[{lang}]", unit="doc")
    try:
        while summaries["docs"] < args.num_docs and attempts < args.num_docs * MAX_ATTEMPTS_FACTOR:
            attempts += 1
            try:
                title = await fetch_random_title(session, lang)
            except (ClientResponseError, DatasetBuildError, aiohttp.ClientError) as exc:
                LOGGER.warning("[%s] Failed to fetch random title: %s", lang, exc)
                continue

            doc_id = f"{lang}_{rng.getrandbits(32):08x}_{attempts:05d}"
            pdf_path = directories["pdf"] / f"{doc_id}.pdf"

            try:
                await download_pdf(session, lang, title, pdf_path, semaphore)
            except (ClientResponseError, aiohttp.ClientError) as exc:
                LOGGER.warning("[%s] Failed to download PDF for '%s': %s", lang, title, exc)
                cleanup([pdf_path])
                continue

            try:
                images = await asyncio.to_thread(
                    convert_pdf_to_images,
                    pdf_path,
                    directories["images"],
                    doc_id,
                    args.images_per_doc,
                )
            except Exception as exc:  # noqa: BLE001 broad but ensures cleanup
                LOGGER.warning("[%s] Failed to convert PDF to images: %s", lang, exc)
                cleanup([pdf_path])
                continue

            if not images:
                LOGGER.info("[%s] No images generated for '%s'", lang, title)
                cleanup([pdf_path])
                continue

            try:
                text_path, text_length = await asyncio.to_thread(
                    extract_text,
                    pdf_path,
                    directories["text"],
                    doc_id,
                )
            except DatasetBuildError as exc:
                LOGGER.warning("[%s] Failed to extract text: %s", lang, exc)
                cleanup([pdf_path, *images])
                continue

            if text_length < args.min_text_len:
                LOGGER.info(
                    "[%s] Skipping '%s' due to short text (%d characters)",
                    lang,
                    title,
                    text_length,
                )
                cleanup([pdf_path, text_path, *images])
                continue

            metadata_entry = {
                "lang": lang,
                "pdf_path": os.path.relpath(pdf_path, args.out_dir),
                "image_paths": [os.path.relpath(path, args.out_dir) for path in images],
                "text_path": os.path.relpath(text_path, args.out_dir),
                "num_pages": len(images),
                "text_length": text_length,
                "title": title,
            }
            metadata_entries.append(metadata_entry)
            summaries["docs"] += 1
            summaries["images"] += len(images)
            progress.update(1)
    finally:
        progress.close()

    if summaries["docs"]:
        with metadata_path.open("w", encoding="utf-8") as handle:
            for entry in metadata_entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return summaries


async def build_dataset(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    timeout = aiohttp.ClientTimeout(total=180)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(
        limit_per_host=args.max_concurrency,
        ssl=ssl_context,
    )
    summary: Dict[str, Dict[str, int]] = {}
    headers = {"User-Agent": args.user_agent}
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers=headers,
    ) as session:
        for lang in args.langs:
            LOGGER.info("Processing language '%s'", lang)
            summary[lang] = await process_language(session, lang, args)
    return summary


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def print_summary(summary: Dict[str, Dict[str, int]]) -> None:
    total_docs = sum(stats.get("docs", 0) for stats in summary.values())
    total_images = sum(stats.get("images", 0) for stats in summary.values())
    for lang, stats in summary.items():
        print(f"[{lang}] {stats.get('docs', 0)} документов, {stats.get('images', 0)} изображений")
    print(f"Всего: {total_docs} PDF, {total_images} изображений")


async def async_main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging()
    summary = await build_dataset(args)
    print_summary(summary)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
