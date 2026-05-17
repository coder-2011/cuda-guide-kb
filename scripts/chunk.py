#!/usr/bin/env python3
"""Chunk the CUDA Programming Guide Markdown into page-aware search records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC = ROOT / "docs" / "cuda-programming-guide.md"
DEFAULT_OUT = ROOT / "data" / "chunks.jsonl"
DEFAULT_SECTIONS = ROOT / "data" / "sections.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--words", type=int, default=850)
    parser.add_argument("--overlap", type=int, default=120)
    return parser.parse_args()


def parse_outline(markdown: str) -> list[dict]:
    outline_match = re.search(
        r"## PDF Outline\n\n(?P<outline>.*?)\n## Extracted Pages",
        markdown,
        flags=re.S,
    )
    if not outline_match:
        return []

    entries: list[dict] = []
    stack: list[str] = []
    for line in outline_match.group("outline").splitlines():
        match = re.match(r"^(?P<indent>\s*)- (?P<title>.+?) \(page (?P<page>\d+)\)$", line)
        if not match:
            continue
        level = len(match.group("indent")) // 2
        title = match.group("title").strip()
        page = int(match.group("page"))
        stack = stack[:level]
        stack.append(title)
        entries.append(
            {
                "title": title,
                "page": page,
                "level": level + 1,
                "heading_path": list(stack),
            }
        )
    return entries


def page_heading(outline: list[dict], page: int) -> list[str]:
    active: list[str] = []
    for entry in outline:
        if entry["page"] <= page:
            active = entry["heading_path"]
        else:
            break
    return active


def parse_pages(markdown: str) -> list[tuple[int, str]]:
    page_pattern = re.compile(
        r"(?:^|\n)---\n\n## Page (?P<page>\d+)\n\n(?P<text>.*?)(?=\n---\n\n## Page \d+\n\n|\Z)",
        flags=re.S,
    )
    return [(int(match.group("page")), match.group("text").strip()) for match in page_pattern.finditer(markdown)]


def normalize_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_words(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if not words:
        return []
    chunks = []
    start = 0
    step = max(size - overlap, 1)
    while start < len(words):
        part = words[start : start + size]
        if part:
            chunks.append(part)
        if start + size >= len(words):
            break
        start += step
    return chunks


def make_chunks(markdown: str, words_per_chunk: int, overlap: int) -> tuple[list[dict], list[dict]]:
    outline = parse_outline(markdown)
    pages = parse_pages(markdown)
    chunks: list[dict] = []

    for page, text in pages:
        clean = normalize_text(text)
        heading_path = page_heading(outline, page)
        heading = " > ".join(heading_path) if heading_path else "CUDA Programming Guide"
        page_words = clean.split()

        for chunk_index, word_chunk in enumerate(chunk_words(page_words, words_per_chunk, overlap)):
            chunk_text = " ".join(word_chunk)
            chunks.append(
                {
                    "id": f"page-{page:04d}-chunk-{chunk_index:02d}",
                    "source": "docs/cuda-programming-guide.md",
                    "pdf": "docs/cuda-programming-guide.pdf",
                    "source_url": "https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf",
                    "page": page,
                    "heading_path": heading_path,
                    "heading": heading,
                    "text": chunk_text,
                    "search_text": f"{heading}\n{chunk_text}",
                }
            )

    sections = [
        {
            "id": f"section-{index:04d}",
            "title": entry["title"],
            "page": entry["page"],
            "level": entry["level"],
            "heading_path": entry["heading_path"],
            "search_text": " > ".join(entry["heading_path"]),
        }
        for index, entry in enumerate(outline)
    ]
    return chunks, sections


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    markdown = args.input.read_text(encoding="utf-8")
    chunks, sections = make_chunks(markdown, args.words, args.overlap)
    write_jsonl(args.output, chunks)
    write_jsonl(args.sections, sections)
    print(f"wrote {len(chunks)} chunks to {args.output}")
    print(f"wrote {len(sections)} section records to {args.sections}")


if __name__ == "__main__":
    main()
