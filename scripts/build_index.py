#!/usr/bin/env python3
"""Build the local CUDA guide vector index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data" / "chunks.jsonl"
DEFAULT_SECTIONS = ROOT / "data" / "sections.jsonl"
DEFAULT_INDEX = ROOT / "index" / "cuda-guide-tfidf.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    chunks = read_jsonl(args.chunks)
    sections = read_jsonl(args.sections)
    docs = [
        {
            "kind": "chunk",
            **chunk,
        }
        for chunk in chunks
    ] + [
        {
            "kind": "section",
            **section,
            "source": "docs/cuda-programming-guide.md",
            "pdf": "docs/cuda-programming-guide.pdf",
            "source_url": "https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf",
            "heading": " > ".join(section["heading_path"]),
            "text": section["search_text"],
        }
        for section in sections
    ]

    word_vectorizer = TfidfVectorizer(
        input="content",
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=160_000,
        token_pattern=r"(?u)\b[\w][\w\.\+\:#_/-]*\b",
    )
    char_vectorizer = TfidfVectorizer(
        input="content",
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=True,
        max_features=80_000,
    )

    texts = [doc["search_text"] for doc in docs]
    word_matrix = word_vectorizer.fit_transform(texts)
    char_matrix = char_vectorizer.fit_transform(texts)
    matrix = normalize(sparse.hstack([word_matrix, char_matrix], format="csr"), copy=False)

    args.index.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "word_vectorizer": word_vectorizer,
            "char_vectorizer": char_vectorizer,
            "matrix": matrix,
            "docs": docs,
        },
        args.index,
        compress=3,
    )
    print(f"indexed {len(docs)} records to {args.index}")
    print(f"chunks: {len(chunks)}")
    print(f"sections: {len(sections)}")


if __name__ == "__main__":
    main()
