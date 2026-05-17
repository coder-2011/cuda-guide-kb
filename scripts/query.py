#!/usr/bin/env python3
"""Query the local CUDA Programming Guide index."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
from scipy import sparse
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "index" / "cuda-guide-tfidf.joblib"

TOPIC_EXPANSIONS = {
    "memory": "global memory shared memory local memory constant memory texture memory coalescing cache l1 l2 bandwidth bank conflicts unified memory cudaMemcpyAsync async copy",
    "performance": "occupancy latency throughput bandwidth coalescing profiling Nsight Compute warp stalls memory throughput compute utilization launch overhead",
    "optimize": "optimization occupancy coalescing shared memory registers latency hiding warp divergence instruction throughput memory bandwidth",
    "optimization": "occupancy coalescing shared memory registers latency hiding warp divergence instruction throughput memory bandwidth launch bounds",
    "kernel": "kernel launch grid block thread warp occupancy registers shared memory synchronization",
    "model": "programming model execution model memory model grid block thread warp SIMT memory hierarchy synchronization streams",
    "occupancy": "occupancy active warps resident warps resident blocks multiprocessor registers shared memory launch bounds cudaOccupancyMaxActiveBlocksPerMultiprocessor",
    "memory hierarchy": "global memory shared memory local memory constant memory texture memory registers cache l1 l2 coalescing bank conflicts",
    "synchronization": "__syncthreads cooperative groups barriers fences memory ordering atomics",
    "threads": "threadIdx blockIdx blockDim gridDim warp SIMT divergence occupancy",
    "streams": "stream concurrency cudaStream cudaMemcpyAsync events synchronization overlap",
    "unified memory": "managed memory migration page fault prefetch cudaMallocManaged cudaMemPrefetchAsync",
    "broad": "overview programming model memory model execution model performance optimization best practices",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="question to ask")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    return parser.parse_args()


def expand_query(question: str) -> str:
    lowered = question.lower()
    expansions = []
    has_symbol = bool(re.search(r"\b__[A-Za-z0-9_]+\b|cuda[A-Z][A-Za-z0-9_]+", question))
    if not has_symbol and re.search(r"\b(how|what|why|overview|explain|broad|general|think about)\b", lowered):
        expansions.append(TOPIC_EXPANSIONS["broad"])
    for trigger, expansion in TOPIC_EXPANSIONS.items():
        if trigger in lowered and expansion not in expansions:
            expansions.append(expansion)
    return question + "\n" + "\n".join(expansions)


def excerpt(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def keyword_score(question: str, doc: dict) -> float:
    query_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z_][A-Za-z0-9_:+./-]*", question)
        if len(term) > 2
    }
    if not query_terms:
        return 0.0
    haystack = f"{doc.get('heading', '')} {doc.get('text', '')}".lower()
    hits = sum(1 for term in query_terms if term in haystack)
    score = min(hits / max(len(query_terms), 1), 1.0) * 0.08
    for symbol in re.findall(r"\b__[A-Za-z0-9_]+\b|cuda[A-Z][A-Za-z0-9_]+", question):
        if symbol.lower() in haystack:
            score += 0.3
            if symbol.lower() in haystack[:1200]:
                score += 0.12
            if any(term in haystack for term in ("semantics", "synchronization functions", "intrinsics coordinate")):
                score += 0.08
    return score


def is_broad_question(question: str) -> bool:
    lowered = question.lower()
    has_symbol = bool(re.search(r"\b__[A-Za-z0-9_]+\b|cuda[A-Z][A-Za-z0-9_]+", question))
    return not has_symbol and bool(
        re.search(r"\b(how|why|overview|explain|broad|general|think about|should i)\b", lowered)
    )


def search(index_path: Path, question: str, top_k: int) -> list[dict]:
    index = joblib.load(index_path)
    expanded = expand_query(question)
    word_query = index["word_vectorizer"].transform([expanded])
    char_query = index["char_vectorizer"].transform([expanded])
    query_matrix = normalize(sparse.hstack([word_query, char_query], format="csr"), copy=False)
    scores = (index["matrix"] @ query_matrix.T).toarray().ravel()

    ranked = []
    for row, base_score in enumerate(scores):
        doc = index["docs"][row]
        score = float(base_score) + keyword_score(question, doc)
        if doc.get("kind") == "section" and is_broad_question(question):
            score += 0.015
        elif doc.get("kind") == "section":
            score -= 0.03
        ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)

    results = []
    seen_pages = set()
    for score, doc in ranked:
        page = doc.get("page")
        key = (doc.get("kind"), page, doc.get("heading"))
        if key in seen_pages:
            continue
        seen_pages.add(key)
        result = dict(doc)
        result["score"] = round(score, 4)
        result["excerpt"] = excerpt(doc.get("text", ""))
        results.append(result)
        if len(results) >= top_k:
            break
    return results


def print_markdown(question: str, results: list[dict]) -> None:
    print(f"# CUDA Guide Query\n\n**Question:** {question}\n")
    for rank, result in enumerate(results, start=1):
        heading = result.get("heading") or "CUDA Programming Guide"
        page = result.get("page", "?")
        kind = result.get("kind", "chunk")
        print(f"## {rank}. {heading}")
        print(f"- Score: `{result['score']}`")
        print(f"- Type: `{kind}`")
        print(f"- Page: `{page}`")
        print(f"- Source: `{result.get('source', 'docs/cuda-programming-guide.md')}`")
        print("")
        print(result["excerpt"])
        print("")


def main() -> None:
    args = parse_args()
    question = " ".join(args.question).strip()
    results = search(args.index, question, args.top_k)
    if args.json:
        print(json.dumps({"question": question, "results": results}, indent=2, ensure_ascii=False))
    else:
        print_markdown(question, results)


if __name__ == "__main__":
    main()
