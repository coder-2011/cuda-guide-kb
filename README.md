# CUDA Programming Guide KB

A small, local knowledge base for querying the NVIDIA CUDA Programming Guide.

This repository includes:

- The downloaded CUDA Programming Guide PDF.
- A Markdown text extraction of the full guide.
- Page-aware chunks and section records.
- A local TF-IDF vector index.
- A minimal query CLI for broad questions and exact CUDA symbol lookup.
- A pure Codex skill in `skills/cuda-programming-guide/`.

No MCP server or database daemon is required.

## Quick Start

Install the small Python dependency set:

```bash
python3 -m pip install -r requirements.txt
```

Query the prebuilt index:

```bash
python3 scripts/query.py "How should I think about CUDA memory performance?"
python3 scripts/query.py "What does __syncthreads guarantee?"
python3 scripts/query.py "When should I use streams and async copies?" --top-k 10
```

Return JSON for another program or agent:

```bash
python3 scripts/query.py "Explain occupancy at a high level" --json
```

## Rebuild

The repo ships with generated chunks and index files. Rebuild them from the Markdown source with:

```bash
python3 scripts/chunk.py
python3 scripts/build_index.py
python3 -m unittest discover -s tests
```

## Files

- `docs/cuda-programming-guide.pdf`: source PDF downloaded from NVIDIA.
- `docs/cuda-programming-guide.md`: full Markdown text extraction with page boundaries.
- `data/chunks.jsonl`: page-aware content chunks.
- `data/sections.jsonl`: guide outline records for broad retrieval.
- `index/cuda-guide-tfidf.joblib`: local vector index.
- `scripts/query.py`: query CLI.
- `skills/cuda-programming-guide/SKILL.md`: pure skill instructions for agents.

## How It Works

The query CLI uses a local vector index built with scikit-learn TF-IDF vectors:

- Word n-grams help broad CUDA concept queries.
- Character n-grams help exact CUDA names, APIs, and intrinsics.
- Section records help broad questions route into the right part of the guide.
- Query expansion adds CUDA-specific terms for common broad topics like memory, performance, streams, synchronization, and occupancy.

Each result includes the source file, guide page, heading path, score, and excerpt.

## Source

Source PDF:

https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf

The CUDA Programming Guide content is owned by NVIDIA. The repository code and indexing scripts are separate from NVIDIA's documentation rights.
