# CUDA Programming Guide KB

A local knowledge base for you and your agents, all abt cuda! Alot of CUDA information is well documented, but sparsely so, so LLMs don't end up training on the esoteric hard parts. The skill included is built to make agents make tons of queries to the kb so that they never miss information. CUDA isn't as forgiving either, so double checking is very useful. 

There is an agent skill included to use the knowledge base, and PROMPT.md includes a prompt you can copy paste for an agent to set up the knowledge base

I use this alot, I hope you find it useful :)


## Setup Guide

## How It Works

This repo is intentionally simple. It is a local retrieval system over the CUDA Programming Guide, not a hosted service and not an MCP server.

### 1. Source documents

The source PDF is stored at `docs/cuda-programming-guide.pdf`.

The Markdown file at `docs/cuda-programming-guide.md` is a text extraction of the full PDF. It preserves page boundaries as `## Page N` headings and includes the PDF outline near the top. Page boundaries are important because query results can point back to the guide page that produced the match.

### 2. Chunking

`scripts/chunk.py` reads the Markdown file and writes two generated data files:

- `data/chunks.jsonl`: page-aware text chunks from the guide body.
- `data/sections.jsonl`: section records parsed from the PDF outline.

Chunks are the main answer context. Section records are lighter routing hints that help broad questions land near the right chapter before a person or model reads the matching excerpts.

Each chunk keeps metadata:

- source Markdown path
- source PDF path
- NVIDIA source URL
- page number
- heading path
- searchable text

### 3. Local index

`scripts/build_index.py` builds `index/cuda-guide-tfidf.joblib` from the chunk and section records.

The index uses scikit-learn TF-IDF vectors:

- Word n-grams catch normal CUDA concepts and phrases.
- Character n-grams help with exact CUDA names, symbols, and APIs such as `__syncthreads`, `cudaMemcpyAsync`, `threadIdx.x`, and `cudaMallocManaged`.
- Chunk records provide concrete excerpt text.
- Section records improve recall for broad questions.

This is not an embedding API and it does not require network access at query time.

### 4. Querying

`scripts/query.py` loads the prebuilt index, expands the user's question slightly, scores all records, and prints the top matches.

The query path is:

1. Read the user's question.
2. Add a small amount of CUDA-specific query expansion for broad topics like memory, performance, streams, synchronization, occupancy, and the execution model.
3. Vectorize the expanded query with the same word and character TF-IDF vectorizers used during indexing.
4. Score all guide chunks and section records.
5. Add a small exact-symbol boost when the question contains CUDA-style symbols or APIs.
6. Give broad questions a small preference for section records so the results include useful map-like entry points.
7. Return ranked matches with page numbers, heading paths, scores, and excerpts.

The query expansion is not an answer key. It does not contain canned CUDA explanations. It only adds related search terms so vague questions like "How should I think about CUDA memory performance?" can find guide sections about coalescing, shared memory, bandwidth, bank conflicts, and unified memory.

### 5. Agent usage

Agents should use the skill in `skills/cuda-programming-guide/` or follow `AGENTS.md`.

For broad questions, ask the broad question directly:

```bash
python3 scripts/query.py "How should I think about CUDA memory performance?" --top-k 10
```

For exact API or intrinsic questions, include the exact name:

```bash
python3 scripts/query.py "What does __syncthreads guarantee?" --top-k 8
```

For programmatic use, request JSON:

```bash
python3 scripts/query.py "Explain occupancy at a high level" --json --top-k 8
```

Each result includes the source file, guide page, heading path, score, and excerpt. A model should synthesize from the returned excerpts and cite the guide pages when making factual claims.

Source PDF:

https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf
