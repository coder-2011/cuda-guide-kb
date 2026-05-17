# CUDA Programming Guide Knowledge Base

A local knowledge base for you and your agents, all abt cuda! Alot of CUDA information is well documented, but sparsely so, so LLMs don't end up training on the esoteric hard parts. The skill included is built to make agents make tons of queries to the kb so that they never miss information. CUDA isn't as forgiving either, so double checking is very useful. 

There is an agent skill included to use the knowledge base, and PROMPT.md includes a prompt you can copy paste for an agent to set up the knowledge base

I use this alot, I hope you find it useful :)


## Setup Guide

This repo ships with the docs, chunks, and index already built. Setup is just: get the repo, make sure the small Python dependency set is available, then run a query.

### 1. Get the repo

```bash
git clone https://github.com/coder-2011/cuda-guide-kb.git
cd cuda-guide-kb
```

If the repo is already present:

```bash
cd cuda-guide-kb
git pull
```

### 2. Use your existing Python environment

Do not create a virtual environment unless you personally want one. This repo is meant to work inside whatever environment is natural for the machine or project.

The query tool only needs:

- `joblib`
- `numpy`
- `scikit-learn`
- `scipy`

Check whether they are already available:

```bash
python3 - <<'PY'
import importlib.util
mods = ["joblib", "numpy", "sklearn", "scipy"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("missing:", missing)
PY
```

If nothing is missing, skip installation.

### 3. Install only if needed

If dependencies are missing, install them in the way that best fits your environment.

Plain pip:

```bash
python3 -m pip install -r requirements.txt
```

uv:

```bash
uv pip install -r requirements.txt
```

Conda, system images, prebuilt dev containers, or an existing project package manager are fine too. The repo does not care how the packages arrive; it only needs `python3 scripts/query.py ...` to import them.

### 4. Verify it works

```bash
python3 scripts/query.py "How should I think about CUDA memory performance?" --top-k 4
python3 scripts/query.py "What does __syncthreads guarantee?" --top-k 4
python3 -m unittest discover -s tests
```

The first query should return memory-performance sections like coalesced global memory access, shared memory bank conflicts, or unified memory performance hints. The second should return the thread block synchronization section around `__syncthreads`.

### 5. Query the KB

Broad question:

```bash
python3 scripts/query.py "What parts of the CUDA model matter most for kernel optimization?" --top-k 10
```

Exact API or intrinsic:

```bash
python3 scripts/query.py "What does cudaMemcpyAsync do?" --top-k 8
python3 scripts/query.py "What does __syncthreads guarantee?" --top-k 8
```

JSON output for agents or scripts:

```bash
python3 scripts/query.py "Explain occupancy at a high level" --json --top-k 8
```

### 6. Use the agent skill

The bundled skill lives at:

```text
skills/cuda-programming-guide/
```

Use it in whichever way your agent runtime supports:

- Point the agent at this repo and tell it to use `skills/cuda-programming-guide/`.
- Copy the skill folder into your normal skills directory.
- Symlink the skill folder into your normal skills directory.

For a pasteable agent instruction, use `PROMPT.md`.

### 7. Rebuild only when needed

You do not need to rebuild the index for normal use. Rebuild only if you change the Markdown doc, chunker, or indexing logic:

```bash
python3 scripts/chunk.py
python3 scripts/build_index.py
python3 -m unittest discover -s tests
```

This regenerates `data/chunks.jsonl`, `data/sections.jsonl`, and `index/cuda-guide-tfidf.joblib`.

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
