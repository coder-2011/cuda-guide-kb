# Agent Instructions

Use this repo when you need answers grounded in the NVIDIA CUDA Programming Guide.

## Install Skill

The bundled agent skill lives at `skills/cuda-programming-guide/`.

If the runtime uses a skills directory, install it with a symlink:

```bash
SKILL_HOME="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$SKILL_HOME"
ln -sfn "$PWD/skills/cuda-programming-guide" "$SKILL_HOME/cuda-programming-guide"
```

If symlinks are not supported, copy `skills/cuda-programming-guide` into the runtime's skills directory. If repo-local skills are supported, point the runtime directly at that folder.

## Query First

For CUDA programming, optimization, memory, kernel execution, synchronization, streams, occupancy, or CUDA API questions, query the local index before relying on model memory:

```bash
python3 scripts/query.py "QUESTION HERE" --top-k 8
```

Use JSON when you need machine-readable retrieval output:

```bash
python3 scripts/query.py "QUESTION HERE" --json --top-k 8
```

## Broad Questions

For broad questions, ask broad questions directly. The query tool is tuned to route these into section-level guide records and relevant chunks:

```bash
python3 scripts/query.py "How should I think about CUDA memory performance?"
python3 scripts/query.py "Explain CUDA streams and asynchronous execution at a high level."
python3 scripts/query.py "What parts of the CUDA model matter most for kernel optimization?"
```

Synthesize answers from several returned sections/chunks. Include page references when giving factual claims.

## Exact Questions

For exact APIs, symbols, or intrinsics, include the exact name:

```bash
python3 scripts/query.py "What does __syncthreads guarantee?"
python3 scripts/query.py "What is cudaMemcpyAsync used for?"
python3 scripts/query.py "How does cudaMallocManaged relate to unified memory?"
```

Prefer the highest-ranked chunk that contains the exact symbol.

## Rebuild

If docs or chunking change:

```bash
python3 scripts/chunk.py
python3 scripts/build_index.py
python3 -m unittest discover -s tests
```

Do not add an MCP server. This repo is intentionally a pure file-and-script knowledge base.
