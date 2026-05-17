---
name: cuda-programming-guide
description: Ground CUDA programming answers in the local NVIDIA CUDA Programming Guide knowledge base. Use for CUDA C++, kernel optimization, memory hierarchy, synchronization, streams, occupancy, execution model, memory model, CUDA APIs, CUDA intrinsics, and broad CUDA questions where the agent is unsure or should cite guide pages. Pure skill only; no MCP server. Query the repo CLI before answering.
---

# CUDA Programming Guide

Use the local CUDA guide index before answering CUDA questions from memory.

## Locate the Repo

Run commands from the repo root containing `scripts/query.py`. If needed:

```bash
cd /home/ubuntu/cuda-guide-kb
```

## Query First

For any factual CUDA answer, query before answering:

```bash
python3 scripts/query.py "QUESTION" --top-k 8
```

Use JSON when another tool or script will consume the results:

```bash
python3 scripts/query.py "QUESTION" --json --top-k 8
```

## Query Strategy

For broad questions, keep the user wording broad and ask for more results:

```bash
python3 scripts/query.py "How should I think about CUDA memory performance?" --top-k 10
python3 scripts/query.py "What parts of the CUDA model matter most for kernel optimization?" --top-k 10
```

For exact API, intrinsic, or symbol questions, include the exact name:

```bash
python3 scripts/query.py "What does __syncthreads guarantee?" --top-k 8
python3 scripts/query.py "What is cudaMemcpyAsync used for?" --top-k 8
python3 scripts/query.py "How does cudaMallocManaged relate to unified memory?" --top-k 8
```

If the first result set is weak, rerun once with a broader phrase and `--top-k 12`:

```bash
python3 scripts/query.py "CUDA synchronization thread block memory ordering barriers" --top-k 12
```

## Answering

- Read several returned excerpts, not only the first one.
- Prefer chunks over bare section records for detailed claims.
- Use section records as a map when the question is broad.
- Cite guide page numbers when making factual CUDA claims.
- Separate guide-grounded facts from outside reasoning.
- If the local index does not contain enough evidence, say so plainly.

## Common Failure Modes

- Do not answer CUDA details from memory before querying.
- Do not treat query expansion terms as answers; they only help retrieval.
- Do not cite a page unless that page appears in the returned results.
- Do not add an MCP server. This skill is intentionally file-and-script only.
