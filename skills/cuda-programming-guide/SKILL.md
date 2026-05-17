---
name: cuda-programming-guide
description: Use when answering CUDA programming, CUDA C++, kernel optimization, memory hierarchy, synchronization, streams, occupancy, or CUDA API questions that should be grounded in the NVIDIA CUDA Programming Guide. This is a pure skill with no MCP server; use the bundled repo query CLI.
---

# CUDA Programming Guide

Use the local CUDA guide index before answering CUDA questions from memory.

## Workflow

1. From the repo root, run:

   ```bash
   python3 scripts/query.py "QUESTION" --top-k 8
   ```

2. For broad questions, keep the user's wording broad. The index includes section records for routing:

   ```bash
   python3 scripts/query.py "How should I think about CUDA memory performance?" --top-k 10
   ```

3. For API, intrinsic, or symbol questions, include the exact name:

   ```bash
   python3 scripts/query.py "What does __syncthreads guarantee?" --top-k 8
   ```

4. Read the returned excerpts and page numbers. Answer from the retrieved context and cite guide pages when possible.

5. If results look too narrow, rerun with a more general query and `--top-k 12`.

## Output Modes

Markdown:

```bash
python3 scripts/query.py "Explain occupancy at a high level"
```

JSON:

```bash
python3 scripts/query.py "Explain occupancy at a high level" --json
```

## Grounding Rule

For factual CUDA claims, prefer the retrieved guide chunks over general model knowledge. If the guide results do not answer the question, say that the local CUDA guide index did not contain enough evidence and then clearly separate any outside reasoning.
