# Agents Setup

Use this file as the setup handoff for any agent that should answer CUDA questions from this repository.

## Purpose

This repo is a local, queryable knowledge base built from the NVIDIA CUDA Programming Guide.

Agents should use it before answering factual CUDA programming questions. The KB includes:

- source guide PDF and Markdown in `docs/`
- page-aware chunks and section records in `data/`
- a local TF-IDF index in `index/cuda-guide-tfidf.joblib`
- a query CLI at `scripts/query.py`
- an optional skill at `skills/cuda-programming-guide/`

Do not add an MCP server. This KB is intentionally local files plus scripts.

## Clone Or Update

If the repo is not present:

```bash
git clone https://github.com/coder-2011/cuda-guide-kb.git
cd cuda-guide-kb
```

If the repo is already present:

```bash
cd cuda-guide-kb
git pull
```

Run all commands from the repo root.

## Install The Agent Skill

The bundled skill lives at:

```text
skills/cuda-programming-guide/
```

If the runtime uses a skills directory, symlink it:

```bash
SKILL_HOME="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$SKILL_HOME"
ln -sfn "$PWD/skills/cuda-programming-guide" "$SKILL_HOME/cuda-programming-guide"
```

If symlinks are not supported, copy `skills/cuda-programming-guide` into the runtime's skills directory. If repo-local skills are supported, point the runtime directly at `skills/cuda-programming-guide/`.

## Check Dependencies

Do not create a virtual environment unless explicitly asked. First check whether the current Python already has the dependencies:

```bash
python3 - <<'PY'
import importlib.util
mods = ["joblib", "numpy", "sklearn", "scipy"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("missing:", missing)
PY
```

If anything is missing, install the small requirements file with the least disruptive tool available:

```bash
python3 -m pip install -r requirements.txt
```

If system Python refuses package installation or the agent should avoid changing the environment, `scripts/query.py` will automatically re-run through `uv` when `uv` is installed. The explicit form is:

```bash
uv run --with joblib --with numpy --with scikit-learn==1.8.0 --with scipy python scripts/query.py "QUESTION HERE" --top-k 8
```

## Verify The KB

Run:

```bash
python3 scripts/query.py "How should I think about CUDA memory performance?" --top-k 4
python3 scripts/query.py "What does __syncthreads guarantee?" --top-k 4
python3 scripts/query.py "Explain occupancy at a high level" --json --top-k 4
python3 -m unittest discover -s tests
```

Expected behavior:

- Memory query returns memory, coalescing, bandwidth, shared memory, or related performance context.
- `__syncthreads` query returns thread block synchronization context.
- Occupancy query returns occupancy, active warps, resident blocks, or multiprocessor context.
- Tests pass.

## Query Rules

For factual CUDA questions, query before answering from memory:

```bash
python3 scripts/query.py "QUESTION HERE" --top-k 8
```

For structured output:

```bash
python3 scripts/query.py "QUESTION HERE" --json --top-k 8
```

## Query Strategy

For broad questions, keep the wording broad and request several results:

```bash
python3 scripts/query.py "How should I think about CUDA memory performance?" --top-k 10
python3 scripts/query.py "Explain CUDA streams and asynchronous execution at a high level." --top-k 10
python3 scripts/query.py "What parts of the CUDA model matter most for kernel optimization?" --top-k 10
```

For exact APIs, symbols, or intrinsics, include the exact name:

```bash
python3 scripts/query.py "What does __syncthreads guarantee?" --top-k 8
python3 scripts/query.py "What does cudaMemcpyAsync do?" --top-k 8
python3 scripts/query.py "How does cudaMallocManaged relate to unified memory?" --top-k 8
```

If the first result set is weak, rerun once with related exact terms:

```bash
python3 scripts/query.py "CUDA synchronization thread block memory ordering barriers" --top-k 12
```

## Answering Rules

- Read multiple returned excerpts, not only the first result.
- Prefer chunk results for detailed claims.
- Use section results as navigation hints when useful.
- Cite returned guide page numbers for factual claims.
- Do not cite a page that did not appear in the query output.
- Separate guide-grounded facts from outside reasoning.
- If the KB does not contain enough evidence, say what is missing.

## Rebuild Only When Needed

Rebuild only if the source guide, chunker, or index logic changed:

```bash
python3 scripts/chunk.py
python3 scripts/build_index.py
python3 -m unittest discover -s tests
```
