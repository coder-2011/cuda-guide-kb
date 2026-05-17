# Agent Setup Prompt

Use this prompt with an agent when you want it to set up and use this CUDA guide knowledge base.

```text
You are working with the CUDA Programming Guide knowledge base:

https://github.com/coder-2011/cuda-guide-kb

Your job is to make the KB usable in the current environment and then use it to answer CUDA questions with guide-grounded evidence.

Setup rules:
- Keep the setup portable.
- Do not create a virtual environment unless I explicitly ask for one.
- Use the Python environment that already makes sense for this machine or project.
- Before installing anything, check whether the dependencies are already present.
- If dependencies are missing, choose the least disruptive install method available for this environment: pip, uv pip, conda, system image, or the current project package manager.
- The required Python packages are listed in requirements.txt.
- Do not add an MCP server. This KB is file-and-script based.

Setup steps:
1. Clone the repo if it is not already present:

   git clone https://github.com/coder-2011/cuda-guide-kb.git
   cd cuda-guide-kb

2. Check dependency availability:

   python3 -c 'import importlib.util; mods=["joblib","numpy","sklearn","scipy"]; print("missing:", [m for m in mods if importlib.util.find_spec(m) is None])'

3. If dependencies are missing, install them in the most appropriate way for the current environment. A normal fallback is:

   python3 -m pip install -r requirements.txt

4. Verify the KB works:

   python3 scripts/query.py "How should I think about CUDA memory performance?" --top-k 4
   python3 scripts/query.py "What does __syncthreads guarantee?" --top-k 4
   python3 -m unittest discover -s tests

How to use the KB:
- For broad CUDA questions, query broadly and request more results:

  python3 scripts/query.py "What parts of the CUDA model matter most for kernel optimization?" --top-k 10

- For exact APIs, intrinsics, or symbols, include the exact name:

  python3 scripts/query.py "What does cudaMemcpyAsync do?" --top-k 8
  python3 scripts/query.py "What does __syncthreads guarantee?" --top-k 8

- For machine-readable retrieval, use JSON:

  python3 scripts/query.py "Explain occupancy at a high level" --json --top-k 8

Answering rules:
- Query the KB before answering factual CUDA details.
- Read several returned excerpts, not only the first result.
- Prefer chunk results for detailed claims.
- Use section results as a map for broad questions.
- Cite guide page numbers from the returned results.
- If the KB does not contain enough evidence, say so and separate outside reasoning from guide-grounded facts.
- Keep iterating on the query if the first query is too narrow or too vague.
```
