# Agent Setup Prompt

Paste this into the agent that should set up and use the CUDA Programming Guide Knowledge Base.

```text
You are setting up and using the CUDA Programming Guide Knowledge Base:

https://github.com/coder-2011/cuda-guide-kb

Make the KB usable in the current environment, then use it to answer CUDA questions with evidence from the guide.

Rules:
- Do not create a virtual environment unless explicitly asked.
- Use the Python environment that already fits the machine or project.
- Inspect the repo before acting. Read the README, AGENTS.md, scripts, tests, and requirements as needed.
- If these instructions are incomplete or inaccurate, figure out the correct path from the codebase and take the necessary action.
- Do not add an MCP server. This KB is local files plus scripts.

Setup:
1. Clone or update the repo.

   git clone https://github.com/coder-2011/cuda-guide-kb.git
   cd cuda-guide-kb

   If it already exists, enter it and pull the latest changes.

2. Check dependencies before installing anything.

   python3 -c 'import importlib.util; mods=["joblib","numpy","sklearn","scipy"]; print("missing:", [m for m in mods if importlib.util.find_spec(m) is None])'

3. If packages are missing, install `requirements.txt` with the least disruptive tool available in the environment.

   python3 -m pip install -r requirements.txt

4. Verify the KB works.

   python3 scripts/query.py "How should I think about CUDA memory performance?" --top-k 4
   python3 scripts/query.py "What does __syncthreads guarantee?" --top-k 4
   python3 -m unittest discover -s tests

Use:
- Query the KB before answering factual CUDA questions.
- Read multiple returned results.
- For broad questions, use broad queries with `--top-k 8` or higher.
- For API, intrinsic, or symbol questions, include the exact name.
- Use `--json` when structured retrieval output is useful.
- Cite guide page numbers from the returned results.
- If the KB is not enough, say what is guide-grounded and what is outside reasoning.

Example queries:

python3 scripts/query.py "What parts of the CUDA model matter most for kernel optimization?" --top-k 10
python3 scripts/query.py "What does cudaMemcpyAsync do?" --top-k 8
python3 scripts/query.py "Explain occupancy at a high level" --json --top-k 8
```
